"""V32 continuous power-system LP, extended from the frozen V31 model.
All powers MW, energies MWh; no ENS profile is used as an exogenous task.
Hydrogen is an independent, priced, exogenously available fuel supply scenario.
Optional zone electricity-output budgets and event-only capacity derating do
not represent hydrogen production, transport, or chronological fuel inventory.
iDSR preserves each supplied window's energy, including circular year-edge IDs.
There is no export to omitted external regions: net imports include external
injection, while net exports subtract it. Gross imports include this injection.
"""
import time,warnings,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix,vstack
warnings.filterwarnings('ignore',category=OptimizeWarning)

class Builder:
 def __init__(self):self.n=0;self.blocks={};self.lo=[];self.hi=[];self.cost=[];self.er=[];self.ec=[];self.ev=[];self.eb=[];self.ur=[];self.uc=[];self.uv=[];self.ub=[];self.ne=0;self.nu=0
 def block(self,name,shape,upper=np.inf,lower=0,cost=0):
  ids=np.arange(self.n,self.n+int(np.prod(shape))).reshape(shape);self.n+=ids.size;self.blocks[name]=ids
  self.lo.append(np.broadcast_to(lower,shape).ravel().copy());self.hi.append(np.broadcast_to(upper,shape).ravel().copy());self.cost.append(np.broadcast_to(cost,shape).ravel().copy());return ids
 def rows(self,terms,rhs,eq=True):
  rhs=np.asarray(rhs);n=rhs.size;start=self.ne if eq else self.nu
  for ids,coef in terms:
   ids=np.asarray(ids);co=np.broadcast_to(coef,ids.shape).ravel();rr=np.broadcast_to(np.arange(n).reshape(rhs.shape),ids.shape).ravel() if ids.shape==rhs.shape else np.repeat(np.arange(n),ids.size//n)
   (self.er if eq else self.ur).append(rr+start);(self.ec if eq else self.uc).append(ids.ravel());(self.ev if eq else self.uv).append(co)
  (self.eb if eq else self.ub).append(rhs.ravel())
  if eq:self.ne+=n
  else:self.nu+=n
 def matrix(self):
  def sp(r,c,v,n):return coo_matrix((np.concatenate(v),(np.concatenate(r),np.concatenate(c))),shape=(n,self.n)).tocsr() if v else None
  return sp(self.er,self.ec,self.ev,self.ne),np.concatenate(self.eb),sp(self.ur,self.uc,self.uv,self.nu),np.concatenate(self.ub) if self.ub else None

def idsr_inputs(d,T,Z):
 """Validate the explicit flexible-consumption contract, without choosing windows."""
 keys=['idsr_baseline','idsr_zone','idsr_upper','idsr_window']
 if not any(k in d for k in keys):
  return np.zeros((T,0)),np.zeros(0,dtype=int),np.zeros((T,0)),np.zeros((T,0),dtype=int)
 if not all(k in d for k in keys):raise ValueError('All four iDSR input fields are required together')
 base=np.asarray(d['idsr_baseline'],float);upper=np.asarray(d['idsr_upper'],float)
 zone=np.asarray(d['idsr_zone']);window=np.asarray(d['idsr_window'])
 if base.ndim!=2 or base.shape[0]!=T or upper.shape!=base.shape or window.shape!=base.shape:
  raise ValueError('iDSR baseline, upper and window must have equal (T,K) shapes')
 if zone.shape!=(base.shape[1],) or zone.dtype.kind not in 'iu' or ((zone<0)|(zone>=Z)).any():
  raise ValueError('iDSR zone must be a valid integer K-vector')
 if window.dtype.kind not in 'iu' or (window<0).any():raise ValueError('iDSR window IDs must be nonnegative integers')
 if not np.isfinite(base).all() or not np.isfinite(upper).all() or (base<0).any() or (upper<base-1e-8).any():
  raise ValueError('iDSR requires finite nonnegative baseline <= hourly consumption upper bound')
 return base,zone.astype(int),upper,window

def hydrogen_inputs(d,T,Z):
 """Validate optional H2 supply scenarios; one hourly step equals one MWh/MW."""
 cap=np.broadcast_to(np.asarray(d.get('hydrogen_cap',0.),float),(T,Z))
 if not np.isfinite(cap).all() or (cap<0).any():raise ValueError('Hydrogen capacity must be finite and nonnegative')
 budget=None
 if 'hydrogen_annual_budget_MWh' in d:
  budget=np.asarray(d['hydrogen_annual_budget_MWh'],float)
  if budget.shape!=(Z,) or not np.isfinite(budget).all() or (budget<0).any():
   raise ValueError('hydrogen_annual_budget_MWh must be a finite nonnegative Z-vector of electricity-output MWh')
 derating=np.asarray(d.get('hydrogen_event_derating',0.),float)
 if derating.ndim!=0 or not np.isfinite(derating) or not 0<=derating<=1:
  raise ValueError('hydrogen_event_derating must be a finite scalar in [0,1]')
 derating=float(derating)
 if 'hydrogen_event_derating' in d:
  event=np.asarray(d['event'])
  if event.shape!=(T,) or not np.isin(event,[0,1]).all():
   raise ValueError('Hydrogen event derating requires a T-hour boolean or binary event mask')
  if derating:cap=cap*(1-derating*event[:,None])
 scope=dict(fuel_supply='independent priced exogenous fuel availability',
  annual_electricity_budget_applied=budget is not None,
  annual_budget_MWh_by_zone=None if budget is None else budget.tolist(),
  event_capacity_derating=derating,gas_cut_derates_hydrogen=False,
  hydrogen_inventory_modeled=False,time_step_hours=1,
  budget_definition='sum of hydrogen electricity output over all modeled hourly steps, separately by zone')
 return cap,budget,scope

def hydrogen_scope_text(scope):
 budget='zone annual electricity-output budget applied' if scope['annual_electricity_budget_applied'] else 'no annual electricity-output budget'
 return ('independent priced exogenous fuel availability; '+budget+
  '; event-only capacity derating='+str(scope['event_capacity_derating'])+
  '; gas_cut does not derate hydrogen; hydrogen inventory is not modeled')

def interface_constraints(b,d,ext,flow,T,Z):
 """Separate net/gross limits. Only legacy inputs retain the V28 gross mapping."""
 names=['net_import_cap','net_export_cap','gross_import_cap','gross_export_cap']
 explicit=any(k in d for k in names)
 if explicit and not all(k in d for k in names):raise ValueError('Provide all four net/gross interface limits together')
 inf=np.full((T,Z),np.inf)
 caps={k:np.broadcast_to(np.asarray(d[k],float),(T,Z)) for k in names} if explicit else {
  'net_import_cap':inf,'net_export_cap':inf,
  'gross_import_cap':np.broadcast_to(np.asarray(d.get('import_cap',inf),float),(T,Z)),
  'gross_export_cap':np.broadcast_to(np.asarray(d.get('export_cap',inf),float),(T,Z))}
 specs=[('net_import_cap',1,1,-1),('net_export_cap',-1,-1,1),
        ('gross_import_cap',1,1,0),('gross_export_cap',0,0,1)]
 counts={}
 for name,external_sign,inward_sign,outward_sign in specs:
  cap=caps[name]
  if np.isnan(cap).any() or (cap<0).any():raise ValueError(name+' must be nonnegative, with +inf allowed')
  finite=np.isfinite(cap);counts[name]=int(finite.sum())
  if not finite.any():continue
  start=b.nu
  b.rows([(ext[finite],external_sign)] if external_sign else [],cap[finite],False)
  rowids=np.full((T,Z),-1,dtype=np.int64);rowids[finite]=np.arange(finite.sum())+start
  for nodes,sign in [(d['edge_to'],inward_sign),(d['edge_from'],outward_sign)]:
   if not sign:continue
   rr=rowids[:,nodes];keep=rr>=0
   b.ur.append(rr[keep]);b.uc.append(flow[keep]);b.uv.append(np.full(int(keep.sum()),sign))
 return dict(mode='explicit_net_and_gross' if explicit else 'legacy_V28_gross_limits',
             finite_rows=counts,external_exports_modeled=False,
             net_import_definition='internal_in + external_import - internal_out',
             net_export_definition='internal_out - internal_in - external_import')

def build_problem(d,tech='biomethane',power=None,energy=None,gas_cut=0.,external_availability=.5,design=False,costs=None,target_eens=None,zero_hours=None,allow_boost_hours=None):
 T,Z=d['demand'].shape;H=len(d['hydro_zone']);E=len(d['edge_from']);D=len(d['dsr_zone']);q=d['q'];eta=float(d['eta_bio']);beta=float(d['beta']);c=costs or {};b=Builder();idx=np.arange(T)
 if T%24:raise ValueError('The explicit daily DSR model requires a whole number of days')
 ibase,izone,iupper,iwindow=idsr_inputs(d,T,Z)
 ibase_node=np.zeros((T,Z));iextra_node=np.zeros((T,Z))
 for k,z in enumerate(izone):
  ibase_node[:,z]+=ibase[:,k];iextra_node[:,z]+=iupper[:,k]-ibase[:,k]
 if (d['demand']-ibase_node < -1e-7).any():raise ValueError('iDSR baseline cannot exceed the node demand containing it')
 gascap=d['gas_cap']*(1-float(gas_cut)*d['event'][:,None]);p=np.asarray(q*eta if power is None else power);s=np.asarray(q*6 if energy is None else energy)
 if 'gas_relief' in d:gascap=gascap+np.minimum(np.asarray(d['gas_relief']),d['gas_cap']-gascap)
 gas=b.block('gas',(T,Z),gascap,cost=c.get('system_gas_eur_per_MWh_el',90.))
 hydrogen_cap,hydrogen_budget,hydrogen_scope=hydrogen_inputs(d,T,Z)
 hydrogen=b.block('hydrogen',(T,Z),hydrogen_cap,cost=c.get('system_hydrogen_eur_per_MWh_el',d.get('hydrogen_cost_eur_per_MWh',154.496)))
 if hydrogen_budget is not None:b.rows([(hydrogen.T,1)],hydrogen_budget,False)
 split_thermal='thermal_cap' in d
 oth=b.block('other_thermal',(T,Z),np.zeros((T,Z)) if split_thermal else d['other_thermal_cap'],cost=c.get('system_other_eur_per_MWh_el',25.))
 thermal=b.block('thermal_dispatch',d['thermal_cap'].shape,d['thermal_cap'],cost=d['thermal_cost']) if split_thermal else np.empty((T,0),dtype=int)
 firm=b.block('firm_capacity',(T,Z),d['firm_capacity_MW'],cost=0) if 'firm_capacity_MW' in d else None
 ext=b.block('external_import',(T,Z),d['external_cap']*external_availability,cost=c.get('system_external_eur_per_MWh',100.))
 spill=b.block('curtailment',(T,Z),np.maximum(0,d['vre']+d['ror']+d['fixed_other']-d['base']))
 uns=b.block('unserved',(T,Z),d['demand']+iextra_node,cost=0)
 hg=b.block('hydro_generation',(T,H),d['hydro_gen_cap'],cost=.001)
 hp=b.block('hydro_pumping',(T,H),d['hydro_pump_cap'],cost=.001)
 hs=b.block('hydro_inventory',(T,H),d['hydro_max'],d['hydro_min'])
 hw=b.block('hydro_spill',(T,H),cost=.0001)
 bc=b.block('existing_battery_charge',(T,Z),d['battery_power'],cost=.001)
 bd=b.block('existing_battery_discharge',(T,Z),d.get('battery_discharge_power',d['battery_power']),cost=.001)
 bs=b.block('existing_battery_inventory',(T,Z),d['battery_energy'])
 flow=b.block('flow',(T,E),d['edge_cap'],cost=.0001)
 ds=b.block('dsr',(T,D),d['dsr_cap'],cost=c.get('system_dsr_eur_per_MWh',2000.))
 iconsume=b.block('idsr_consumption',ibase.shape,iupper)
 idsr_groups=[]
 for k in range(ibase.shape[1]):
  labels,inverse=np.unique(iwindow[:,k],return_inverse=True)
  rhs=np.bincount(inverse,weights=ibase[:,k],minlength=len(labels))
  start=b.ne;b.rows([],rhs)
  b.er.append(start+inverse);b.ec.append(iconsume[:,k]);b.ev.append(np.ones(T))
  idsr_groups.append(dict(column=k,labels=labels,inverse=inverse,baseline_MWh=rhs))
 if tech=='none':p=np.zeros(Z)
 if design:
  P=b.block('design_power',(Z,),c.get('max_power_MW',100*q+100),cost=c.get('power_eur_per_MW_year',0))
  S=b.block('design_energy',(Z,),c.get('max_energy_MWh',48*q+100),cost=c.get('energy_eur_per_MWh_year',0))
 else:P=S=None
 if tech in ['biomethane','natural_gas','none']:
  boost=b.block('boost',(T,Z),np.inf if design else p,cost=c.get('boost_eur_per_MWh_el',1e6))
  if tech in ['biomethane','none']:
   gs=b.block('gas_inventory',(T,Z),np.inf if design else s)
   kappa=float(d.get('compression_kappa',.03))
   cmax=np.minimum(2*q,np.maximum(eta*q-d['base'],0)/kappa) if kappa>0 else 2*q
   gc=b.block('gas_charge',(T,Z),cmax,cost=c.get('gas_charge_eur_per_MWh_fuel',.0001))
   withdraw_limit=np.maximum(12*q,beta*q+p/eta) if bool(d.get('relaxed_frontier',False)) else 12*q
   gw=b.block('gas_withdraw',(T,Z),withdraw_limit,cost=.0001)
   gu=b.block('other_gas_use',(T,Z),cost=0)
   prev=np.roll(gs,1,axis=0)
   b.rows([(gs,1),(prev,-1),(gc,-1),(gw,1)],np.zeros((T,Z)))
   production=d.get('q_hourly',np.broadcast_to(q,(T,Z)))
   b.rows([(boost,1/eta),(gc,1+kappa/eta),(gw,-1),(gu,1)],production-d['base'][None,:]/eta)
   if design:
    b.rows([(gs,1),(np.broadcast_to(S,(T,Z)),-1)],np.zeros((T,Z)),False)
    b.rows([(gs[-1],1),(S,-.5)],np.zeros(Z))
   else:b.rows([(gs[-1],1)],.5*s)
   # Explicit finite net injection and withdrawal limits, proportional to production units.
   b.rows([(gs,1),(prev,-1)],np.broadcast_to(2*q,(T,Z)),False)
   b.rows([(prev,1),(gs,-1)],np.broadcast_to(withdraw_limit,(T,Z)),False)
  else:
   # Purchased NG uses the same existing pressure-sensitive supply condition.
   factor=1-gas_cut*d['event'][:,None]
   if design:b.rows([(boost,1),(np.broadcast_to(P,(T,Z)),-factor)],np.zeros((T,Z)),False)
   else:b.rows([(boost,1)],p[None,:]*np.ones((T,1))*factor,False)
   if design:b.rows([(S,1)],np.zeros(Z))
  if design:b.rows([(boost,1),(np.broadcast_to(P,(T,Z)),-1)],np.zeros((T,Z)),False)
  newcharge=None
 elif tech=='battery':
  newcharge=b.block('new_battery_charge',(T,Z),np.inf if design else p,cost=.001)
  boost=b.block('boost',(T,Z),np.inf if design else p,cost=c.get('boost_eur_per_MWh_el',.001))
  ss=b.block('new_battery_inventory',(T,Z),np.inf if design else s)
  eta_b=float(c.get('battery_eta',float(d['battery_eta'])))
  b.rows([(ss,1),(np.roll(ss,1,axis=0),-1),(newcharge,-eta_b),(boost,1/eta_b)],np.zeros((T,Z)))
  if design:
   for v in [newcharge,boost]:b.rows([(v,1),(np.broadcast_to(P,(T,Z)),-1)],np.zeros((T,Z)),False)
   b.rows([(ss,1),(np.broadcast_to(S,(T,Z)),-1)],np.zeros((T,Z)),False)
   b.rows([(ss[-1],1),(S,-.5)],np.zeros(Z))
  else:b.rows([(ss[-1],1)],.5*s)
 else:raise ValueError(tech)
 # Nodal balances assemble each technology only in its own node.
 rhs=d['demand']-d['vre']-d['ror']-d['fixed_other']-ibase_node
 terms=[(gas,1),(hydrogen,1),(oth,1),(ext,1),(spill,-1),(uns,1),(bc,-1),(bd,1),(boost,1)]
 if newcharge is not None:terms.append((newcharge,-1))
 if firm is not None:terms.append((firm,1))
 start=b.ne;balance_start=start;b.rows(terms,rhs)
 for arr,zone,sign in [(thermal,d.get('thermal_zone',np.zeros(0,dtype=int)),1),(hg,d['hydro_zone'],1),(hp,d['hydro_zone'],-1),(ds,d['dsr_zone'],1),(flow,d['edge_to'],1),(flow,d['edge_from'],-1),(iconsume,izone,-1)]:
  b.er.append((np.arange(T)[:,None]*Z+zone[None,:]+start).ravel());b.ec.append(arr.ravel());b.ev.append(np.full(arr.size,sign))
 # ENS cannot exceed actual node demand after flexible consumption is rescheduled.
 if ibase.shape[1]:
  start=b.nu;b.rows([(uns,1)],d['demand']-ibase_node,False)
  b.ur.append((np.arange(T)[:,None]*Z+izone[None,:]+start).ravel())
  b.uc.append(iconsume.ravel());b.uv.append(-np.ones(iconsume.size))
 # Hydro reservoirs preserve reported start and end levels; no gas-style annual free stock.
 hrhs=d['hydro_inflow'].copy();hrhs[0]+=d['hydro_initial']
 b.rows([(hs,1),(hg,1),(hp,-float(d['pump_eta'])),(hw,1)],hrhs)
 st=b.ne-T*H;b.er.append((st+np.arange(H,T*H)).ravel());b.ec.append(hs[:-1].ravel());b.ev.append(np.full((T-1)*H,-1.))
 b.rows([(hs[-1],1)],d['hydro_terminal'])
 eb=float(d['battery_eta']);b.rows([(bs,1),(np.roll(bs,1,axis=0),-1),(bc,-eb),(bd,1/eb)],np.zeros((T,Z)))
 b.rows([(bs[-1],1)],.5*d['battery_energy'])
 # DSR daily energy limits and directional zone interface constraints.
 if D:b.rows([(ds.reshape(T//24,24,D).transpose(0,2,1).reshape(-1,24),1)],np.tile(d['dsr_cap']*d['dsr_hours'],T//24),False)
 interface_scope=interface_constraints(b,d,ext,flow,T,Z)
 if target_eens is not None:b.rows([(uns.ravel()[None,:],1)],np.array([target_eens]),False)
 if zero_hours is not None:
  zh=np.asarray(zero_hours,dtype=bool)
  if zh.any():b.rows([(uns[zh],1)],np.zeros((int(zh.sum()),Z)))
 eq,rhs,ub,budget=b.matrix();bounds=np.column_stack([np.concatenate(b.lo),np.concatenate(b.hi)]);cost=np.concatenate(b.cost)
 if allow_boost_hours is not None:
  allowed=np.asarray(allow_boost_hours,dtype=bool)
  if allowed.shape!=(T,):raise ValueError('allow_boost_hours must be a T-hour boolean mask')
  bounds[boost[~allowed].ravel(),1]=0
 return dict(builder=b,eq=eq,rhs=rhs,ub=ub,budget=budget,bounds=bounds,cost=cost,tech=tech,power=p,energy=s,design=design,data=d,gas_cut=gas_cut,external_availability=external_availability,allow_boost_hours=allow_boost_hours,idsr_groups=idsr_groups,interface_scope=interface_scope,hydrogen_scope=hydrogen_scope,balance_start=balance_start)

def solve(problem,objective='eens',time_limit=1800,secondary=True,run_crossover=True,display=False,method='highs-ds'):
 p=problem;b=p['builder'];uns=b.blocks['unserved'];c=p['cost'].copy()
 if objective=='eens':c[:]=0;c[uns.ravel()]=1
 opts=dict(time_limit=float(time_limit),presolve=True,threads=2,run_crossover=run_crossover,disp=display,primal_feasibility_tolerance=1e-7,dual_feasibility_tolerance=1e-7)
 t=time.time();res=linprog(c,A_eq=p['eq'],b_eq=p['rhs'],A_ub=p['ub'],b_ub=p['budget'],bounds=p['bounds'],method=method,options=opts)
 calls=[dict(stage=objective,method=method,status=int(res.status),seconds=time.time()-t)]
 if not res.success:return dict(success=False,message=res.message,calls=calls,n_variables=b.n,n_equalities=b.ne,n_inequalities=b.nu)
 minens=float(res.x[uns].sum())
 first_arrays={k:res.x[v] for k,v in b.blocks.items()} if secondary else None
 if p.get('stage1_save_prefix'):
  prefix=p['stage1_save_prefix'];saved={k:res.x[v] for k,v in b.blocks.items()}
  np.savez_compressed(prefix+'.npz',**saved)
  pe=float(abs(p['eq']@res.x-p['rhs']).max())
  pu=float(np.maximum(p['ub']@res.x-p['budget'],0).max())
  pb=float(max(np.maximum(p['bounds'][:,0]-res.x,0).max(),np.maximum(res.x-p['bounds'][:,1],0).max()))
  if max(pe,pu,pb)>2e-4:raise RuntimeError('Primary-stage physical residual too large')
  Path(prefix+'.json').write_text(json.dumps(dict(success=True,stage='minimum_EENS',EENS_MWh=minens,equality_error_MWh=pe,inequality_error=pu,bound_error=pb,physical_verified=True,calls=calls,**p.get('stage1_metadata',{})),ensure_ascii=False,indent=1)+'\n')
 if objective=='eens' and secondary:
  rr=coo_matrix((np.ones(uns.size),(np.zeros(uns.size),uns.ravel())),shape=(1,b.n)).tocsr()
  ub=vstack([p['ub'],rr],format='csr');budget=np.r_[p['budget'],minens+1e-4]
  tt=time.time();r2=linprog(p['cost'],A_eq=p['eq'],b_eq=p['rhs'],A_ub=ub,b_ub=budget,bounds=p['bounds'],method=method,options=opts)
  calls.append(dict(stage='operating_cost_tiebreak',method=method,status=int(r2.status),seconds=time.time()-tt))
  if not r2.success:return dict(success=False,message=r2.message,calls=calls,n_variables=b.n,n_equalities=b.ne,n_inequalities=b.nu)
  res=r2
 x=res.x;arr={k:x[v] for k,v in b.blocks.items()};eqerr=float(abs(p['eq']@x-p['rhs']).max());uberr=float(np.maximum(p['ub']@x-p['budget'],0).max());be=float(max(np.maximum(p['bounds'][:,0]-x,0).max(),np.maximum(x-p['bounds'][:,1],0).max()))
 if max(eqerr,uberr,be)>2e-4:raise RuntimeError('Physical residual too large '+str((eqerr,uberr,be)))
 if objective=='eens' and arr['unserved'].sum()>minens+3e-4:raise RuntimeError('Lexicographic EENS failed')
 T,Z=p['data']['demand'].shape;us=arr['unserved'].sum(axis=1);eta=float(p['data']['eta_bio'])
 result=dict(success=True,solver_method=method,arrays=arr,EENS_MWh=float(us.sum()),system_shortage_hours=int((us>1e-3).sum()),zone_shortage_hours=int((arr['unserved']>1e-3).sum()),boost_MWh=float(arr['boost'].sum()),compression_electricity_MWh=float(arr.get('gas_charge',np.zeros(1)).sum()*float(p['data'].get('compression_kappa',.03))),net_boost_MWh=float(arr['boost'].sum()),auxiliary_fuel_MWh=float(arr.get('gas_charge',np.zeros(1)).sum()*float(p['data'].get('compression_kappa',.03))/eta),gas_generation_MWh=float(arr['gas'].sum()),external_import_MWh=float(arr['external_import'].sum()),hydro_generation_MWh=float(arr['hydro_generation'].sum()),hydro_pumping_MWh=float(arr['hydro_pumping'].sum()),existing_battery_charge_MWh=float(arr['existing_battery_charge'].sum()),DSR_MWh=float(arr['dsr'].sum()),new_battery_charge_MWh=float(arr.get('new_battery_charge',np.zeros(1)).sum()),boost_fuel_MWh=float(arr['boost'].sum()/eta) if p['tech'] in ['biomethane','natural_gas'] else 0.,objective_cost_eur=float(p['cost']@x),minimum_EENS_MWh=minens,calls=calls,solve_seconds=time.time()-t,n_variables=b.n,n_equalities=b.ne,n_inequalities=b.nu,equality_error_MWh=eqerr,inequality_error=uberr,bound_error=be,power_MW=float(arr['design_power'].sum()) if p['design'] else float(p['power'].sum()),energy_MWh=float(arr['design_energy'].sum()) if p['design'] else float(p['energy'].sum()),tech=p['tech'],gas_cut=p['gas_cut'],external_availability=p['external_availability'])
 ibase=np.asarray(p['data'].get('idsr_baseline',np.zeros((T,0))))
 shifted=ibase-arr['idsr_consumption']
 window_error=0.
 for group in p['idsr_groups']:
  actual=np.bincount(group['inverse'],weights=arr['idsr_consumption'][:,group['column']],minlength=len(group['labels']))
  window_error=max(window_error,float(abs(actual-group['baseline_MWh']).max(initial=0)))
 result.update(hydrogen_generation_MWh=float(arr['hydrogen'].sum()),
  hydrogen_supply_scope=hydrogen_scope_text(p['hydrogen_scope']),hydrogen_scope=p['hydrogen_scope'],
  idsr_consumption_MWh=float(arr['idsr_consumption'].sum()),idsr_baseline_MWh=float(ibase.sum()),
  idsr_shifted_away_MWh=float(np.maximum(shifted,0).sum()),idsr_recovery_consumption_MWh=float(np.maximum(-shifted,0).sum()),
  idsr_window_energy_error_MWh=window_error,
  idsr_annual_energy_error_MWh=float(abs(arr['idsr_consumption'].sum(axis=0)-ibase.sum(axis=0)).max(initial=0)),
  interface_scope=p['interface_scope'])
 if p.get('allow_boost_hours') is not None:
  allowed=np.asarray(p['allow_boost_hours'],dtype=bool);result['allowed_boost_hours']=int(allowed.sum());result['boost_outside_allowed_hours_MWh']=float(arr['boost'][~allowed].sum())
 countries=p['data'].get('countries',np.array([str(i) for i in range(Z)]))
 result['country_shortage_hours']=int(sum((arr['unserved'][:,countries==ct].sum(axis=1)>1e-3).sum() for ct in np.unique(countries)))
 result['hydro_simultaneous_generation_pumping_MWh']=float(np.minimum(arr['hydro_generation'],arr['hydro_pumping']).sum())
 result['battery_simultaneous_charge_discharge_MWh']=float(np.minimum(arr['existing_battery_charge'],arr['existing_battery_discharge']).sum())
 if 'gas_charge' in arr:
  result['gas_simultaneous_charge_withdraw_MWh']=float(np.minimum(arr['gas_charge'],arr['gas_withdraw']).sum())
  production=p['data'].get('q_hourly',np.broadcast_to(p['data']['q'],(T,Z)))
  result['annual_gas_balance_error_MWh']=float(abs(production.sum(axis=0)-(np.asarray(p['data']['base'])*T/eta+arr['boost'].sum(axis=0)/eta+arr['other_gas_use'].sum(axis=0)+arr['gas_charge'].sum(axis=0)*float(p['data'].get('compression_kappa',.03))/eta)).max())
 if first_arrays is not None:result['stage1_arrays']=first_arrays
 if hasattr(res,'eqlin') and res.eqlin is not None:
  result['nodal_dual']=np.asarray(res.eqlin.marginals[p['balance_start']:p['balance_start']+T*Z]).reshape(T,Z)
 return result

def load_data(path):
 with np.load(path) as a:return {k:a[k].copy() for k in a.files}
