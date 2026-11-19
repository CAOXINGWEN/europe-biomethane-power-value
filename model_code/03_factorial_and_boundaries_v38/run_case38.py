"""Run exactly one frozen V38 job; preserve successful and failed attempts."""
import sys, time, os, argparse, traceback, fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *

def execute(job):
    import model36 as model
    import solver36, flexload37
    start=time.time()
    dep=dependencies(job)
    reg=registry()
    for script in [Path(__file__),Path(__file__).with_name('common38.py')]:
        assert reg['source_sha256'][relative(script)]==sha(script), 'Frozen runner changed'
    assert sha(HERE/'protocol/jobs_v38.json')==reg['jobs_sha256']
    for p,h in dep.items():
        assert reg['source_sha256'][p]==h, 'Frozen source changed: '+p
    d=build_input(job)
    def lp(*a,**kw):
        kw['options']=dict(kw.get('options') or {},threads=4)
        return solver36.linprog(*a,**kw)
    model.linprog=lp
    flexload37.patch(model,job['year'])
    zero=np.zeros(len(d['zones']))
    p=model.build_problem(d,tech='none',power=zero,energy=zero,gas_cut=.3,external_availability=.5,
          costs=dict(system_gas_eur_per_MWh_el=d['gas_cost'],boost_eur_per_MWh_el=legacy.BOOST_VOM,
                     gas_charge_eur_per_MWh_fuel=.0001,system_external_eur_per_MWh=100.))
    name=job['name']; out=HERE/'results'
    os.environ['V36_SOLVER_LOG_PREFIX']=str(HERE/'logs'/name)
    p['stage1_save_prefix']=str(out/(name+'_s1'))
    p['stage1_metadata']=dict(parent_case=name,job=job,registry_sha256=sha(HERE/'protocol/registration_v38.json'))
    r=model.solve(p,time_limit=7200,method='copt',secondary=True)
    if not r['success']: raise RuntimeError(json.dumps({k:v for k,v in r.items() if k!='arrays'},default=str))
    a=r.pop('arrays'); s1=r.pop('stage1_arrays',None); dual=r.pop('nodal_dual',None)
    if dual is not None: a['nodal_dual']=dual
    a.update(thermal_zone=d['thermal_zone'],thermal_technology=d['thermal_technology'],thermal_efficiency=d['thermal_efficiency'],
             gas_efficiency=d['gas_efficiency'],zones=d['zones'],countries=d['countries'],resource_q_hourly=d['resource_q_hourly'],
             power_vector=zero,buffer_vector=zero)
    fu=p['flex_units']
    a.update(flex_zone=fu['zone'],flex_type=fu['type'],flex_cap=fu['cap'],flex_price=fu['price'],flex_eff=fu['eff'])
    pn=out/(name+'.npz')
    common37.save_arrays(pn,a)
    r.update(job=job,elapsed_seconds=time.time()-start,completed_utc=now(),physical_verified=True,
             stages_saved=2 if s1 is not None else 1,arrays_sha256=sha(pn),dependencies_sha256=dep,
             registry_sha256=sha(HERE/'protocol/registration_v38.json'),stage_methods=os.environ.get('V36_STAGE_METHODS','1,2'),
             zone_EENS_MWh=np.maximum(a['unserved'],0).sum(axis=0).tolist(),resource_MWh_by_zone=(d['resource_q']*8760).tolist(),
             site_generation_MWh=float(.4*d['resource_q_hourly'].sum()) if job['use']=='U1' else 0.)
    s1p=out/(name+'_s1.json'); sr=json.loads(s1p.read_text()); sr['arrays_sha256']=sha(out/(name+'_s1.npz')); dump(s1p,sr)
    dump(out/(name+'.json'),r)
    if job['batch']=='QA':
        import accounting37
        refp=V37/'results/F1_2035_U1_Central_WS17'
        ref=json.loads(refp.with_suffix('.json').read_text())
        ra=arrays(V37/'results/F0_2035_R0_WS17.npz')
        m=accounting37.finish_u1(accounting37.pair_metrics(a,ra))
        mr=accounting37.finish_u1(accounting37.pair_metrics(arrays(refp.with_suffix('.npz')),ra))
        checks=dict(EENS=abs(r['EENS_MWh']-ref['EENS_MWh'])<=.001,
                    objective=abs(r['objective_cost_eur']-ref['objective_cost_eur'])<=max(1,1e-8*abs(ref['objective_cost_eur'])),
                    displacement=abs(m['D_fossil_per_MWh']-mr['D_fossil_per_MWh'])<=1e-5,
                    balance=abs(m['balance_residual_relative'])<=1e-8)
        dump(HERE/'qa/reproduction.json',dict(passed=all(checks.values()),checks=checks,new_metrics=m,reference_metrics=mr,
             EENS_difference=r['EENS_MWh']-ref['EENS_MWh'],cost_difference=r['objective_cost_eur']-ref['objective_cost_eur']))
    print(json.dumps({k:r[k] for k in ['success','EENS_MWh','elapsed_seconds','stages_saved']}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);arg=ap.parse_args()
    job=next(j for j in jobs() if j['name']==arg.job)
    lock=(HERE/'logs'/(arg.job+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    target=HERE/'results'/(arg.job+'.json')
    if target.exists(): raise FileExistsError(str(target))
    try: execute(job)
    except Exception:
        if job['batch']=='QA':
            dump(HERE/'qa/reproduction.json',dict(passed=False,error=traceback.format_exc(),utc=now()))
        if not target.exists(): dump(target,dict(success=False,job=job,completed_utc=now(),error=traceback.format_exc()))
        raise
if __name__=='__main__': main()
