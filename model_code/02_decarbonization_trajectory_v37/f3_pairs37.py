"""Pair metrics for the single-country F3 runs (U1 one country minus F0 R0, same year and weather); appends to report/preliminary_F3_by_case.csv."""
import sys,json,glob; sys.path.insert(0,'scripts')
import numpy as np,pandas as pd
from pathlib import Path
from accounting37 import pair_metrics,finish_u1,FOSSIL
def load(p):
    with np.load(p,allow_pickle=False) as f: return {k:f[k] for k in f.files if k not in ('flow','nodal_dual','hydro_inventory','existing_battery_inventory','gas_inventory')}
years=[int(y) for y in sys.argv[1:]]; out=Path('report/preliminary_F3_by_case.csv')
old=pd.read_csv(out) if out.exists() else pd.DataFrame(columns=['name']); done=set(old.name); rows=[]
for y in years:
    for f in sorted(glob.glob(f'results/F3_{y}_U1_*_WS*.json')):
        if 'attempt' in f or '_s1' in f: continue
        r=json.load(open(f)); name=r['job']['name']
        if name in done or not (r.get('success') and r.get('physical_verified')): continue
        w=r['job']['weather']; u=load(f.replace('.json','.npz')); z=load(f'results/F0_{y}_R0_WS{w:02d}.npz')
        m=pair_metrics(u,z); x=finish_u1(m); site=x['site_generation_MWh']
        # where the displaced fossil generation happens (by country of the generating zone)
        ctry=np.asarray(u['countries']); loc={}
        dg=(z['gas']-u['gas']).sum(axis=0)
        tech=np.asarray(u['thermal_technology']); tz=np.asarray(u['thermal_zone'])
        dt=(z['thermal_dispatch']-u['thermal_dispatch']).sum(axis=0)
        for c in np.unique(ctry):
            ix=ctry==c; v=dg[ix].sum()+sum(dt[(tz==j)&np.isin(tech,FOSSIL)].sum() for j in np.flatnonzero(ix))
            loc[f'fossil_gen_reduction_in_{c}_MWh']=float(v)
        rows.append(dict(name=name,year=y,country=r['job']['country'],weather=w,scale=r.get('injection_scale',1.),B=m['biomethane_MWh'],
            fossil_MWh=m['fossil_fuel_displaced_MWh'],D=x['D_fossil_per_MWh'],avoided=x['avoided_GWP100_t_per_MWh'],avoided20=x['avoided_GWP20_t_per_MWh'],
            s_f=x['s_f'],m=x['m_t_per_MWh_e'],resid=x['balance_residual_relative'],
            **{k.replace('balance_','share_').replace('_MWh',''):v/site for k,v in m.items() if k.startswith('balance_')},**loc))
        if len(rows)%5==0: pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(out,index=False)
    pd.concat([old,pd.DataFrame(rows)],ignore_index=True).to_csv(out,index=False); print(y,'pairs',len(rows),flush=True)
