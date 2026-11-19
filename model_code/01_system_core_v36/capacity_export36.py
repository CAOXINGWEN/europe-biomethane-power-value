"""ELCC search points and explicitly allocated country shares of the system equivalent."""
import json
import numpy as np,pandas as pd
from common36 import HERE,load_input


def export():
    path=HERE/'protocol/elcc_state.json'
    if not path.exists():return
    state=json.loads(path.read_text());points=[];alloc=[];d=load_input(23,'Central','R0')
    for use,s in state.items():
        for i,v in enumerate(s.get('evaluations',[])):
            points.append(dict(use={'R1':'U1','R2':'U2'}[use],evaluation=i,firm_capacity_MW=v['capacity_MW'],mean_reduction_MWh=v['mean_reduction_MWh'],target_mean_reduction_MWh=s['target_mean_reduction_MWh'],absolute_error_MWh=abs(v['mean_reduction_MWh']-s['target_mean_reduction_MWh']),mode=v['mode'],solved_risk_weather_count=len(v['job_names']),all_weather_divisor=36,search_complete=s['complete'],converged=s.get('converged',False)))
        if s.get('complete') and s.get('converged'):
            shares=np.asarray(s['shares'])
            for country in np.unique(d['countries']):
                share=float(shares[d['countries']==country].sum())
                alloc.append(dict(country=str(country),use={'R1':'U1','R2':'U2'}[use],system_ELCC_MW=s['ELCC_MW'],country_capacity_share=share,allocated_system_equivalent_firm_capacity_MW=share*s['ELCC_MW'],allocation_scope='Allocation of one joint system ELCC, not an independently identified national causal ELCC',search_absolute_error_MWh=s['absolute_error_MWh'],search_tolerance_MWh=s['tolerance_MWh']))
    if points:pd.DataFrame(points).to_csv(HERE/'report/ELCC_search_points.csv',index=False)
    if alloc:pd.DataFrame(alloc).to_csv(HERE/'report/ELCC_allocated_country_capacity.csv',index=False)
