"""Small physical counterexamples for every new model extension."""
from pathlib import Path
import sys,json,copy
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import model36 as m
import solver36


def tiny():
    T=24;Z=1;zero=np.zeros((T,Z));q=np.zeros(Z)
    return dict(demand=zero.copy(),vre=zero.copy(),ror=zero.copy(),fixed_other=zero.copy(),gas_cap=zero.copy(),other_thermal_cap=zero.copy(),q=q,base=q.copy(),eta_bio=np.array(.4),beta=np.array(.75),compression_kappa=np.array(.03),event=np.ones(T,bool),hydro_zone=np.array([0]),hydro_gen_cap=zero.copy(),hydro_pump_cap=zero.copy(),hydro_min=zero.copy(),hydro_max=zero.copy(),hydro_initial=q.copy(),hydro_terminal=q.copy(),hydro_inflow=zero.copy(),battery_power=q.copy(),battery_energy=q.copy(),edge_from=np.zeros(0,dtype=int),edge_to=np.zeros(0,dtype=int),edge_cap=np.zeros((T,0)),external_cap=zero.copy(),dsr_zone=np.zeros(0,dtype=int),dsr_cap=np.zeros(0),dsr_hours=np.zeros(0),pump_eta=np.array(.8),battery_eta=np.array(np.sqrt(.9)))


def solve(d,**kw):
    p=m.build_problem(d,costs=dict(system_gas_eur_per_MWh_el=90,boost_eur_per_MWh_el=8),**kw)
    r=m.solve(p,method='copt',secondary=True,time_limit=60)
    assert r['success'],r
    assert r['EENS_MWh']<=r['minimum_EENS_MWh']+3e-4
    return r


def main():
    m.linprog=solver36.linprog;checks=[]
    d=tiny();d['demand'][:]=10;d['gas_cap'][:]=20
    d.update(thermal_cap=np.full((24,1),3.),thermal_cost=np.array([20.]),thermal_zone=np.array([0]),thermal_technology=np.array(['Lignite']))
    r=solve(d)
    assert abs(r['arrays']['thermal_dispatch'].sum()-72)<1e-3
    assert abs(r['arrays']['gas'].sum()-168)<1e-3
    assert 'nodal_dual' in r and abs(r['nodal_dual']-90).max()<1e-5
    checks.append('Thermal split respects capacities, merit order, dual value and two-stage EENS cap')
    d=tiny();d['demand'][:]=10;d['gas_cap'][:]=10;d['gas_relief']=np.full((24,1),100.)
    r=solve(d,gas_cut=.3);assert r['minimum_EENS_MWh']<1e-6
    assert r['arrays']['gas'].max()<=10+1e-6
    d['event'][12:]=False;r=solve(d,gas_cut=.3);assert r['minimum_EENS_MWh']<1e-6
    checks.append('Injection restores derated capacity only and never exceeds pre-event capacity')
    d=tiny();d['demand'][:]=10;d['gas_cap'][:]=5
    r0=solve(d);d['firm_capacity_MW']=np.array([2.]);r=solve(d)
    assert abs(r0['minimum_EENS_MWh']-r['minimum_EENS_MWh']-48)<1e-6
    checks.append('Fully reliable capacity reduces shortage by the correct delivered hourly energy')
    d=tiny();d['demand'][12]=10;d['q']=np.array([1.]);d['q_hourly']=np.ones((24,1));d['beta']=np.array(0.)
    allow=np.zeros(24,bool);allow[12]=True
    r1=solve(d,power=np.array([8.]),energy=np.array([22.]),allow_boost_hours=allow)
    d['relaxed_frontier']=np.array(True)
    r2=solve(d,power=np.array([8.]),energy=np.array([22.]),allow_boost_hours=allow)
    assert r2['minimum_EENS_MWh']<r1['minimum_EENS_MWh']-1
    assert abs(r2['arrays']['boost'].sum()-8)<1e-3
    assert r2['annual_gas_balance_error_MWh']<2e-4
    checks.append('Relaxed frontier permits a physical peak above the old 12q withdrawal limit while conserving annual gas')
    r=solve(tiny());assert r['EENS_MWh']==0 and len(r['stage1_arrays'])>0
    checks.append('No-biomethane counterfactual is feasible and saves both stages')
    out=Path(__file__).resolve().parents[1]/'qa/model_extension_tests.json'
    out.write_text(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
