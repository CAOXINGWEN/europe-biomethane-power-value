"""Run a registered V36 physical case and save both complete LP stages."""
from pathlib import Path
import sys,os,json,time,argparse,traceback,fcntl
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common36 import HERE,V32,ROOT,now,sha,dump,arrays,load_input,design


def execute(job):
    import model36 as model
    import solver36
    def lp(*a,**kw):
        kw['options']=dict(kw.get('options') or {},threads=4);return solver36.linprog(*a,**kw)
    model.linprog=lp
    start=time.time();started_utc=now();w=job['weather'];batch=job['batch'];scenario=job.get('scenario','Central')
    d=load_input(w,scenario,batch,beta=float(job.get('beta',.75)))
    if job.get('gas_efficiency') is not None:
        d['gas_efficiency'][:]=job['gas_efficiency']
        if batch=='R3':d['gas_relief']=d['resource_q_hourly']*job['gas_efficiency']*d['event'][:,None]
    if batch=='R3' and job.get('network_pool'):
        cut=.3*d['gas_cap']*d['event'][:,None];out=np.zeros_like(cut)
        for h in np.flatnonzero(d['event']):
            remaining=float(d['gas_relief'][h].sum())
            for z in np.argsort(-cut[h],kind='stable'):
                out[h,z]=min(remaining,cut[h,z]);remaining-=out[h,z]
                if remaining<=1e-12:break
        d['gas_relief']=out
    P=np.zeros_like(d['q']);S=P.copy();allowed=None
    if batch in ['R2','R5']:
        ds=design(scenario);P=np.array(ds['power']);S=np.array(ds['energy'])
        ba=arrays(HERE/'results'/f'R0_WS{w:02d}.npz');allowed=ba['unserved'].sum(axis=1)>.001
        if batch=='R5':
            weights=np.array(ds['weights']);weights=np.where(d['q']>0,weights,0)
            P=weights/weights.sum()*float(job['power_GW'])*1000
            S=float(job['buffer_h'])*d['q'];d['relaxed_frontier']=np.array(True)
    if batch=='R4':d['firm_capacity_MW']=np.asarray(job['firm_shares'])*float(job['firm_MW'])
    if batch=='R1' and job.get('site_efficiency') is not None:
        d['demand']+=(.4-job['site_efficiency'])*d['resource_q_hourly']
    units=json.loads((HERE/'inputs/dispatch_cost_units.json').read_text())
    costs=dict(system_gas_eur_per_MWh_el=d['gas_cost'],boost_eur_per_MWh_el=units['boost_variable_OM_EUR2024_per_MWh'],
               gas_charge_eur_per_MWh_fuel=.0001,system_external_eur_per_MWh=100.)
    p=model.build_problem(d,tech='biomethane' if batch in ['R2','R5'] else 'none',power=P,energy=S,gas_cut=.3,external_availability=.5,costs=costs,allow_boost_hours=allowed)
    os.environ['V36_SOLVER_LOG_PREFIX']=str(HERE/'logs'/job['name'])
    p['stage1_save_prefix']=str(HERE/'results'/(job['name']+'_s1'))
    p['stage1_metadata']=dict(parent_case=job['name'],job=job,registry_sha256=sha(HERE/'protocol/registration_v36.json'))
    # Nonzero-input source and code hashes bind the result to its implementation.
    dependencies={str(x.relative_to(HERE)):sha(x) for x in [HERE/'scripts/model36.py',HERE/'scripts/solver36.py',HERE/'scripts/common36.py',HERE/'scripts/run_case.py',HERE/'inputs/thermal_split_2035.npz',HERE/'inputs/resource_zone_weights.npz',HERE/'inputs/uk_resource_v36.csv']}
    result=model.solve(p,time_limit=3600,method='copt',secondary=(batch not in ['R4','R5']))
    if not result['success']:raise RuntimeError(json.dumps(result))
    a=result.pop('arrays');s1=result.pop('stage1_arrays',None);dual=result.pop('nodal_dual',None)
    if dual is not None:a['nodal_dual']=dual
    # Retain raw thermal block plus explicit technology labels for later analysis.
    a.update(thermal_zone=d['thermal_zone'],thermal_technology=d['thermal_technology'],thermal_efficiency=d['thermal_efficiency'],gas_efficiency=d['gas_efficiency'],zones=d['zones'],countries=d['countries'],resource_q_hourly=d['resource_q_hourly'],power_vector=P,buffer_vector=S)
    un=np.maximum(a['unserved'],0)
    result.update(job=job,started_utc=started_utc,elapsed_seconds=time.time()-start,completed_utc=now(),physical_verified=max(result['equality_error_MWh'],result['inequality_error'],result['bound_error'],result.get('annual_gas_balance_error_MWh',0))<=2e-4,zone_EENS_MWh=un.sum(axis=0).tolist(),zones=d['zones'].tolist(),countries=d['countries'].tolist(),resource_MWh_by_zone=(d['resource_q']*8760).tolist(),base_generation_MWh_by_zone=(d['base']*8760).tolist(),source_input_sha256=sha(V32/'inputs'/f'2035_WS{w:02d}.npz'),dependencies_sha256=dependencies,registry_sha256=sha(HERE/'protocol/registration_v36.json'))
    name=job['name'];out=HERE/'results';pn=out/(name+'.npz');np.savez_compressed(pn,**a);result['arrays_sha256']=sha(pn)
    if s1 is not None:
        p1=out/(name+'_s1.npz')
        saved=json.loads((out/(name+'_s1.json')).read_text());saved.update(arrays_sha256=sha(p1),dependencies_sha256=dependencies)
        dump(out/(name+'_s1.json'),saved)
    # The main JSON+NPZ represent the final cost stage, with stage 1 separately retained.
    result['stages_saved']=1 if s1 is None else 2
    dump(out/(name+'.json'),result)
    print(json.dumps({k:result[k] for k in ['success','EENS_MWh','elapsed_seconds','stages_saved']},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);args=ap.parse_args()
    jobs=json.loads((HERE/'protocol/jobs_v36.json').read_text())['jobs'];job=next(j for j in jobs if j['name']==args.job)
    lock=(HERE/'logs'/(args.job+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Case is already running')
    path=HERE/'results'/(args.job+'.json')
    if path.exists():raise FileExistsError('Existing receipt must be retained: '+str(path))
    try:execute(job)
    except Exception:
        dump(path,dict(success=False,job=job,completed_utc=now(),error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
