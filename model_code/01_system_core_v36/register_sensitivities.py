"""Register explicitly requested injection and efficiency sensitivities."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common36 import HERE,dump,now,sha


def main():
    # Controller must not be editing an adaptive manifest simultaneously.
    import fcntl
    lock=(HERE/'logs/controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Stop controller before appending this registration')
    p=HERE/'protocol/jobs_v36.json';v=json.loads(p.read_text());names={j['name'] for j in v['jobs']};extra=[]
    for w in range(1,37):
        extra.append(dict(name=f'R3_Central_pool_WS{w:02d}',batch='R3',scenario='Central',weather=w,network_pool=True,variant='network_pool'))
        for eta in [.4,.58]:extra.append(dict(name=f'R3_Central_eta{eta:g}_WS{w:02d}',batch='R3',scenario='Central',weather=w,gas_efficiency=eta,variant='injection_efficiency_sensitivity',scope='Changes H72 injection capacity conversion; ordinary generator merit order remains the registered Central mix. Fuel-accounting efficiencies are reported separately.'))
        for eta in [.35,.45]:extra.append(dict(name=f'R1_Central_eta{eta:g}_WS{w:02d}',batch='R1',scenario='Central',weather=w,site_efficiency=eta,variant='site_efficiency_sensitivity'))
    added=[j for j in extra if j['name'] not in names]
    v['jobs'].extend(added);v['amendments'].append(dict(utc=now(),type='design_sensitivity_registration',jobs_added=len(added),hypotheses_unchanged=True));dump(p,v)
    p=HERE/'protocol/registration_v36.json';v=json.loads(p.read_text());v['amendments'].append(dict(id='amendment_05',registered_utc=now(),results_read_before_change='Only R0 benchmark and stage checkpoints; no U1/U2/U3 results read.',change='Expand the named design sensitivities before their execution: central-resource U3 pooled injection (36 cases), U3 H72 conversion efficiencies .40 and .58 with the central merit-order costs held fixed (72), and U1 site efficiencies .35 and .45 (72). All use 36 weather states and two stages. The primary R0–R5 specifications and H1–H4 criteria are unchanged. Additional result labels explicitly distinguish these sensitivity arms.'))
    dump(p,v);print(json.dumps(dict(added=len(added),physical_jobs=len(names)+len(added))))


if __name__=='__main__':main()
