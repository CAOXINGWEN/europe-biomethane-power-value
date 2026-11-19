"""Required V35 pipeline reproduction before new scientific runs."""
from pathlib import Path
import json,sys,os,time,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'scripts'))
import numpy as np
import model36 as model
import solver36


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def main():
    out=HERE/'qa'/('reproduction_extended_main_eval_rule_eens_WS13'+os.environ.get('V36_REPRO_SUFFIX','')+'.json')
    if out.exists():
        saved=json.loads(out.read_text())
        for key,file in [('model36_sha256','model36.py'),('solver36_sha256','solver36.py')]:
            if saved.get(key)!=hashlib.sha256((HERE/'scripts'/file).read_bytes()).hexdigest():raise RuntimeError('Cached reproduction binds an earlier implementation; select an unused V36_REPRO_SUFFIX.')
        print(out.read_text());return
    array_path=out.with_name(out.name[:-5]+'.npz')
    if array_path.exists():raise FileExistsError('Existing reproduction arrays retained; select an unused V36_REPRO_SUFFIX.')
    v32=ROOT/'dataset_one/44_followup_corrected_system_v32/stage_b'
    d=load(v32/'inputs/2035_WS13.npz');ba=load(v32/'results/baseline_WS13.npz')
    ds=json.loads((v32/'results/design_rule_eens.json').read_text())
    pwr=np.array(ds['power']);energy=np.array(ds['energy']);allowed=ba['unserved'].sum(axis=1)>.001
    p=model.build_problem(d,power=pwr,energy=energy,gas_cut=.3,external_availability=.5,allow_boost_hours=allowed,zero_hours=~allowed)
    for name in ['dsr','idsr_consumption']:
        ids=p['builder'].blocks[name].ravel();values=ba[name].ravel();p['bounds'][ids,0]=values;p['bounds'][ids,1]=values
    p['bounds'][p['builder'].blocks['gas_withdraw'].ravel(),1]=np.tile(.75*d['q']+pwr/.4,8760)
    def lp(*args,**kw):
        kw['options']=dict(kw.get('options') or {},threads=4);return solver36.linprog(*args,**kw)
    model.linprog=lp
    r=model.solve(p,secondary=False,time_limit=3600,method='copt')
    a=r.pop('arrays',None)
    if a is not None:np.savez_compressed(array_path,**a)
    refpath=ROOT/'dataset_one/47_review_response_v35/results/main_eval_rule_eens_WS13.json'
    ref=json.loads(refpath.read_text());diff=abs(r.get('EENS_MWh',1e100)-ref['EENS_MWh'])
    r.pop('nodal_dual',None)
    r.update(reference_EENS_MWh=ref['EENS_MWh'],absolute_difference_MWh=diff,passed=bool(r['success'] and diff<=1e-6),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),model36_sha256=hashlib.sha256((HERE/'scripts/model36.py').read_bytes()).hexdigest(),solver36_sha256=hashlib.sha256((HERE/'scripts/solver36.py').read_bytes()).hexdigest())
    if array_path.exists():r.update(arrays_file=str(array_path.relative_to(HERE)),arrays_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest())
    out.write_text(json.dumps(r,ensure_ascii=False,indent=1)+'\n');print(json.dumps({k:r[k] for k in ['success','passed','EENS_MWh','absolute_difference_MWh','solve_seconds']},indent=1),flush=True)
    if not r['passed']:raise SystemExit(1)


if __name__=='__main__':main()
