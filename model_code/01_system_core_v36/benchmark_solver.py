"""Compare an exact alternative LP algorithm on the same R0 pilot matrices."""
from pathlib import Path
import sys,os,json,time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common36 import HERE,load_input,dump,now,sha
import model36 as model
import solver36


def main():
    d=load_input(13,'Central','R0')
    costs=dict(system_gas_eur_per_MWh_el=d['gas_cost'],boost_eur_per_MWh_el=7.443609026,gas_charge_eur_per_MWh_fuel=.0001,system_external_eur_per_MWh=100.)
    p=model.build_problem(d,tech='none',power=np.zeros(24),energy=np.zeros(24),gas_cut=.3,external_availability=.5,costs=costs)
    os.environ['V28_LP_METHOD']='2';os.environ['V28_SOLVER_THREADS']='4';os.environ['V28_SOLVER_MEMORY_MB']='12288'
    os.environ.pop('V28_WARM_SECONDARY',None)
    stages=[]
    def lp(*a,**kw):
        label='primary' if not stages else 'cost'
        os.environ['V28_SOLVER_LOG']=str(HERE/'logs'/f'benchmark_barrier_{label}.log')
        r=solver36.linprog(*a,**kw);stages.append(dict(stage=label,success=bool(r.success),seconds=float(r.copt_seconds)));return r
    model.linprog=lp
    r=model.solve(p,time_limit=1800,method='copt',secondary=True)
    a=r.pop('arrays',None);r.pop('stage1_arrays',None);r.pop('nodal_dual',None)
    if a is not None:np.savez_compressed(HERE/'qa/benchmark_R0_WS13_barrier.npz',**a)
    r.update(utc=now(),settings=dict(LpMethod=2,threads=4,memory_MiB=12288,warm_secondary=False),stages=stages,model_sha256=sha(HERE/'scripts/model36.py'))
    dump(HERE/'qa/benchmark_R0_WS13_barrier.json',r)
    print(json.dumps({k:r.get(k) for k in ['success','EENS_MWh','objective_cost_eur','calls','solve_seconds']},indent=1),flush=True)


if __name__=='__main__':main()
