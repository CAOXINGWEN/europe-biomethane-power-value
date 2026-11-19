"""Optional COPT backend for exactly the same sparse LP matrices."""
from pathlib import Path
import sys,time,os
import numpy as np
from scipy.sparse import vstack,csc_matrix
from scipy.optimize import OptimizeResult
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'dataset_one/39_public_validation_system_comparison_v27/validation/solver_vendor'))
import coptpy as cp
_cache=None

def linprog(c,A_eq=None,b_eq=None,A_ub=None,b_ub=None,bounds=None,method=None,options=None):
 global _cache
 opts=options or {};t=time.time()
 warm=(os.environ.get('V28_WARM_SECONDARY')=='1' and _cache is not None and _cache['eq'] is A_eq and _cache['bounds'] is bounds and A_ub.shape[0]==_cache['ub'].shape[0]+1)
 if warm:
  assert (A_ub[:-1]-_cache['ub']).nnz==0
  env=_cache['env'];m=_cache['model'];row=A_ub.getrow(A_ub.shape[0]-1)
  m.addConstr(cp.quicksum(float(v)*m.getVar(int(i)) for i,v in zip(row.indices,row.data))<=float(b_ub[-1]))
  m.setInfo(cp.COPT.Info.Obj,m.getVars(),np.asarray(c,float));_cache=None
  m.solve();status=int(m.getAttr(cp.COPT.Attr.LpStatus));success=status==cp.COPT.OPTIMAL
  x=np.asarray(m.getValues(),float) if bool(m.getAttr(cp.COPT.Attr.HasLpSol)) else None
  dual=np.asarray(m.getDuals(),float) if success else None
  return OptimizeResult(success=success,status=0 if success else status,message='COPT warm LP status '+str(status),x=x,fun=float(np.dot(c,x)) if x is not None else np.nan,nit=0,copt_status=status,copt_seconds=time.time()-t,warm_basis_reused=True,eqlin=OptimizeResult(marginals=dual[:len(b_eq)]) if dual is not None else None)
 _cache=None
 env=cp.Envr();m=env.createModel('v28_identical_sparse_LP')
 m.setParam(cp.COPT.Param.Logging,0);m.setParam(cp.COPT.Param.LogToConsole,0)
 m.setParam(cp.COPT.Param.Threads,int(opts.get('threads',4)))
 m.setParam('MemLimit',float(os.environ.get('V32_SOLVER_MEMORY_MB','12288')))
 m.setParam(cp.COPT.Param.TimeLimit,float(opts.get('time_limit',1800)))
 m.setParam(cp.COPT.Param.FeasTol,float(opts.get('primal_feasibility_tolerance',1e-7)))
 m.setParam(cp.COPT.Param.DualTol,float(opts.get('dual_feasibility_tolerance',1e-7)))
 if os.environ.get('V36_STAGE_METHODS'):
  primary=bool(np.all((np.asarray(c)==0)|(np.asarray(c)==1)))
  chosen=int(os.environ['V36_STAGE_METHODS'].split(',')[0 if primary else 1])
  m.setParam('LpMethod',chosen);m.setParam('Threads',int(opts.get('threads',4)))
  if os.environ.get('V36_SOLVER_LOG_PREFIX'):
   m.setParam('Logging',1);m.setLogFile(os.environ['V36_SOLVER_LOG_PREFIX']+('_primary.log' if primary else '_cost.log'))
 if os.environ.get('V28_LP_METHOD'):
  m.setParam('LpMethod',int(os.environ['V28_LP_METHOD']))
  m.setParam('Threads',int(os.environ.get('V28_SOLVER_THREADS','8')))
  if os.environ['V28_LP_METHOD']=='4':m.setParam('ConcurrentLpMode',0)
  m.setParam('MemLimit',float(os.environ.get('V28_SOLVER_MEMORY_MB','24576')))
 if os.environ.get('V28_SOLVER_LOG'):
  m.setParam('Logging',1);m.setLogFile(os.environ['V28_SOLVER_LOG'])
 mats=[];lower=[];upper=[]
 if A_eq is not None:mats.append(A_eq);lower.append(b_eq);upper.append(b_eq)
 if A_ub is not None:mats.append(A_ub);lower.append(np.full(len(b_ub),-cp.COPT.INFINITY));upper.append(b_ub)
 A=vstack(mats,format='csc');lhs=np.concatenate(lower);rhs=np.concatenate(upper)
 lo=np.asarray(bounds[:,0],float);hi=np.asarray(bounds[:,1],float);hi=np.where(np.isinf(hi),cp.COPT.INFINITY,hi);lo=np.where(np.isneginf(lo),-cp.COPT.INFINITY,lo)
 m.loadMatrix(np.asarray(c,float),A,np.asarray(lhs,float),np.asarray(rhs,float),lo,hi)
 m.solve();status=int(m.getAttr(cp.COPT.Attr.LpStatus));success=status==cp.COPT.OPTIMAL
 x=np.asarray(m.getValues(),float) if bool(m.getAttr(cp.COPT.Attr.HasLpSol)) else None
 if success and os.environ.get('V28_WARM_SECONDARY')=='1':_cache=dict(env=env,model=m,eq=A_eq,bounds=bounds,ub=A_ub)
 dual=np.asarray(m.getDuals(),float) if success else None
 r=OptimizeResult(success=success,status=0 if success else status,message='COPT LP status '+str(status),x=x,fun=float(np.dot(c,x)) if x is not None else np.nan,nit=0,copt_status=status,copt_seconds=time.time()-t,warm_basis_reused=False,eqlin=OptimizeResult(marginals=dual[:len(b_eq)]) if dual is not None else None)
 return r
