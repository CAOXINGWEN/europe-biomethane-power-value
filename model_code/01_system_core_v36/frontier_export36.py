"""Numerical R5 interpolation and threshold coordinates, without plotting or extrapolation."""
import numpy as np
import pandas as pd
from common36 import HERE,dump,now


def crossings(x,y,target):
    """Retain every crossing, including non-monotone segments and exact node hits."""
    values=[]
    for a,b,ya,yb in zip(x[:-1],x[1:],y[:-1],y[1:]):
        if abs(ya-target)<1e-12:values.append(float(a))
        if (ya-target)*(yb-target)<0:values.append(float(a+(b-a)*(target-ya)/(yb-ya)))
    if abs(y[-1]-target)<1e-12:values.append(float(x[-1]))
    return sorted(set(round(v,12) for v in values))


def export_frontier(frame,output_directory=None):
    if frame.empty:return
    power=np.array([.26,1.,4.,16.]);storage=np.array([22.,168.,720.])
    surface=[];threshold=[];coverage=[]
    for beta,g in frame.groupby('beta'):
        grid=g.pivot(index='buffer_h',columns='power_GW',values='reduction_fraction').reindex(index=storage,columns=power)
        sample=g.pivot(index='buffer_h',columns='power_GW',values='sampled_weathers').reindex(index=storage,columns=power)
        complete=bool(grid.notna().all().all() and (sample==8).all().all())
        coverage.append(dict(beta=float(beta),complete=complete,registered_grid_cells=12,available_cells=len(g)))
        if not complete:continue
        z=grid.to_numpy();fine_s=np.unique(np.r_[storage,np.linspace(22,720,141)])
        fine_p=np.unique(np.r_[power,np.linspace(.26,16,161)])
        for s in fine_s:
            # Interpolate first along storage, then power: bilinear in physical coordinates.
            at_p=np.array([np.interp(s,storage,z[:,j]) for j in range(len(power))])
            for pg,val in zip(fine_p,np.interp(fine_p,power,at_p)):
                surface.append(dict(beta=float(beta),power_GW=float(pg),buffer_h=float(s),interpolated_reduction_fraction=float(val)))
            for target in [.01,.05,.10]:
                points=crossings(power,at_p,target)
                if points:
                    for branch,pg in enumerate(points):threshold.append(dict(beta=float(beta),target_fraction=target,buffer_h=float(s),power_GW=pg,branch=branch,status='within_tested_power_interval'))
                else:
                    status='above_target_throughout_tested_power_interval' if np.min(at_p)>target else 'below_target_throughout_tested_power_interval'
                    threshold.append(dict(beta=float(beta),target_fraction=target,buffer_h=float(s),power_GW=np.nan,branch=np.nan,status=status))
    out=output_directory if output_directory is not None else HERE/'report'
    if surface:pd.DataFrame(surface).to_csv(out/'frontier_interpolated_surface.csv',index=False)
    if threshold:pd.DataFrame(threshold).to_csv(out/'frontier_threshold_coordinates.csv',index=False)
    dump(out/'frontier_interpolation_metadata.json',dict(created_utc=now(),method='Bilinear in linear power_GW and buffer_h coordinates, within measured rectangle only',domain=dict(power_GW=[.26,16.],buffer_h=[22,720]),thresholds=[.01,.05,.10],extrapolation=False,monotonicity_imposed=False,all_crossing_branches_retained=True,coverage=coverage,partial=not(len(coverage)==2 and all(r['complete'] for r in coverage))))
