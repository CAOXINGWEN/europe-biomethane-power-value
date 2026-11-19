"""Freeze a single reviewed LP pair so a reader need not fetch the ERAA arrays."""
from pathlib import Path
import numpy as np
import pandas as pd
import window_dispatch as w


def export(intervention,stem):
    d,b0,b1,sched,labels,start=w.window_data(2035,17,4)
    w.flexload37.patch(w.model36,2035)
    raw=next(v for n,v,_ in w.injections(d,2035,True) if n==intervention)
    eff=.351 if intervention=="Pinerolo" else .4
    dest=w.HERE/"example"
    pack={}
    try:
        for name,change in [("base",0.),("site",eff*raw)]:
            dj=dict(d,demand=d["demand"]-change)
            p=w.model36.build_problem(dj,tech="none",gas_cut=.3,external_availability=.5,
                    costs={"system_gas_eur_per_MWh_el":d["gas_cost"],"system_external_eur_per_MWh":100.})
            w.boundary_states(p,dj,b0,b1,sched,labels)
            if name=="base":
                for matrix in ("eq","ub"):
                    a=p[matrix]
                    pack.update({matrix+"_data":a.data,matrix+"_indices":a.indices,matrix+"_indptr":a.indptr,matrix+"_shape":a.shape})
                ids=p["builder"].blocks
                fuel=np.zeros(len(p["cost"]));co2=fuel.copy()
                fuel[ids["gas"]]=1/d["gas_efficiency"]
                co2[ids["gas"]]=w.account.EF["GWP100"]["gas"]/d["gas_efficiency"]
                for j,tech in enumerate(d["thermal_technology"]):
                    if tech in w.account.FOSSIL:
                        fuel[ids["thermal_dispatch"][:,j]]=1/d["thermal_efficiency"][j]
                        co2[ids["thermal_dispatch"][:,j]]=w.account.EF["GWP100"][tech]/d["thermal_efficiency"][j]
                pack.update(fuel=fuel,emissions=co2,cost=p["cost"],uns=ids["unserved"].ravel(),
                            raw=raw.sum(),electricity=eff*raw.sum())
            else:
                from scipy.sparse import csr_matrix
                for matrix in ("eq","ub"):
                    a=csr_matrix((pack[matrix+"_data"],pack[matrix+"_indices"],pack[matrix+"_indptr"]),shape=pack[matrix+"_shape"])
                    assert (p[matrix]-a).nnz==0
                assert np.array_equal(p["cost"],pack["cost"])
            pack.update({name+"_"+k:p[k] for k in ("rhs","budget","bounds")})
    finally:
        w.model36.build_problem=w.ORIGINAL_BUILD
    t=pd.read_csv(w.HERE/"source_data/window_pairs.csv").query(
        "year==2035 and weather==17 and month==4 and availability=='all_hours' and price=='fixed' and intervention==@intervention").iloc[0]
    pack.update(expected_D=t.D,expected_mu=t.electricity_MEF)
    np.savez_compressed(dest/(stem+"_2035_ws17_april.npz"),**pack)


if __name__=="__main__":
    export("Pinerolo","pinerolo")
    w.model36.build_problem=w.ORIGINAL_BUILD
    export("region_raw","region")
