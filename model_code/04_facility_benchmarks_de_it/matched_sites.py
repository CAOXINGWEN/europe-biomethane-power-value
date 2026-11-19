"""Move an identical constant electricity injection between the two market nodes."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import os

import numpy as np
import pandas as pd
import window_dispatch as w


def trial(key):
    year, weather, month=key
    out=w.HERE/"experiments/matched"/f"{year}_ws{weather:02}_m{month:02}.csv"
    if out.exists():
        return
    d,b0,b1,sched,labels,start=w.window_data(year,weather,month)
    # Use the published Pinerolo CHP median and its dose at both nodes.
    val=pd.read_csv(w.HERE/"source_data/facility_inputs.csv").set_index("key").value
    eta=val["it_chp_electric"]
    hourly_raw=12*val["it_chp_input_month"]/8760
    rows=[]
    w.flexload37.patch(w.model36,year)
    try:
        r,re=w.solve(d,100.,.5,(b0,b1,sched,labels))
        for name,zone in [("Ruhleben","DE00"),("Pinerolo","ITN1")]:
            raw=np.zeros_like(d["demand"])
            raw[:,list(d["zones"]).index(zone)]=hourly_raw
            u,ue=w.solve(d,100.,.5,(b0,b1,sched,labels),raw,eta)
            u["resource_q_hourly"]=raw
            m=w.account.pair_metrics(u,r)
            f=w.account.finish_u1(m,eta)
            if abs(f["balance_residual_relative"])>5e-5:
                raise ValueError((key,name,f))
            rows.append(dict(year=year,weather=weather,month=month,site=name,node=zone,power_MW=hourly_raw*eta,
                          hours=w.HOURS,mu=f["MEF_t_per_MWh_e"],D=f["D_fossil_per_MWh"],
                          residual=max(re,ue),energy_closure_MWh=f["balance_residual_MWh"]))
    finally:
        w.model36.build_problem=w.ORIGINAL_BUILD
    tmp=out.with_suffix(".part")
    pd.DataFrame(rows).to_csv(tmp,index=False)
    os.replace(tmp,out)


if __name__=="__main__":
    (w.HERE/"experiments/matched").mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(trial,product(w.YEARS,w.WEATHERS,w.MONTHS)))
