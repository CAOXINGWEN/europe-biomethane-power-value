"""Hold 2035 raw supply fixed and recompute all eight endpoints per trade rule."""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
import window_dispatch as w

sys.path.insert(0,str(w.ROOT/"dataset_one/51_mechanism_boundaries_v38/scripts"))
import common38 as factorial

ENDS=(2028,2035)


def cell(key):
    bg, vre, priceyear, weather, month = key
    out=w.HERE/"experiments/factorial"/f"B{bg}_V{vre}_C{priceyear}_ws{weather:02}_m{month:02}.csv"
    if out.exists():
        return
    d,b0,b1,sched,labels,start=w.window_data(bg,weather,month)
    donor=factorial.build_input(dict(year=bg,weather=weather,vre_year=vre,price_year=priceyear,use="R0"))
    sl=slice(start,start+w.HOURS)
    d["vre"]=donor["vre"][sl]
    d["gas_cost"],d["thermal_cost"]=donor["gas_cost"],donor["thermal_cost"]
    raw=donor["resource_q_hourly"][sl]/w.K
    costs=pd.read_csv(w.HERE/"source_data/reference/eraa_dispatch_costs.csv").set_index("year")
    index=costs.loc[priceyear,"gas_CCGT_EUR_per_MWhe"]/costs.loc[2028,"gas_CCGT_EUR_per_MWhe"]
    w.flexload37.patch(w.model36,bg)
    rows=[]
    try:
        for trade,price,avail in [("flat_all",100.,.5),
                                  ("linked_event",100.*index,1.-.5*d["event"][:,None])]:
            r,re=w.solve(d,price,avail,(b0,b1,sched,labels))
            u,ue=w.solve(d,price,avail,(b0,b1,sched,labels),raw)
            u["resource_q_hourly"]=raw
            m=w.account.pair_metrics(u,r)
            f=w.account.finish_u1(m)
            if abs(f["balance_residual_relative"]) > 5e-5:
                raise ValueError((key,trade,f))
            rows.append(dict(background=bg,vre=vre,cost=priceyear,weather=weather,month=month,trade=trade,
                             raw_MWh=raw.sum(),D=f["D_fossil_per_MWh"],residual=max(re,ue),
                             import_share=m["balance_reduced_external_import_MWh"]/(0.4*raw.sum()),
                             energy_closure_MWh=f["balance_residual_MWh"]))
    finally:
        w.model36.build_problem=w.ORIGINAL_BUILD
    tmp=out.with_suffix(".part")
    pd.DataFrame(rows).to_csv(tmp,index=False)
    os.replace(tmp,out)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--workers",type=int,default=4)
    a=cli.parse_args()
    (w.HERE/"experiments/factorial").mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(cell,product(ENDS,ENDS,ENDS,w.WEATHERS,w.MONTHS)))
