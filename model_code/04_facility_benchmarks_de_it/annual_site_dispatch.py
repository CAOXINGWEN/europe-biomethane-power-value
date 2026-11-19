"""Complete 8760-hour facility doses with fixed weekly boundary stocks."""
import argparse
from calendar import monthrange
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
import window_dispatch as w


def run(key):
    year, weather, block, grid = key
    if grid == "week":
        start = 168 * block
        hours = min(168, 8760 - start)
        group = "site_year"
    else:
        # All four ERAA target calendars omit leap day.
        lengths = [24 * monthrange(2035,m)[1] for m in range(1,13)]
        start, hours = sum(lengths[:block]), lengths[block]
        group = "site_month"
    out = w.HERE / "experiments" / group / f"{year}_ws{weather:02}_b{block:02}.csv"
    if out.exists():
        return
    d, b0, b1, sched, labels, _ = w.window_data(year,weather,hour0=start,hours=hours)
    setup = b0,b1,sched,labels
    w.flexload37.patch(w.model36,year)
    try:
        r,rerr = w.solve(d,100.,0.5,setup)
        rows = []
        for name,raw,eff in w.injections(d,year,True):
            if name not in ("Ruhleben","Pinerolo"):
                continue
            u,uerr = w.solve(d,100.,0.5,setup,raw,eff)
            u["resource_q_hourly"] = raw
            m = w.account.pair_metrics(u,r)
            f = w.account.finish_u1(m,eff)
            if abs(f["balance_residual_relative"]) > 5e-5:
                raise ValueError((key,name,f["balance_residual_relative"]))
            rows.append(dict(year=year,weather=weather,block=block,start=start,hours=hours,site=name,
                             raw_MWh=raw.sum(),electricity_MWh=raw.sum()*eff,avoided_t=m["avoided_GWP100_t"],
                             fossil_MWh=m["fossil_fuel_displaced_MWh"],import_MWh=m["balance_reduced_external_import_MWh"],
                             residual=max(rerr,uerr),energy_closure_MWh=f["balance_residual_MWh"]))
    finally:
        w.model36.build_problem=w.ORIGINAL_BUILD
    tmp=out.with_suffix(".part")
    pd.DataFrame(rows).to_csv(tmp,index=False)
    os.replace(tmp,out)
    if block in (0,26,52):
        print(out.name,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--workers",type=int,default=6)
    cli.add_argument("--grid",choices=["week","month"],default="week")
    a=cli.parse_args()
    monthly = a.grid == "month"
    group = "site_month" if monthly else "site_year"
    weathers = (17,) if monthly else (17,6,9,15,36,31,33,20)
    (w.HERE/"experiments"/group).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:
        list(pool.map(run,product((2028,2035),weathers,range(12 if monthly else 53),(a.grid,))))
