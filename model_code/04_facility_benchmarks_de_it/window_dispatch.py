"""Paired chronological dispatch in the first calendar week of each quarter.

Run from any directory. Each case solves the frozen R0 and a local perturbation
under identical boundary stocks, prices and capacities. No weather extrapolation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "dataset_one/50_decarbonization_trajectory_v37/scripts"))
import run_case37 as legacy
import accounting37 as account
import flexload37
import model36

K = 0.98 * 0.96
MONTHS = (1, 4, 7, 10)
YEARS = (2028, 2030, 2033, 2035)
WEATHERS = (17, 6)
HOURS = 168
ORIGINAL_BUILD = model36.build_problem


def window_data(year, weather, month=None, *, hour0=None, hours=HOURS):
    full, _ = legacy.load_input(dict(year=year, weather=weather, resource="Central", use="R0"))
    dates = pd.to_datetime(full["timestamp_ns"])
    start = hour0 if hour0 is not None else np.flatnonzero((dates.month == month) & (dates.day == 1) & (dates.hour == 0)).item()
    stop = start + hours
    if start < 0 or stop > 8760 or hours % 24:
        raise ValueError((start, hours))
    d = {k: v[start:stop] if getattr(v, "ndim", 0) and v.shape[0] == 8760 else v for k, v in full.items()}
    source = legacy.r0_dir(year, True) / (legacy.r0_name(year, weather, True) + ".npz")
    with np.load(source) as ref:
        hydro = ref["hydro_inventory"]
        d["hydro_initial"] = hydro[start - 1] if start else full["hydro_initial"]
        d["hydro_terminal"] = hydro[stop - 1]
        storage = ref["existing_battery_inventory"]
        initial = storage[start - 1] if start else 0.5 * d["battery_energy"]
        terminal = storage[stop - 1]
        scheduled = ref["idsr_consumption"][start:stop]
    return d, initial, terminal, scheduled, full["idsr_window"], start


def boundary_states(p, d, b0, b1, scheduled, full_windows):
    # A storage equation joins hour zero to the terminal hour in the legacy
    # cyclic matrix. Shift its RHS to enforce two possibly different stocks.
    ids = p["builder"].blocks["existing_battery_inventory"]
    eq = p["eq"]
    col = eq.tocsc()
    for zone in range(ids.shape[1]):
        j = ids[-1, zone]
        rows = col.indices[col.indptr[j]:col.indptr[j + 1]]
        fixed = next(i for i in rows if eq.indptr[i+1] - eq.indptr[i] == 1)
        first = next(i for i in rows if eq[i, j] < 0)
        p["rhs"][fixed] = b1[zone]
        p["rhs"][first] = b0[zone] - b1[zone]
    # Preserve archived R0 energy inside any truncated iDSR window.
    builder = p["builder"]
    icons = builder.blocks["idsr_consumption"]
    for g in p["idsr_groups"]:
        k = g["column"]
        for idx, label in enumerate(g["labels"]):
            mask = g["inverse"] == idx
            if mask.sum() == (full_windows[:, k] == label).sum():
                continue
            hour = int(np.flatnonzero(mask)[0])
            j = icons[hour, k]
            rows = col.indices[col.indptr[j]:col.indptr[j+1]]
            row = next(i for i in rows if i < p["balance_start"] and eq[i, j] > 0)
            p["rhs"][row] = float(scheduled[mask, k].sum())


def injections(d, year, basic):
    gas = d["resource_q_hourly"] / K
    yield "region_raw", gas, 0.40
    if not basic:
        return
    yield "region_legacy_dose", gas * K, 0.40
    # Berlin raw volume is reported, methane fraction is a scenario.
    # Pinerolo's CHP-period monthly median is used solely to set the dose.
    site = pd.read_csv(HERE / "source_data/facility_inputs.csv").set_index("key")["value"]
    berlin_raw = site["de_raw_volume"] * site["de_raw_ch4_assumed"] * site["ch4_lhv"] / 1000
    italy_raw = site["it_chp_input_month"] * 12
    for key, zone, raw, eff in [("Ruhleben","DE00",berlin_raw,0.4),
                                ("Pinerolo","ITN1",italy_raw,site["it_chp_electric"])]:
        profile = np.zeros_like(gas)
        profile[:, list(d["zones"]).index(zone)] = raw / 8760
        yield key, profile, eff
    if year in (2028, 2035):
        for country in ("SI", "HR"):
            profile = np.where(d["countries"] == country, gas, 0.)
            yield country + "_actual", profile, 0.4
            # Same floor as V16 converted to raw energy, same within-day shape.
            total_raw_MW = d["resource_q"][d["countries"] == country].sum() / K
            yield country + "_floor", profile * max(1., (20. / K) / total_raw_MW), 0.4


def solve(d, price, avail, setup, raw=None, eff=0.4):
    dj = dict(d)
    if raw is not None:
        dj["demand"] = d["demand"] - eff * raw
    p = model36.build_problem(dj, tech="none", gas_cut=0.30, external_availability=avail,
                              costs={"system_gas_eur_per_MWh_el": d["gas_cost"],
                                     "system_external_eur_per_MWh":price})
    boundary_states(p, dj, *setup)
    ans = model36.solve(p, time_limit=600, method="highs-ds")
    if not ans["success"]:
        raise RuntimeError(ans["message"])
    arr = ans["arrays"]
    for k in ("thermal_technology", "thermal_efficiency", "gas_efficiency"):
        arr[k] = d[k]
    arr["flex_type"], arr["flex_eff"] = p["flex_units"]["type"], p["flex_units"]["eff"]
    residual = max(ans[k] for k in ("equality_error_MWh", "inequality_error", "bound_error"))
    return arr, residual


def run(key):
    year, weather, month = key
    folder = HERE / "experiments/windows"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{year}_ws{weather:02}_m{month:02}.csv"
    if out.exists():
        return
    start_time = time.monotonic()
    d, b0, b1, sched, full_win, start = window_data(year, weather, month)
    setup = b0, b1, sched, full_win
    costs = pd.read_csv(HERE / "source_data/reference/eraa_dispatch_costs.csv").set_index("year")
    linked = 100 * costs.loc[year, "gas_CCGT_EUR_per_MWhe"] / costs.loc[2028, "gas_CCGT_EUR_per_MWhe"]
    flexload37.patch(model36, year)
    records = []
    try:
        for capacity in ("all_hours", "event_only"):
            avail = 0.5 if capacity == "all_hours" else 1.0 - 0.5 * d["event"][:, None]
            price_types = [("fixed",100.)] if year == 2028 else [("fixed",100.),("linked",linked)]
            for tariff, price in price_types:
                r, rerr = solve(d, price, avail, setup)
                basic = (capacity,tariff) == ("all_hours","fixed")
                for use, raw, eff in injections(d, year, basic):
                    u, uerr = solve(d, price, avail, setup, raw, eff)
                    u["resource_q_hourly"] = raw
                    pair = account.pair_metrics(u, r)
                    metrics = account.finish_u1(pair, eff)
                    if abs(metrics["balance_residual_relative"]) > 5e-5:
                        raise ValueError((key,use,metrics))
                    supply = eff * raw.sum()
                    row = dict(year=year, weather=weather, month=month, hour0=start, hours=HOURS,
                               event_hours=int(d["event"].sum()), availability=capacity, price=tariff,
                               price_eur_MWh=price, intervention=use, raw_MWh=raw.sum(), electricity_MWh=supply,
                               D=metrics["D_fossil_per_MWh"], avoided_t=pair["avoided_GWP100_t"],
                               avoided_t_per_raw=pair["avoided_GWP100_t"]/raw.sum(),
                               electricity_MEF=metrics["MEF_t_per_MWh_e"], baseline_import_MWh=r["external_import"].sum(),
                               import_share=pair["balance_reduced_external_import_MWh"]/supply,
                               EENS_base_MWh=r["unserved"].sum(), EENS_case_MWh=u["unserved"].sum(),
                               residual=max(rerr,uerr), energy_closure_MWh=metrics["balance_residual_MWh"],
                               reuse_2028_price_identity=False)
                    records.append(row)
                    if year == 2028 and use == "region_raw":
                        records.append(dict(row, price="linked", reuse_2028_price_identity=True))
    finally:
        model36.build_problem = ORIGINAL_BUILD
    tmp = out.with_suffix(".part")
    pd.DataFrame(records).to_csv(tmp,index=False)
    os.replace(tmp,out)
    print(out.name,len(records),round(time.monotonic()-start_time,1),"s",flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--workers",type=int,default=4)
    cli.add_argument("--one",nargs=3,type=int,metavar=("YEAR","WS","MONTH"))
    args = cli.parse_args()
    if args.one:
        run(args.one)
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(run,product(YEARS, WEATHERS, MONTHS),chunksize=1))
