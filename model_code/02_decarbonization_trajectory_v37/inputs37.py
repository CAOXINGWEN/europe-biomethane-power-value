"""Target-year 24-zone inputs: V27 system layer + V32 corrected layer (full, mapped DSR) + V32 B gas role.

The V27 layer is copied from 39_.../system/prepare_inputs.py with YEAR/YI as parameters and the
timestamp taken from the no-29-February ERAA calendar. The V32 layer imports the frozen
44_.../scripts/prepare_inputs.py and only replaces its module-level YEAR/CAL; the NTC limit reader is
re-implemented so that the 24 rows of 29.02 in the leap-year TY2028 workbook are dropped.
Usage: inputs37.py --years 2028 2030 2033 --weathers 1 2 ... [--check2035 1 13 36]
"""
from pathlib import Path
import sys, json, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common37 import (HERE, ROOT, DS, RAW, DASH, V2, V27, V32, YI, now, sha, dump, arrays, save_arrays,
                      load_module, v2_module, calendar, gas_role, input_path)

V2C = V2 / '01_inputs_compact'
LEDGER = HERE / 'inputs/system/ledgers'


def build27(year, weather, other_cf=.85):
    v2 = v2_module(); profile_from_constraint, hourly_inflow = v2.profile_from_constraint, v2.hourly_inflow
    YEAR = year; yi = YI[year]
    def load(name): return np.load(V2C / (name + '_v2.npy'), mmap_mode='r')
    z = pd.read_csv(V2C / 'active_eraa24_zone_order_v2.csv'); zones = z.bidding_zone.tolist(); lookup = {s: i for i, s in enumerate(zones)}; T = 8760; Z = len(z)
    d = {k: np.asarray(load(f'{k}_zone_hourly_mw')[yi, weather - 1], float) for k in ['demand', 'vre']}
    d['ror'] = np.asarray(load('hydro_ror_generation_zone_hourly_mw')[yi, weather - 1], float)
    groups = pd.read_csv(V2C / 'thermal_equivalent_unit_groups_v2.csv'); groups = groups[groups.target_year.eq(YEAR)]
    sched = load('thermal_scheduled_group_hourly_mw')[:, groups.group_id.to_numpy()]
    d['gas_cap'] = np.zeros((T, Z)); d['other_thermal_cap'] = np.zeros((T, Z)); bio = np.zeros((T, Z))
    for j, r in enumerate(groups.itertuples()):
        cap = np.asarray(sched[:, j], float) * (1 - r.forced_outage_rate_fraction); zi = lookup[r.bidding_zone]
        if r.technology == 'Natural gas': d['gas_cap'][:, zi] += cap
        else: d['other_thermal_cap'][:, zi] += cap
        if r.technology == 'Biofuel': bio[:, zi] += cap
    raw = pd.read_csv(DASH / 'GenerationCapacities.csv'); raw = raw[raw.data_version.eq('ERAA 2025 final') & raw['Target year'].eq(YEAR) & raw.Market_Node.isin(zones)].copy()
    statuses = raw.Operational_Status.fillna(''); included = raw[statuses.eq('Available on market') | statuses.str.contains('Inelastic supply', case=False)]
    other = included[included.Technology_Simplified.eq('Others RES')]
    fixed = np.zeros(Z); small = np.zeros(Z)
    for r in other.itertuples():
        zi = lookup[r.Market_Node]; fixed[zi] += r.Value * other_cf
        if r.Technology == 'Small biomass': small[zi] += r.Value * other_cf
    fixedgas = included[included.Technology.eq('Natural gas') & included.Operational_Status.str.contains('Inelastic')]
    for r in fixedgas.itertuples(): d['gas_cap'][:, lookup[r.Market_Node]] += r.Value * .92
    qtab = pd.read_csv(DS / '37_base_supply_boost_v25/application/eraa_resource_allocation.csv')
    qtab = qtab[qtab.target_year.eq(YEAR) & qtab.resource.eq('Central')].set_index('zone')
    q = qtab.upgraded_q_MW_fuel.reindex(zones).to_numpy(); base = .4 * .75 * q
    from_small = np.minimum(base, small); from_bio = base - from_small
    feasible = (bio.min(axis=0) + 1e-8 >= from_bio)
    if not feasible.all(): raise ValueError('Base pool mapping infeasible: ' + str(np.array(zones)[~feasible]))
    d['other_thermal_cap'] -= from_bio[None, :]
    d['fixed_other'] = np.broadcast_to(fixed - from_small + base, (T, Z)).copy()
    d['q'] = q; d['base'] = base; d['zones'] = np.array(zones); d['countries'] = z.country_code.to_numpy(dtype=str)
    audit = pd.DataFrame(dict(zone=zones, food_waste_q_MW_fuel=q, base_MW=base, small_biomass_fixed_MW=small, biofuel_min_available_MW=bio.min(axis=0),
                              base_from_small_MW=from_small, base_from_thermal_biofuel_MW=from_bio, base_pool_mapping_feasible=feasible))
    h = pd.read_csv(V2C / 'hydro_asset_audit_v2.csv'); h = h[h.target_year.eq(YEAR) & ~h.technology.eq('Run of river')].copy().reset_index(drop=True)
    arr = {k: [] for k in ['hydro_inflow', 'hydro_gen_cap', 'hydro_pump_cap', 'hydro_min', 'hydro_max']}
    for r in h.itertuples():
        cp = ROOT / r.constraint_file if isinstance(r.constraint_file, str) else None
        ip = ROOT / r.inflow_file if isinstance(r.inflow_file, str) else None
        turb, pump, smin, smax = profile_from_constraint(cp, r.turbine_nameplate_mw, r.pump_nameplate_mw)
        arr['hydro_inflow'].append(hourly_inflow(ip)[weather - 1]); arr['hydro_gen_cap'].append(turb); arr['hydro_pump_cap'].append(pump)
        arr['hydro_min'].append(smin * r.storage_capacity_mwh); arr['hydro_max'].append(smax * r.storage_capacity_mwh)
    for k, v in arr.items(): d[k] = np.array(v).T.astype(float)
    d['hydro_zone'] = h.bidding_zone.map(lookup).to_numpy(); d['hydro_initial'] = h.initial_fraction.to_numpy() * h.storage_capacity_mwh.to_numpy()
    d['hydro_terminal'] = h.terminal_fraction.to_numpy() * h.storage_capacity_mwh.to_numpy(); d['hydro_capacity'] = h.storage_capacity_mwh.to_numpy()
    bat = pd.read_csv(DASH / 'Batteries additional information.csv'); bat = bat[bat.TARGET_YEAR.eq(YEAR) & bat.data_version.eq('ERAA 2025 final') & bat.OP_STAT.eq('Available on market') & bat.MARKET_NODE.isin(zones)]
    bg = bat.groupby('MARKET_NODE')[['MAX LOAD CAP (MW)', 'STORAGE CAPACITY (MWh)']].sum().reindex(zones, fill_value=0)
    d['battery_power'] = bg['MAX LOAD CAP (MW)'].to_numpy(float); d['battery_energy'] = bg['STORAGE CAPACITY (MWh)'].to_numpy(float)
    d['battery_discharge_power'] = included[included.Technology.str.startswith('Battery')].groupby('Market_Node').Value.sum().reindex(zones, fill_value=0).to_numpy(float)
    ntc = np.asarray(load('ntc_expected_available_zone_hourly_mw')[yi], float); edge = np.argwhere(ntc.max(axis=0) > 0)
    d['edge_from'] = edge[:, 0]; d['edge_to'] = edge[:, 1]; d['edge_cap'] = ntc[:, edge[:, 0], edge[:, 1]]
    d['import_cap'] = np.asarray(load('import_limit_zone_hourly_mw')[yi], float); d['export_cap'] = np.asarray(load('export_limit_zone_hourly_mw')[yi], float)
    d['import_cap'] = np.where(np.isinf(d['import_cap']), 1e7, d['import_cap'])
    d['export_cap'] = np.where(np.isinf(d['export_cap']), 1e7, d['export_cap'])
    d['external_cap'] = np.asarray(np.load(DS / '15_reviewer_priority_model_v3/01_inputs/external_inbound_interface_zone_hourly_mw_v3.npy', mmap_mode='r')[yi], float)
    dsr = pd.read_csv(V2C / 'explicit_dsr_duration_groups_v2.csv'); dsr = dsr[dsr.target_year.eq(YEAR) & dsr.capacity_mw.gt(0)].copy()
    d['dsr_zone'] = dsr.bidding_zone.map(lookup).to_numpy(); d['dsr_cap'] = dsr.capacity_mw.to_numpy(); d['dsr_hours'] = dsr.max_hours_per_day.clip(upper=24).to_numpy()
    with np.load(DS / f'19_tail_reliability_v7/01_inputs/packed_{YEAR}.npz') as e: d['event'] = e['events'][weather - 1].astype(bool)
    d['year'] = np.array(YEAR); d['weather'] = np.array(weather); d['other_cf'] = np.array(other_cf)
    d['timestamp_ns'] = calendar(YEAR).astype('int64').to_numpy()
    d['compression_kappa'] = np.array(.03); d['eta_bio'] = np.array(.4); d['beta'] = np.array(.75); d['battery_eta'] = np.array(np.sqrt(.9)); d['pump_eta'] = np.array(.8)
    for k, v in d.items():
        if v.dtype.kind in 'fc' and np.isnan(v).any(): raise ValueError(k)
    return d, dict(base_mapping=audit, hydro_assets=h, batteries=bat, explicit_dsr=dsr)


def limits37(A, zones, year, T=8760):
    """V32 corrected_limits with the 29.02 rows of a leap-year workbook removed before the calendar check."""
    p = RAW / f'ERAA_2025_NTCs/NTCs/NTCs Consolidated TY{year}.xlsx'
    f = A.read_excel(str(p), 'Limits', None)
    mapping = {'Net import limit': 'net_import_cap', 'Net export limit': 'net_export_cap',
               'Gross import limit': 'gross_import_cap', 'Gross export limit': 'gross_export_cap'}
    result = {v: np.full((T, len(zones)), 1e7) for v in mapping.values()}; ledger = []
    body = f.iloc[10:]
    leap = body.iloc[:, 0].astype(str).str.strip().eq('29.02.')
    body = body[~leap]
    assert len(body) == T, (year, len(body))
    rawtime = pd.to_datetime(body.iloc[:, 0].astype(str).str.strip() + str(year), format='%d.%m.%Y') + pd.to_timedelta(pd.to_numeric(body.iloc[:, 1]) - 1, unit='h')
    assert np.array_equal(rawtime.to_numpy(), A.CAL.to_numpy())
    for col in range(2, len(f.columns)):
        node = str(f.iloc[7, col]).split(' - ', 1)[0]; kind = str(f.iloc[8, col])
        if node not in zones or kind not in mapping: continue
        values = pd.to_numeric(body.iloc[:, col], errors='coerce').to_numpy(float)
        if not np.isfinite(values).all(): raise ValueError('Missing NTC limit ' + node + '/' + kind)
        result[mapping[kind]][:, zones.index(node)] = np.minimum(result[mapping[kind]][:, zones.index(node)], values)
        ledger.append(dict(zone=node, raw_curve_type=kind, input_field=mapping[kind], minimum_MW=float(values.min()), maximum_MW=float(values.max())))
    return result, ledger, int(leap.sum())


_A = {}
def v32a(year):
    """Frozen V32 module with module-level YEAR/CAL set to the target year (one module per year)."""
    if year not in _A:
        A = load_module(V32 / 'scripts/prepare_inputs.py', f'v32a_{year}')
        A.YEAR = year; A.CAL = calendar(year)
        _A[year] = A
    return _A[year]


def build(year, weather):
    d27, side = build27(year, weather)
    A = v32a(year); gas_split = gas_role(weather)
    d = d27; zones = list(d['zones']); raw, eva = A.official_inputs(zones)
    d['q_hourly'], d['source_index'] = A.gas_shape(d, gas_split)
    d['gross_demand'] = d['demand'].copy()
    d['legacy_gas_cap'] = d['gas_cap'].copy(); d['legacy_other_thermal_cap'] = d['other_thermal_cap'].copy()
    pv, bat, btmledger = A.btm_inputs(d, raw, weather)
    d['btm_pv'] = pv; d['btm_battery_signed'] = bat
    d['demand'] = d['gross_demand'] - pv - bat
    if d['demand'].min() < 0: raise ValueError('Negative net demand needs explicit supply-side treatment')
    capledger = A.apply_eva(d, raw, eva, restore_hydrogen=True)
    dsrledger = A.dsr_inputs(d, raw, eva, mapped=True)
    lim, limledger, leap_rows = limits37(A, zones, year)
    d.update(lim)
    idsledger = A.idsr_inputs(d, weather, enabled=True)
    d['input_variant'] = np.array('full'); d['dsr_variant'] = np.array('mapped'); d['gas_split'] = np.array(gas_split)
    d['hydrogen_fuel_boundary'] = np.array('Exogenous priced fuel, no verified annual delivery cap; separately reported assumption')
    for k, v in d.items():
        if v.dtype.kind in 'fc' and not np.isfinite(v).all(): raise ValueError('Nonfinite input ' + k)
    assert d['gas_cap'].min() >= -1e-6 and d['other_thermal_cap'].min() >= -1e-6
    ledgers = dict(side, btm=pd.DataFrame(btmledger), capacity=pd.DataFrame(capledger), dsr=pd.DataFrame(dsrledger),
                   idsr=pd.DataFrame(idsledger), limits=pd.DataFrame(limledger), eva_rows=eva)
    return d, ledgers, dict(ntc_leap_rows_removed=leap_rows)


def compare(a, b):
    """Max absolute difference per key between two input dictionaries (strings must match exactly)."""
    out = {}; keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in a or k not in b: out[k] = 'missing in ' + ('new' if k not in a else 'reference'); continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape: out[k] = f'shape {x.shape} vs {y.shape}'; continue
        if x.dtype.kind in 'fcib' and y.dtype.kind in 'fcib':
            out[k] = float(np.abs(x.astype(float) - y.astype(float)).max(initial=0))
        else: out[k] = 0. if np.array_equal(x, y) else 'different'
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--years', nargs='*', type=int, default=[])
    ap.add_argument('--weathers', nargs='*', type=int, default=list(range(1, 37)))
    ap.add_argument('--check2035', nargs='*', type=int, default=[]); a = ap.parse_args()
    for w in a.check2035:
        d, _, _ = build(2035, w); ref = arrays(input_path(2035, w)); diff = compare(d, ref)
        bad = {k: v for k, v in diff.items() if not (isinstance(v, float) and v <= 1e-9)}
        rec = dict(utc=now(), weather=w, reference=str(input_path(2035, w).relative_to(ROOT)), passed=not bad, mismatches=bad,
                   max_abs_difference=max([v for v in diff.values() if isinstance(v, float)], default=0.), keys=len(diff))
        dump(HERE / 'qa' / f'reproduce_inputs_2035_WS{w:02d}.json', rec); print(json.dumps({k: rec[k] for k in ['weather', 'passed', 'max_abs_difference']}), bad or '', flush=True)
    for y in a.years:
        assert y in (2028, 2030, 2033)
        for w in a.weathers:
            path = input_path(y, w)
            if path.exists(): print('EXISTS', path.name, flush=True); continue
            d, ledgers, meta = build(y, w); save_arrays(path, d)
            if w == 13 or not (LEDGER / f'{y}_WS13_capacity.csv').exists():
                for kind, frame in ledgers.items(): frame.to_csv(LEDGER / f'{y}_WS{w:02d}_{kind}.csv', index=False)
            dump(HERE / 'inputs/system' / f'{y}_WS{w:02d}_manifest.json', dict(utc=now(), year=y, weather=w, gas_split=str(d['gas_split']), sha256=sha(path), **meta,
                 code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), Path(__file__).with_name('common37.py'), V32 / 'scripts/prepare_inputs.py']}))
            print('READY', y, w, round(float(d['demand'].sum() / 1e6), 3), 'TWh net demand', flush=True)


if __name__ == '__main__':
    LEDGER.mkdir(parents=True, exist_ok=True); main()
