"""Run one registered V37 case (two LP stages) with the frozen V36 model and COPT backend.

Uses: R0 counterfactual (q=0); U1 run-of-plant site power (demand - 0.40*hourly gas); U2 base 0.75 + boost
reserve (V36 B3 design scaled to the target-year resource). Options: single-country injection, scale factor,
nuclear output fixed at the same-year R0 dispatch, no event gas derating.
Usage: run_case37.py --job NAME
"""
from pathlib import Path
import sys, os, json, time, argparse, traceback, fcntl, hashlib
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common37 import HERE, ROOT, V27, V36, TRAIN, POWER_V36, now, sha, dump, arrays, input_path, thermal_path

sys.path.insert(0, str(V36 / 'scripts'))
LUO = ROOT / 'dataset_one/39_public_validation_system_comparison_v27/station/processed/luo_measured_production_hourly.csv'
RESOURCE = HERE / 'inputs/resource_v37_country_year.csv'
BOOST_VOM = 8.933747986654735          # V36 dispatch_cost_units.json (DEA biogas engine 2030 VOM, EUR2024)
FLOOR_MW = 20.                          # registered minimum single-country injection (MW fuel)
RAW_FLEX = ROOT / 'dataset_one/07_eraa_preprocessed/01_extracted_raw/ERAA_2025_DemandData/Demand data/Additional dispatchable consumption.csv'


def weights(year):
    p = V36 / 'inputs/resource_zone_weights.npz' if year == 2035 else HERE / f'inputs/system/resource_zone_weights_{year}.npz'
    return arrays(p)['weights']


def v36_central_vector():
    """V36 Central deliverable resource (2020-2023 mean, UK 2021-2022) on V36 2035 zone weights."""
    sys.path.insert(0, str(V36 / 'scripts')); import common36
    f = common36.resource_table(); by = f[f.scenario.eq('Central')].set_index('country_code').deliverable_fuel_TWh_LHV
    w = arrays(V36 / 'inputs/resource_zone_weights.npz')
    return np.array([by[c] * 1e6 / 8760 * w['weights'][j] for j, c in enumerate(w['countries'])])


def resource_vector(job, countries):
    """Zone mean fuel power q (MW) for the job; returns (q, scale_factor)."""
    kind = job.get('resource', 'none')
    if kind == 'none': return np.zeros(len(countries)), 1.
    if kind == 'V36Central': q = v36_central_vector()
    else:
        t = pd.read_csv(RESOURCE)
        t = t[t.year.eq(int(job['resource_year'] if 'resource_year' in job else job['year'])) & t.scenario.eq(kind)].set_index('country_code')
        col = 'fleet_fuel_TWh' if kind == 'fleet' else 'deliverable_TWh'
        q = np.array([t.loc[c, col] * 1e6 / 8760 for c in countries]) * weights(int(job['year']))
    scale = float(job.get('scale', 1.))
    if job.get('country'):
        q = np.where(countries == job['country'], q, 0.)
        if q.sum() * scale < FLOOR_MW: scale = FLOOR_MW / q.sum()
        if job.get('scale_mode') == 'x5': scale *= 5.          # B4 L3: five times the B3 injection
    return q * scale, scale


def load_input(job):
    year, w, use = int(job['year']), int(job['weather']), job['use']
    d = arrays(input_path(year, w))
    # amendment_01: initial/terminal reservoir levels clipped into the ERAA hourly [min, max] band of the first/last
    # hour (only PT00 open-loop pumping in 2028 is affected: default 0.5 above MAX_RES 0.405/0.468); no-op for 2035.
    d['hydro_initial'] = np.minimum(np.maximum(d['hydro_initial'], d['hydro_min'][0]), d['hydro_max'][0])
    d['hydro_terminal'] = np.minimum(np.maximum(d['hydro_terminal'], d['hydro_min'][-1]), d['hydro_max'][-1])
    legacy_base = d['base'].copy(); fixed = d['fixed_other'] - legacy_base[None, :]
    q, scale = resource_vector(job, d['countries']); d['resource_q'] = q.copy()
    f = pd.read_csv(LUO)
    x = f.observed_biogas_m3_customer_h.to_numpy()[slice(0, 240) if str(d['gas_split']) == 'train' else slice(240, 480)]
    phase = np.array([int(hashlib.sha256(str(z).encode()).hexdigest()[:8], 16) % len(x) for z in d['zones']])
    raw = x[(np.arange(8760)[:, None] + phase) % len(x)]; hourly = raw / raw.mean(axis=0) * q[None, :]
    d['resource_q_hourly'] = hourly.copy(); d['beta'] = np.array(float(job.get('beta', .75))); d['fixed_other'] = fixed
    if use == 'U2':
        d['q'] = q; d['q_hourly'] = hourly; d['base'] = float(d['beta']) * .4 * q
        d['fixed_other'] = fixed + d['base'][None, :]
    else:
        d['q'] = np.zeros_like(q); d['q_hourly'] = np.zeros_like(hourly); d['base'] = np.zeros_like(q)
    if use == 'U1': d['demand'] = d['demand'] - .4 * hourly
    for k, v in arrays(thermal_path(year)).items():
        if k != 'zones': d[k] = v
    return d, scale


def r0_name(year, w, flex=False):
    # amendment_post_02: main specification (flex) pairs with F0 counterfactuals for every year, including 2035.
    if flex: return f'F0_{year}_R0_WS{w:02d}'
    return f'R0_WS{w:02d}' if year == 2035 else f'B0_{year}_R0_WS{w:02d}'


def r0_dir(year, flex=False):
    return HERE / 'results' if (flex or year != 2035) else V36 / 'results'


def r0_arrays(year, w, flex=False):
    p = r0_dir(year, flex) / (r0_name(year, w, flex) + '.npz')
    return arrays(p), p


def capped_proportional(weights_, caps, total):
    weights_ = np.asarray(weights_, float); caps = np.asarray(caps, float); out = np.zeros_like(caps)
    if total > caps.sum() + 1e-8: raise ValueError('Budget exceeds caps')
    for _ in range(len(caps) + 2):
        rem = total - out.sum(); room = np.maximum(caps - out, 0); active = room > 1e-12
        if rem <= 1e-10: break
        w = np.where(active, weights_, 0)
        if w.sum() <= 0: w = active.astype(float)
        out += np.minimum(room, rem * w / w.sum())
    if abs(out.sum() - total) > 1e-7: raise ValueError('Allocation did not close')
    return out


def design(year, scenario, q, flex=False):
    """V36 rule_eens design on the same year's R0 training weather; power scaled by resource relative to V36 Central."""
    p = HERE / 'inputs' / f"design_{year}_{scenario}{'_flex' if flex else ''}.json"
    if p.exists(): return json.loads(p.read_text())
    wts = np.zeros(len(q)); sources = []
    for w in TRAIN:
        rp = r0_dir(year, flex) / (r0_name(year, w, flex) + '.json'); r = json.loads(rp.read_text())
        if not r.get('success'): raise ValueError('Training R0 failed')
        wts += np.array(r['zone_EENS_MWh']); sources.append(dict(path=str(rp.relative_to(ROOT)), sha256=sha(rp)))
    caps = .4 * (12 - .75) * q; total = POWER_V36 * q.sum() / v36_central_vector().sum()
    power = capped_proportional(wts, caps, total)
    value = dict(year=year, scenario=scenario, power=power.tolist(), energy=(22 * q).tolist(), weights=wts.tolist(), total_power_MW=total,
                 uniform_fallback=bool(wts[q > 0].sum() <= 0), train_weather=TRAIN, training_sources=sources, holdout_used_for_fitting=False)
    dump(p, value); return value


def execute(job):
    import model36 as model
    import solver36
    def lp(*a, **kw):
        kw['options'] = dict(kw.get('options') or {}, threads=4); return solver36.linprog(*a, **kw)
    model.linprog = lp
    start = time.time(); started = now(); year, w, use = int(job['year']), int(job['weather']), job['use']
    flex = bool(job.get('flex')) or job.get('sens') == 'flex_load'
    d, scale = load_input(job)
    P = np.zeros_like(d['q']); S = P.copy(); allowed = None
    if use == 'U2':
        ds = design(year, job['resource'], d['resource_q'], flex); P = np.array(ds['power']); S = np.array(ds['energy'])
        ra, _ = r0_arrays(year, w, flex); allowed = ra['unserved'].sum(axis=1) > .001
    gas_cut = 0. if job.get('sens') == 'no_gascut' else .3
    costs = dict(system_gas_eur_per_MWh_el=d['gas_cost'], boost_eur_per_MWh_el=BOOST_VOM, gas_charge_eur_per_MWh_fuel=.0001, system_external_eur_per_MWh=100.)
    if flex:          # amendment_post_01/02: ERAA electrolysers and power-to-heat as price-bid consumption
        import flexload37; flexload37.patch(model, year)
    p = model.build_problem(d, tech='biomethane' if use == 'U2' else 'none', power=P, energy=S, gas_cut=gas_cut, external_availability=.5, costs=costs, allow_boost_hours=allowed)
    fixed_from = None
    if job.get('sens') == 'nuclear_fixed':
        ra, rp = r0_arrays(year, w, flex); nuclear = np.asarray(ra['thermal_technology']) == 'Nuclear'
        fixed = ra['thermal_dispatch'][:, nuclear]; ids = p['builder'].blocks['thermal_dispatch'][:, nuclear].ravel()
        assert ids.size == fixed.size and np.array_equal(ra['thermal_technology'], d['thermal_technology'])
        p['bounds'][ids, 0] = fixed.ravel(); p['bounds'][ids, 1] = fixed.ravel(); fixed_from = str(rp.relative_to(ROOT))
    out = HERE / 'results'; name = job['name']
    os.environ['V36_SOLVER_LOG_PREFIX'] = str(HERE / 'logs' / name)
    p['stage1_save_prefix'] = str(out / (name + '_s1'))
    p['stage1_metadata'] = dict(parent_case=name, job=job, registry_sha256=sha(HERE / 'protocol/registration_v37.json'))
    deps = {str(x.relative_to(ROOT)): sha(x) for x in [V36 / 'scripts/model36.py', V36 / 'scripts/solver36.py', Path(__file__), Path(__file__).with_name('common37.py'),
                                                        input_path(year, w), thermal_path(year)] + ([RESOURCE] if job.get('resource', 'none') not in ('none', 'V36Central') else []) + ([Path(__file__).with_name('flexload37.py'), RAW_FLEX] if flex else [])}
    # amendment_04: after a failed attempt whose cost stage stopped on barrier time/iteration limits (COPT 8 or 11),
    # solve the cost stage with dual simplex (LpMethod 1) instead of barrier; the method used is recorded.
    import re
    prev = [x.read_text() for x in (HERE / 'results').glob(name + '.attempt_failed_*.json') if not x.name.endswith('_s1.json')]
    if any(re.search(r'operating_cost_tiebreak\\?", \\?"method\\?": \\?"copt\\?", \\?"status\\?": (8|11)', t) for t in prev):
        os.environ['V36_STAGE_METHODS'] = '1,1'
    stage_methods = os.environ.get('V36_STAGE_METHODS', '')
    result = model.solve(p, time_limit=7200, method='copt', secondary=True)   # amendment_03: 2 h per LP stage
    if not result['success']: raise RuntimeError(json.dumps({k: v for k, v in result.items() if k != 'arrays'}, default=str))
    a = result.pop('arrays'); s1 = result.pop('stage1_arrays', None); dual = result.pop('nodal_dual', None)
    if dual is not None: a['nodal_dual'] = dual
    a.update(thermal_zone=d['thermal_zone'], thermal_technology=d['thermal_technology'], thermal_efficiency=d['thermal_efficiency'], gas_efficiency=d['gas_efficiency'],
             zones=d['zones'], countries=d['countries'], resource_q_hourly=d['resource_q_hourly'], power_vector=P, buffer_vector=S)
    if 'flex_units' in p:
        fu = p['flex_units']; a.update(flex_zone=fu['zone'], flex_type=fu['type'], flex_cap=fu['cap'], flex_price=fu['price'], flex_eff=fu['eff'])
    un = np.maximum(a['unserved'], 0)
    result.update(job=job, started_utc=started, elapsed_seconds=time.time() - start, completed_utc=now(),
                  physical_verified=max(result['equality_error_MWh'], result['inequality_error'], result['bound_error'], result.get('annual_gas_balance_error_MWh', 0)) <= 2e-4,
                  zone_EENS_MWh=un.sum(axis=0).tolist(), zones=d['zones'].tolist(), countries=d['countries'].tolist(),
                  resource_MWh_by_zone=(d['resource_q'] * 8760).tolist(), base_generation_MWh_by_zone=(d['base'] * 8760).tolist(),
                  site_generation_MWh=float(.4 * d['resource_q_hourly'].sum()) if use == 'U1' else 0., injection_scale=scale,
                  gas_cut=gas_cut, nuclear_fixed_from=fixed_from, flexible_consumption=flex, stage_methods=stage_methods, dependencies_sha256=deps, registry_sha256=sha(HERE / 'protocol/registration_v37.json'))
    pn = out / (name + '.npz'); np.savez_compressed(pn, **a); result['arrays_sha256'] = sha(pn)
    if s1 is not None:
        p1 = out / (name + '_s1.npz'); saved = json.loads((out / (name + '_s1.json')).read_text())
        saved.update(arrays_sha256=sha(p1), dependencies_sha256=deps); dump(out / (name + '_s1.json'), saved)
    result['stages_saved'] = 1 if s1 is None else 2
    dump(out / (name + '.json'), result)
    print(json.dumps({k: result[k] for k in ['success', 'EENS_MWh', 'elapsed_seconds', 'stages_saved']}), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', required=True); args = ap.parse_args()
    jobs = json.loads((HERE / 'protocol/jobs_v37.json').read_text())['jobs']; job = next(j for j in jobs if j['name'] == args.job)
    lock = (HERE / 'logs' / (args.job + '.lock')).open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Case is already running')
    path = HERE / 'results' / (args.job + '.json')
    if path.exists(): raise FileExistsError('Existing receipt must be retained: ' + str(path))
    try: execute(job)
    except Exception:
        dump(path, dict(success=False, job=job, completed_utc=now(), error=traceback.format_exc())); raise


if __name__ == '__main__': main()
