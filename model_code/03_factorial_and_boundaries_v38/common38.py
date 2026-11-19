"""V38 isolates all new output; V36/V37 imports and arrays are read-only."""
from pathlib import Path
import os, sys, json, hashlib
from datetime import datetime, timezone
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
V37 = ROOT / 'dataset_one/50_decarbonization_trajectory_v37'
V36 = ROOT / 'dataset_one/49_use_value_v36'
PY = '/home/cxw/anaconda3/envs/pv/bin/python'
WEATHERS = [17, 6, 9, 15, 36, 31, 33, 20]
sys.path.insert(0, str(V37 / 'scripts'))
sys.path.insert(0, str(V36 / 'scripts'))
import run_case37 as legacy
import common37
import prepare37
import numpy as np
import pandas as pd

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return common37.sha(p)
def dump(p, value): return common37.dump(p, value)
def arrays(p): return common37.arrays(p)
def relative(p): return str(Path(p).resolve().relative_to(ROOT))
def available(): return common37.available_gib()
def env():
    e = common37.child_env()
    e['V36_STAGE_METHODS'] = '1,2'
    return e
def registry(): return json.loads((HERE / 'protocol/registration_v38.json').read_text())
def jobs(): return json.loads((HERE / 'protocol/jobs_v38.json').read_text())['jobs']
def receipt_ok(name):
    p = HERE / 'results' / (name + '.json')
    if not p.exists(): return False
    r = json.loads(p.read_text())
    return r.get('success') and r.get('physical_verified') and r.get('stages_saved') == 2

def resource_country(mix='food', total='small'):
    r = pd.read_csv(V37 / 'inputs/resource_v37_country_year.csv')
    food = r[(r.year == 2035) & (r.scenario == 'Central')].set_index('country_code').deliverable_TWh
    fleet = r[(r.year == 2035) & (r.scenario == 'fleet')].set_index('country_code').fleet_fuel_TWh
    weights = food if mix == 'food' else fleet
    amount = food.sum() if total == 'small' else fleet.sum()
    return weights / weights.sum() * amount

def thermal_prices(d, year):
    """Reprice existing columns, including technologies retired in the donor year."""
    p = prepare37.prices(year)
    out = []
    for z, tech in zip(d['thermal_zone'], d['thermal_technology']):
        country = str(d['countries'][z])
        group = 1 if country in ('BG', 'CZ') else 3 if country in ('SI', 'RO') else 4 if country == 'GR' else 2
        fuel = next(v for k, v in p.items() if k.startswith('Lignite G' + str(group))) if tech == 'Lignite' else p.get(str(tech), 0.)
        out.append((fuel * 3.6 + prepare37.EF[tech] * p['CO2 price']) / prepare37.ETA[tech] + prepare37.VOM[tech])
    gas = np.full(len(d['zones']), (p['Gas'] * 3.6 + .20196 * p['CO2 price']) / .50 + 1.6)
    return np.asarray(out), gas

def build_input(job):
    year, w = job['year'], job['weather']
    d, _ = legacy.load_input(dict(year=year, weather=w, use='R0', resource='none'))
    source = d if job['vre_year'] == year else legacy.load_input(dict(year=job['vre_year'], weather=w, use='R0', resource='none'))[0]
    assert np.array_equal(d['zones'], source['zones'])
    d['vre'] = source['vre'].copy()
    d['thermal_cost'], d['gas_cost'] = thermal_prices(d, job['price_year'])
    by = resource_country(job.get('mix', 'food'), job.get('total', 'small'))
    q = np.array([by[c] * 1e6 / 8760 for c in d['countries']]) * legacy.weights(2035)
    measured = pd.read_csv(legacy.LUO).observed_biogas_m3_customer_h.to_numpy()
    x = measured[:240] if str(d['gas_split']) == 'train' else measured[240:480]
    phase = np.array([int(hashlib.sha256(str(z).encode()).hexdigest()[:8], 16) % len(x) for z in d['zones']])
    raw = x[(np.arange(8760)[:, None] + phase) % len(x)]
    hourly = raw / raw.mean(axis=0) * q[None, :]
    d['resource_q'], d['resource_q_hourly'] = q, hourly
    if job['use'] == 'U1': d['demand'] = d['demand'] - .4 * hourly
    # R0 retains potential-resource context only, with zero module production/base/boost.
    return d

def dependencies(job):
    paths = [V36/'scripts/model36.py', V36/'scripts/solver36.py', V37/'scripts/common37.py',
             V37/'scripts/run_case37.py', V37/'scripts/prepare37.py', V37/'scripts/flexload37.py',
             V37/'inputs/resource_v37_country_year.csv', V36/'inputs/resource_zone_weights.npz',
             legacy.LUO, legacy.RAW_FLEX, prepare37.DASH/'Commodity Prices.xlsx',
             common37.input_path(job['year'], job['weather']), common37.thermal_path(job['year']),
             common37.input_path(job['vre_year'], job['weather'])]
    return {relative(p): sha(p) for p in sorted(set(paths))}
