"""Shared definitions for the V35 review-response experiments.

V32 stage-B inputs, baselines, designs and model code are read-only. Every V35
case writes a new JSON receipt and a compact NPZ in ../results.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
V32B = ROOT / 'dataset_one/44_followup_corrected_system_v32/stage_b'
V27 = ROOT / 'dataset_one/39_public_validation_system_comparison_v27'
PY = '/home/cxw/anaconda3/envs/pv/bin/python'
T = 8760
TRAIN = [23, 11, 9]
POWER_MW = 25.96676631231336
ZERO_FLOOR_MWh = 1e-3
HOUR_THRESHOLD_MW = 1e-3

# Stress boundaries. 'main' is the frozen V32 boundary; the others are the
# review sensitivities registered in protocol/registration_v35.json.
SCENARIOS = {
    'main': dict(cut=0.3, ext=0.5, uk=False),
    'ext075': dict(cut=0.3, ext=0.75, uk=False),
    'cut05ext075': dict(cut=0.5, ext=0.75, uk=False),
    'uk': dict(cut=0.3, ext=0.5, uk=True),
}

sys.path.insert(0, str(V32B / 'scripts'))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def role(weather):
    return 'train' if weather in TRAIN else 'holdout'


def input_path(weather):
    return V32B / 'inputs' / f'2035_WS{weather:02d}.npz'


_wave = {}


def gas_wave(split):
    """Observed station waveform, identical to the V32 A-stage gas_shape()."""
    if split not in _wave:
        f = pd.read_csv(V27 / 'station/processed/luo_measured_production_hourly.csv')
        x = f.observed_biogas_m3_customer_h.to_numpy()
        _wave[split] = x[slice(0, 240) if split == 'train' else slice(240, 480)]
    return _wave[split]


def gas_shape(zones, q, split):
    x = gas_wave(split)
    phase = np.array([int(hashlib.sha256(str(z).encode()).hexdigest()[:8], 16) % len(x) for z in zones])
    idx = (np.arange(T)[:, None] + phase) % len(x)
    raw = x[idx]
    return raw / raw.mean(axis=0) * q[None, :]


UK_FILE = HERE / 'protocol/uk_resource_v35.json'


def uk_resource_vector():
    """Fixed UK resource: mean resource-to-gross-demand intensity of the 14 resource
    countries, both terms averaged over all 36 weather inputs, applied to the UK zones."""
    if UK_FILE.exists():
        v = json.loads(UK_FILE.read_text())
        return np.asarray(v['q_MW'], float), float(v['intensity_MW_fuel_per_MW_demand'])
    gross, ref = None, None
    for w in range(1, 37):
        d = arrays(input_path(w))
        g = d['gross_demand'].mean(axis=0)
        gross = g if gross is None else gross + g
        if ref is None:
            ref = d
        assert np.array_equal(d['q'], ref['q'])
    gross /= 36.
    q = np.asarray(ref['q'], float)
    uk = np.isin(ref['zones'], ['UK00', 'UKNI'])
    intensity = q[~uk].sum() / gross[~uk].sum()
    out = q.copy()
    out[uk] = intensity * gross[uk]
    dump(UK_FILE, dict(zones=ref['zones'].tolist(), q_MW=out.tolist(), mean_gross_demand_MW=gross.tolist(),
                       intensity_MW_fuel_per_MW_demand=float(intensity),
                       uk_q_MW=out[uk].tolist(), uk_annual_TWh=float(out[uk].sum() * T / 1e6),
                       total_annual_TWh=float(out.sum() * T / 1e6), created_utc=now(),
                       definition='UK00 and UKNI receive the mean resource intensity of the 14 resource countries; 36-weather mean gross demand'))
    return out, float(intensity)


def load_input(scenario, weather):
    d = arrays(input_path(weather))
    if SCENARIOS[scenario]['uk']:
        q, intensity = uk_resource_vector()
        uk = np.isin(d['zones'], ['UK00', 'UKNI'])
        eta, beta = float(d['eta_bio']), float(d['beta'])
        shape = gas_shape(d['zones'], q, str(d['gas_split']))
        d['q'] = q
        d['base'] = np.where(uk, beta * eta * q, d['base'])
        d['q_hourly'] = np.where(uk[None, :], shape, d['q_hourly'])
        d['uk_resource_intensity_MW_per_MW'] = np.array(intensity)
    return d


def baseline_files(scenario, weather):
    if scenario == 'main':
        p = V32B / 'results' / f'baseline_WS{weather:02d}.json'
    else:
        p = HERE / 'results' / f'{scenario}_baseline_WS{weather:02d}.json'
    return p, p.with_suffix('.npz')


def design_file(scenario, rule):
    if scenario == 'main':
        return V32B / 'results' / f'design_rule_{rule}.json'
    return HERE / 'results' / f'design_{scenario}_rule_{rule}.json'


def available_GiB():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    return 0.


def child_env():
    env = os.environ.copy()
    for k in ['V28_LP_METHOD', 'V28_SOLVER_THREADS', 'V28_SOLVER_MEMORY_MB', 'V28_SOLVER_LOG', 'V28_WARM_SECONDARY']:
        env.pop(k, None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', V32_SOLVER_MEMORY_MB='12288', PYTHONUNBUFFERED='1')
    return env
