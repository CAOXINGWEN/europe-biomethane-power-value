"""Standalone numerical verification of headline trajectories, Shapley/Möbius attributions, facility benchmarks, and parity boundaries."""
import itertools
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SD = REPO / 'source_data'


def assert_close(name, a, b, tol=1e-9):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert np.isfinite(err) and err <= tol, f'{name} failed: max_abs_err={err} > {tol}'
    print(f'[PASS] {name} (max_abs_err={err:.2e})')


# 1. Open-source 168-h HiGHS LP dispatch replay
res = subprocess.run([sys.executable, str(REPO / 'example/replay.py')], capture_output=True, text=True)
assert res.returncode == 0, res.stderr
print('[PASS] Open-source SciPy HiGHS 168-hour paired LP benchmark (Pinerolo ITN1 & pan-European injection)')

# 2. Headline 36-weather trajectory (2028-2035)
d = pd.read_csv(SD / 'boundary_base_by_weather.csv')
assert len(d) == 144 and not d.duplicated(['year', 'weather']).any()
assert (np.diff(d.pivot(index='weather', columns='year', values='D').to_numpy(), axis=1) < 0).all()
assert_close('Headline 36-weather fossil displacement D (2028, 2030, 2033, 2035)',
             d.groupby('year').D.mean().to_numpy(), [0.530394816, 0.325834843, 0.243828309, 0.183142713], tol=5e-9)
assert int((d[d.year == 2035].net_primary < 0).sum()) == 30
print('[PASS] All 36 weather years decline monotonically; 30/36 below zero in 2035 at 2.5% methane intensity')

# 3. Exact Shapley & Möbius factorial decompositions (mechanism 2^3 and scale 2^2)
stored = pd.read_csv(SD / 'factor_shapley_by_weather.csv')
inter = pd.read_csv(SD / 'factor_interactions_by_weather.csv')
for kind, names, levels in [
    ('mechanism', ['background_year', 'vre_year', 'price_year'], [(2028, 2035)] * 3),
    ('scale', ['mix', 'total'], [('food', 'fleet'), ('small', 'large')]),
]:
    data = pd.read_csv(SD / ('mechanism_by_weather.csv' if kind == 'mechanism' else 'scale_by_weather.csv'))
    for weather, g in data.groupby('weather'):
        for metric in ['D', 'avoided']:
            cells = {tuple(int(getattr(r, n) == levels[i][1]) for i, n in enumerate(names)): getattr(r, metric) for r in g.itertuples()}
            effects = np.zeros(len(names))
            perms = list(itertools.permutations(range(len(names))))
            for perm in perms:
                state = [0] * len(names)
                for i in perm:
                    before = cells[tuple(state)]
                    state[i] = 1
                    effects[i] += (cells[tuple(state)] - before) / len(perms)
            target = stored[(stored.kind == kind) & (stored.weather == weather) & (stored.metric == metric)].set_index('factor')
            assert np.max(np.abs(effects - target.loc[names, 'effect'].to_numpy())) <= 1e-10
            its = inter[(inter.kind == kind) & (inter.weather == weather) & (inter.metric == metric)]
            for row in its.itertuples():
                idxs = [names.index(n) for n in row.factors.split('+')]
                val = sum((-1) ** (len(idxs) - sum(bits)) * cells[tuple(bits[idxs.index(k)] if k in idxs else 0 for k in range(len(names)))]
                          for bits in itertools.product([0, 1], repeat=len(idxs)))
                assert abs(val - row.effect) <= 1e-10
print('[PASS] Exact Shapley main effects & Möbius interactions verified across all weather years (mechanism & scale)')

# 4. Operational biowaste facilities (Berlin Ruhleben DE00 & northern Italian Pinerolo ITN1) & parity bounds
sy = pd.read_csv(SD / 'site52/site_year.csv')
assert_close('Pinerolo annual D_raw (2028, 2035)',
             [round(sy[(sy.site == 'Pinerolo') & (sy.year == 2028)].D.mean(), 3),
              round(sy[(sy.site == 'Pinerolo') & (sy.year == 2035)].D.mean(), 3)], [0.660, 0.464], tol=1e-6)
assert_close('Berlin Ruhleben annual D_raw (2028, 2035)',
             [round(sy[(sy.site == 'Ruhleben') & (sy.year == 2028)].D.mean(), 3),
              round(sy[(sy.site == 'Ruhleben') & (sy.year == 2035)].D.mean(), 3)], [0.557, 0.095], tol=1e-6)
ruh35 = pd.read_csv(SD / 'site52/ruhleben_thresholds.csv').query('year == 2035')
assert len(ruh35) == 768 and int((ruh35.grid_minus_chp_at_q1 > 0).sum()) == 738
ts35 = pd.read_csv(SD / 'site52/threshold_summary.csv').query('year == 2035')
assert len(ts35) == 270 and int((ts35.grid_minus_site_q1 > 0).sum()) == 252
at36 = pd.read_csv(SD / 'site52/access_and_timing.csv')
assert len(at36) == 36 and bool((at36.increment_over_keep_Mt > 0.40).all())
print('[PASS] Operational facility benchmarks (Pinerolo & Berlin Ruhleben), 270-cell parity grid (252/270 > 0), and 36 rollout scenarios verified')
print('\nAll numerical and LP replay checks PASSED.')
