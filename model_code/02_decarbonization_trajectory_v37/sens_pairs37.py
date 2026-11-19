"""Pair metrics for the F4 checks and F5 sensitivities (main specification): U case minus its counterfactual.

F4_L1/L2/L3, F5_nuc and F5_fleet pair with F0 (same year and weather); F5_gas U1 pairs with F5_gas R0 (no gas cut).
Output: report/final_F4F5_by_case.csv (one row per U case; balance shares are fractions of the added site electricity).
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from multiprocessing import Pool
from common37 import HERE, arrays
from accounting37 import pair_metrics, finish_u1
from analyze37 import JOBS, ok, r0_path


def one(name):
    job = JOBS[name]; r = ok(name); p0, r0 = r0_path(job)
    if r is None or r0 is None: return dict(name=name, missing=True)
    u = arrays(HERE / 'results' / (name + '.npz')); z = arrays(p0)
    m = pair_metrics(u, z); x = finish_u1(m); site = x['site_generation_MWh']
    sh = {k.replace('balance_', 'share_').replace('_MWh', ''): v / site for k, v in m.items() if k.startswith('balance_')}
    return dict(name=name, batch=job['batch'], year=job['year'], weather=job['weather'], resource=job.get('resource'), country=job.get('country', 'ALL'),
                sens=job.get('sens', 'none'), scale=r.get('injection_scale', 1.), r0=Path(p0).stem, B=m['biomethane_MWh'], site_MWh=site,
                fossil_MWh=m['fossil_fuel_displaced_MWh'], D=x['D_fossil_per_MWh'], avoided=x['avoided_GWP100_t_per_MWh'], avoided20=x['avoided_GWP20_t_per_MWh'],
                s_f=x['s_f'], m=x['m_t_per_MWh_e'], resid=x['balance_residual_relative'], dEENS=m['delta_EENS_MWh'], EENS_R0=r0['EENS_MWh'],
                extra_h2_MWh=m.get('extra_hydrogen_MWh', np.nan), **{f'dfuel_{k}': v for k, v in m.items() if k.startswith('delta_fuel_')}, **sh)


def main():
    names = [n for n, j in JOBS.items() if j['batch'] in ('F4', 'F5') and j['use'] != 'R0']
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as p: rows = p.map(one, names, chunksize=1)
    T = pd.DataFrame(rows); T.to_csv(HERE / 'report/final_F4F5_by_case.csv', index=False)
    print(len(T), 'missing', int(T.get('missing', pd.Series(dtype=bool)).fillna(False).sum()))
    print(T.groupby(['batch', 'year', 'sens', 'resource', 'country']).agg(n=('D', 'size'), D=('D', 'mean'), resid=('resid', lambda v: float(np.abs(v).max()))).round(4))


if __name__ == '__main__': main()
