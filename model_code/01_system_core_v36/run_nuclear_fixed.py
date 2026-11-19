"""Sensitivity R1n: U1 (Central) with nuclear output fixed at the R0 counterfactual dispatch.

Answers whether the low fossil displacement of site power depends on the model letting nuclear
plants ramp down. Everything else is identical to the registered R1 Central case: the registered
run_case.execute() is reused unchanged; only the thermal_dispatch columns labelled 'Nuclear' get
lower = upper = R0 final-stage dispatch after the problem is built.
Usage: run_nuclear_fixed.py --weather N
"""
from pathlib import Path
import sys, json, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common36 import HERE, arrays

import model36
import run_case

_build = model36.build_problem


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--weather', type=int, required=True)
    w = ap.parse_args().weather
    r0 = arrays(HERE / 'results' / f'R0_WS{w:02d}.npz')
    nuclear = np.asarray(r0['thermal_technology']) == 'Nuclear'
    fixed = r0['thermal_dispatch'][:, nuclear]

    def build(*a, **kw):
        p = _build(*a, **kw)
        ids = p['builder'].blocks['thermal_dispatch'][:, nuclear].ravel()
        assert ids.size == fixed.size
        p['bounds'][ids, 0] = fixed.ravel()
        p['bounds'][ids, 1] = fixed.ravel()
        return p
    model36.build_problem = build
    job = dict(name=f'R1n_Central_WS{w:02d}', batch='R1', scenario='Central', weather=w,
               variant='nuclear_fixed_at_R0', scope='U1 Central with nuclear dispatch fixed at the R0 final-stage solution')
    out = HERE / 'results' / (job['name'] + '.json')
    if out.exists():
        raise FileExistsError(out)
    run_case.execute(job)


if __name__ == '__main__':
    main()
