"""Resource-proportional and shortage-proportional layouts for each V35 scenario.

The rules are the V32 B3 rules (rule_designs.capped_proportional) refitted on the
scenario's own three training baselines. Running this for the 'main' scenario
must reproduce the frozen V32 designs exactly; check() verifies that.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common35 import HERE, TRAIN, POWER_MW, SCENARIOS, now, sha, dump, arrays, load_input, baseline_files, design_file
from rule_designs import capped_proportional


def training_eens(scenario):
    total = None
    audit = []
    for w in TRAIN:
        pj, pn = baseline_files(scenario, w)
        br = json.loads(pj.read_text())
        assert br['success']
        if br.get('analytical_zero'):
            ue = np.zeros(len(br['zone_EENS_MWh']))
        else:
            ue = np.maximum(arrays(pn)['unserved'], 0.).sum(axis=0)
        total = ue if total is None else total + ue
        audit.append(dict(weather=w, EENS_MWh=float(br['EENS_MWh']), baseline_sha256=sha(pj)))
    return total, audit


def fit(scenario):
    d = load_input(scenario, TRAIN[0])
    q = np.asarray(d['q'], float)
    caps = float(d['eta_bio']) * (12. - float(d['beta'])) * q
    eens, audit = training_eens(scenario)
    out = {}
    for rule, weights in [('resource', q), ('eens', eens)]:
        power = capped_proportional(weights, caps, POWER_MW)
        out[rule] = dict(strategy='rule_' + rule, scenario=scenario, power=power.tolist(),
                         energy=(22. * q).tolist(), zones=d['zones'].tolist(), weights=weights.tolist(),
                         local_power_cap_MW=caps.tolist(), training_audit=audit,
                         created_utc=now(), holdout_used_for_fitting=False)
    return out


def make(scenario):
    for rule, v in fit(scenario).items():
        p = design_file(scenario, rule)
        if not p.exists():
            dump(p, v)


def check_main():
    """The refit on V32 B0 training baselines must equal the frozen B3 designs."""
    rows = {}
    for rule, v in fit('main').items():
        frozen = json.loads(design_file('main', rule).read_text())
        diff = float(np.abs(np.asarray(v['power']) - np.asarray(frozen['power'])).max())
        ediff = float(np.abs(np.asarray(v['energy']) - np.asarray(frozen['energy'])).max())
        rows[rule] = dict(max_power_difference_MW=diff, max_energy_difference_MWh=ediff,
                          passed=diff < 1e-9 and ediff < 1e-9)
    dump(HERE / 'qa/design_reproduction_main.json', rows)
    return rows


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-main']:
        print(json.dumps(check_main(), indent=1))
    else:
        for s in sys.argv[1:]:
            make(s)
