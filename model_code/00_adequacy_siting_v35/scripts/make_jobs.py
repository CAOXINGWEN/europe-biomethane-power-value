"""Register the V35 case list before any new result is read."""
import json

from common35 import HERE, TRAIN, SCENARIOS, V32B, now, sha, dump

MAIN_POSITIVE_HOLDOUT = [2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 17, 18, 19, 21, 25, 26, 27, 32, 33, 36]
HOLDOUT = [w for w in range(1, 37) if w not in TRAIN]
A2 = V32B.parent / 'results'


def a2_eens(w):
    return json.loads((A2 / f'WS{w:02d}_full_mapped_cut0.3_ext0.75_baseline.json').read_text())['EENS_MWh']


def main_eens(w):
    return json.loads((V32B / 'results' / f'baseline_WS{w:02d}.json').read_text())['EENS_MWh']


def main():
    jobs = [dict(name='main_eval_rule_eens_WS13', scenario='main', kind='eval', rule='eens', weather=13,
                 purpose='pipeline reproduction of V32 B3_rule_eens_WS13')]
    for rule in ['eens', 'resource']:
        for w in MAIN_POSITIVE_HOLDOUT:
            jobs.append(dict(name=f'main_greedy_rule_{rule}_WS{w:02d}', scenario='main', kind='greedy',
                             rule=rule, weather=w))
    # Zero-shortfall weathers with prior evidence get analytic zero baselines, no LP.
    known_zero = dict(ext075=[w for w in range(1, 37) if a2_eens(w) <= 1e-3],
                      uk=[w for w in range(1, 37) if main_eens(w) <= 1e-3], cut05ext075=[])
    for sc in ['ext075', 'uk', 'cut05ext075']:
        for w in TRAIN + HOLDOUT:
            kind = 'zero_baseline' if w in known_zero[sc] else 'baseline'
            jobs.append(dict(name=f'{sc}_baseline_WS{w:02d}', scenario=sc, kind=kind, weather=w))
        for w in HOLDOUT:
            for rule in ['eens', 'resource']:
                jobs.append(dict(name=f'{sc}_eval_rule_{rule}_WS{w:02d}', scenario=sc, kind='eval',
                                 rule=rule, weather=w))
    reg = dict(
        registered_utc=now(),
        purpose='Review response V35: stress-boundary dependence (R1-M1), no-foresight dispatch (R1-M3), UK local resource (R1-M4).',
        scenarios=SCENARIOS,
        train_weather=TRAIN, holdout_weather=HOLDOUT, main_positive_holdout=MAIN_POSITIVE_HOLDOUT,
        power_budget_MW=25.96676631231336, buffer='22 q per zone for both rules',
        rules=dict(resource='power proportional to annual mean production q, local cap eta(12-beta)q',
                   eens='power proportional to summed zone EENS of the scenario\'s three training baselines, same cap'),
        greedy_rule='scripts/greedy.py: full boost in every paired-baseline shortfall hour from stock above a protective floor; no stock reserved for later shortfall',
        metric='ratio of summed paired net reductions (rule_eens / rule_resource) over positive holdout weathers; zero-baseline weathers are analytic zero service',
        zero_baseline_rule='ext075: V32 A2 baseline of the same electrical LP has EENS <= 0.001 MWh; uk: main V32 B0 baseline EENS <= 0.001 MWh, UK base supply only adds generation, and UK base-gas feasibility with 16q is checked by the protective-floor recursion.',
        reporting='All registered scenarios are reported whether or not cut05ext075 meets the hour-ratio calibration criterion [0.5, 2] and Spearman >= 0.6.',
        code_sha256={p.name: sha(p) for p in sorted((HERE / 'scripts').glob('*.py'))},
        v32_model_sha256=sha(V32B / 'scripts/model.py'),
        jobs=jobs)
    dump(HERE / 'protocol/jobs_v35.json', reg)
    print(len(jobs), 'jobs registered')


if __name__ == '__main__':
    main()
