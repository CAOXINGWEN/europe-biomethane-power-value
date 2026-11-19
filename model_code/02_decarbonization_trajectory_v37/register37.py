"""Write protocol/jobs_v37.json from the registration (idempotent).

amendment_post_02: main specification = ERAA-complete system with dispatchable consumption (flex=True), batches
F0-F5 for all four target years (2035 counterfactual recomputed). The registered specification is retained as B0
(complete) and B1 (all years) for the registered-specification robustness check. Registered B2-B6 jobs that never
ran (no receipt of any kind) are withdrawn; jobs with receipts can never be removed or changed.
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common37 import HERE, now, sha, dump

COUNTRIES = ['AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'FR', 'DE', 'GR', 'IT', 'PT', 'RO', 'SI', 'ES', 'GB']
W36 = list(range(1, 37)); YEARS = [2028, 2030, 2033, 2035]


def jobs():
    sub = json.loads((HERE / 'protocol/weather_subsets.json').read_text()); s8, s12 = sub['subset8'], sub['subset12']
    J = [dict(name='REPRO_2035_R0_WS13', batch='REPRO', year=2035, weather=13, use='R0', resource='none')]
    # Registered specification (no dispatchable consumption): counterfactual and site power for all years.
    for y in [2028, 2030, 2033]:
        J += [dict(name=f'B0_{y}_R0_WS{w:02d}', batch='B0', year=y, weather=w, use='R0', resource='none') for w in W36]
    for y in YEARS:
        J += [dict(name=f'B1_{y}_U1_Central_WS{w:02d}', batch='B1', year=y, weather=w, use='U1', resource='Central') for w in W36]
    # Main specification (amendment_post_02): ERAA electrolysers and power-to-heat as price-bid consumption.
    F = dict(flex=True)
    for y in YEARS:
        J += [dict(name=f'F0_{y}_R0_WS{w:02d}', batch='F0', year=y, weather=w, use='R0', resource='none', **F) for w in W36]
    for y in YEARS:
        J += [dict(name=f'F1_{y}_U1_Central_WS{w:02d}', batch='F1', year=y, weather=w, use='U1', resource='Central', **F) for w in W36]
    for y in YEARS:
        J += [dict(name=f'F2_{y}_U2_Central_WS{w:02d}', batch='F2', year=y, weather=w, use='U2', resource='Central', beta=.75, **F) for w in W36]
    for y in YEARS:
        for c in COUNTRIES:
            J += [dict(name=f'F3_{y}_U1_{c}_WS{w:02d}', batch='F3', year=y, weather=w, use='U1', resource='Central', country=c, **F) for w in s8]
    J += [dict(name=f'F4_L1_2028_U1_V36Central_WS{w:02d}', batch='F4', year=2028, weather=w, use='U1', resource='V36Central', **F) for w in s12]
    J += [dict(name=f'F4_L2_2028_U1_High_WS{w:02d}', batch='F4', year=2028, weather=w, use='U1', resource='High', **F) for w in s12]
    for c in ['DE', 'SI']:
        J += [dict(name=f'F4_L3_2028_U1_{c}_x5_WS{w:02d}', batch='F4', year=2028, weather=w, use='U1', resource='Central', country=c, scale_mode='x5', **F) for w in s8[:4]]
    J += [dict(name=f'F5_nuc_2028_U1_Central_WS{w:02d}', batch='F5', year=2028, weather=w, use='U1', resource='Central', sens='nuclear_fixed', **F) for w in s12]
    for y in [2028, 2035]:
        J += [dict(name=f'F5_gas_{y}_R0_WS{w:02d}', batch='F5', year=y, weather=w, use='R0', resource='none', sens='no_gascut', **F) for w in s12]
        J += [dict(name=f'F5_gas_{y}_U1_Central_WS{w:02d}', batch='F5', year=y, weather=w, use='U1', resource='Central', sens='no_gascut', **F) for w in s12]
        J += [dict(name=f'F5_fleet_{y}_U1_WS{w:02d}', batch='F5', year=y, weather=w, use='U1', resource='fleet', **F) for w in s8]
    return J


def has_receipt(name):
    return any((HERE / 'results').glob(name + '.*')) or any((HERE / 'results').glob(name + '_s1.*'))


def main():
    p = HERE / 'protocol/jobs_v37.json'; new = jobs(); names = {j['name'] for j in new}; withdrawn = []
    if p.exists():
        old = json.loads(p.read_text())
        for j in old['jobs']:
            if j['name'] in names:
                if j != next(n for n in new if n['name'] == j['name']): raise ValueError('Registered job definition cannot change: ' + j['name'])
            elif has_receipt(j['name']): raise ValueError('Job with receipts cannot be removed: ' + j['name'])
            else: withdrawn.append(j['name'])
        withdrawn += old.get('withdrawn_never_run', [])
    by = {}
    for j in new: by[j['batch']] = by.get(j['batch'], 0) + 1
    dump(p, dict(created_utc=now(), registration_sha256=sha(HERE / 'protocol/registration_v37.json'), count=len(new), by_batch=by,
                 withdrawn_never_run=sorted(set(withdrawn)), jobs=new))
    print(len(new), by, 'withdrawn', len(set(withdrawn)))


if __name__ == '__main__': main()
