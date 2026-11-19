"""Freeze R0-R3 jobs and adaptive R4/R5 rules before any new dispatch results."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common36 import HERE,ROOT,now,sha,dump,SCENARIOS,TRAIN


def main():
    path=HERE/'protocol/jobs_v36.json'
    if path.exists():raise FileExistsError('Registered jobs exist')
    for name in ['model_extension_tests.json','thermal_split_reconciliation.json','reproduction_extended_main_eval_rule_eens_WS13.json']:
        r=json.loads((HERE/'qa'/name).read_text());assert r['passed'],name
    r=json.loads((HERE/'qa/reproduction_extended_main_eval_rule_eens_WS13.json').read_text())
    assert r['model36_sha256']==sha(HERE/'scripts/model36.py')
    regpath=HERE/'protocol/registration_v36.json';reg=json.loads(regpath.read_text())
    reg['amendments'].append(dict(id='amendment_03',registered_utc=now(),results_read_before_change='No V36 national dispatch results; only historical reproduction and synthetic unit tests.',change='Freeze numerical implementation before R0: gas .50 fallback (existing fine-unit mix absent), ERAA Common Data standard efficiencies for other thermal categories, exact V32 aggregate capacity reconciliation excluding hydrogen and including fixed gas and original biomass-pool subtraction. Central operating-cost merit order uses ERAA 2025 official 2035 commodity/carbon prices in EUR2024, plus IPCC combustion factors; Biofuel retains explicit 25 EUR/MWh marginal-cost assumption. U1–U3 all jointly reoptimize DSR/iDSR under identical two-stage objective. R5 and R4 minimize EENS only, matching batch counts. ELCC uses a common firm-capacity size across all positive R0 weathers, matching 36-weather mean reduction; U1 shares are .4q, U2 shares are .75*.4q+installed boost power. National dispatch differences are location-based, not causal resource-origin attribution. Preserve all raw outputs and separate U3 direct fuel substitution from changed system dispatch.'))
    dump(regpath,reg)
    order=TRAIN+[13]+[w for w in range(1,37) if w not in TRAIN+[13]]
    jobs=[dict(name=f'R0_WS{w:02d}',batch='R0',weather=w,scenario='Central') for w in order]
    for batch in ['R1','R2','R3']:
        for scenario in ['Central','Low','High']:
            jobs.extend(dict(name=f'{batch}_{scenario}_WS{w:02d}',batch=batch,weather=w,scenario=scenario) for w in order)
    value=dict(registered_utc=now(),registration_sha256=sha(regpath),jobs=jobs,adaptive_rules=dict(R4=dict(type='global ELCC bisection, identical capacity across positive R0 weathers',uses=['R1','R2'],max_iterations=12,initial_upper='twice .4q total for U1, twice (base+boost) for U2; expand if needed',tolerance_MWh=1.,relative_tolerance=.005,weather_average_divisor=36),R5=dict(beta=[.75,0.],power_GW=[.26,1.,4.,16.],buffer_h=[22,168,720],sample_sorted_positive_R0_ranks=[2,5,8,11,14,17,20,23],stratification='contiguous groups of three positive weathers, representatives ranks 2,5,...,20,23; last group size 2; zero-risk weathers contribute zero',scenario='Central')),code_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'scripts').glob('*.py')) if p.name in ['model36.py','solver36.py','common36.py','run_case.py','prepare36.py','register_jobs.py']},amendments=[])
    dump(path,value);print(json.dumps(dict(registered_physical_jobs=len(jobs),registered_LP_stages=720,adaptive_batches=['R4','R5'])))


if __name__=='__main__':main()
