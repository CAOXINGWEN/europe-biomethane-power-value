"""Freeze a post-review experiment without changing V37 hypotheses or receipts."""
import sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common38 import *

def main():
    for folder in ['protocol','inputs','results','logs','qa','report','literature/raw']:
        (HERE/folder).mkdir(parents=True, exist_ok=True)
    target = HERE/'protocol/registration_v38.json'
    if target.exists(): raise FileExistsError('Registration already frozen')
    js, cells, bindings = [], [], {}
    def bind(path):
        r = json.loads(path.with_suffix('.json').read_text())
        assert r.get('success') and r.get('physical_verified') and r.get('stages_saved') == 2
        assert sha(path) == r['arrays_sha256']
        for p in [path, path.with_suffix('.json')]: bindings[relative(p)] = sha(p)
        return dict(origin='V37', path=relative(path))
    def add(j):
        js.append(j)
        return dict(origin='V38', path=relative(HERE/'results'/(j['name']+'.npz')))
    for w in WEATHERS:
        for b,v,c in itertools.product([2028,2035],repeat=3):
            pair = dict(kind='mechanism', weather=w, background_year=b, vre_year=v, price_year=c)
            for use in ['R0','U1']:
                j=dict(name=f'M_B{b}_V{v}_C{c}_{use}_WS{w:02d}', batch='M', year=b, vre_year=v, price_year=c,
                       weather=w,use=use,mix='food',total='small',priority=2)
                if b==v==c==2035 or (b==v==c==2028 and use=='R0'):
                    pair[use]=bind(V37/'results'/f"F{'0' if use=='R0' else '1'}_{b}_{use}{'_Central' if use=='U1' else ''}_WS{w:02d}.npz")
                else: pair[use]=add(j)
            cells.append(pair)
        for mix,total in itertools.product(['food','fleet'],['small','large']):
            pair=dict(kind='scale',weather=w,mix=mix,total=total)
            pair['R0']=bind(V37/'results'/f'F0_2035_R0_WS{w:02d}.npz')
            if (mix,total)==('food','small'):
                pair['U1']=bind(V37/'results'/f'F1_2035_U1_Central_WS{w:02d}.npz')
            elif (mix,total)==('fleet','large'):
                pair['U1']=bind(V37/'results'/f'F5_fleet_2035_U1_WS{w:02d}.npz')
            else:
                pair['U1']=add(dict(name=f'S_{mix}_{total}_WS{w:02d}',batch='S',year=2035,vre_year=2035,price_year=2035,
                                   weather=w,use='U1',mix=mix,total=total,priority=1))
            cells.append(pair)
    pilot=dict(name='QA_reproduce_F1_2035_WS17',batch='QA',year=2035,vre_year=2035,price_year=2035,
               weather=17,use='U1',mix='food',total='small',priority=0)
    js.insert(0,pilot)
    static = {}
    for p in (HERE/'scripts').glob('*.py'): static[relative(p)]=sha(p)
    for j in js: static.update(dependencies(j))
    registration=dict(version='V38',registered_utc=now(),status='post_review_followup',
      seen_prior_results=True,user_authorization='2026-11-14 implement four-section Results, mechanism and scale controls, accounting boundaries and nearest literature comparison',
      source_plan='manuscript/plans/paper2_plan_v4.5_20261109/PAPER2_PLAN_AND_EXPERIMENTS_v4.5_zh.md',
      prior_T1_T5_unchanged=True,weather_selection='Same eight input-stratified V37 F3/F5 weather states; no new outcome-based selection',weathers=WEATHERS,
      mechanism=dict(design='Full 2x2x2 factorial: 2028/2035 for VRE array, thermal dispatch prices, all remaining system inputs',
        resource='2035 central country vector and 2035 within-country zone weights fixed for all cells',
        V='vre hourly array only, same weather state; does not change run-of-river or household PV/battery offsets embedded in demand',
        C='gas_cost and thermal_cost; underlying capacities and efficiencies fixed to background year',
        B='All remaining physical inputs and exogenous settings, including demand, network, storage, thermal capacities, hydro, H72 mask, hydrogen and flexible-load bids',
        limits='Conditional synthetic model interventions, not forecast scenarios. B is a combined residual group; no independent causal claims for its constituents.',
        outcome='R0-minus-U1 fossil fuel and avoided emissions per MWh; exact 3-factor Shapley plus interactions for each paired weather'),
      scale=dict(design='2035 2x2 total fuel x country shares',small_TWh=float(resource_country().sum()),large_TWh=float(resource_country(total='large').sum()),
        shares=['2035 central food waste','2023 observed biogas fleet'],within_country_weights='2035 fixed',
        outcome='Exact 2-factor Shapley allocation of D difference and interaction; no all-technology extrapolation'),
      accounting=dict(primary='Retain registered h=0, lambda=.025, GWP100; separately disclose expanded credits',
        heat='Net externally useful heat / catalog thermal output; distinguish zero-net threshold from matching injection',
        hydrogen='Incremental electrolyser output credit only to explicitly displaced fossil hydrogen; no credit for shift in other electrolysis',
        imports='Avoided external generation marginal factor scenarios, not measured marginal factors',
        correlated_boundaries='Joint scenario grid with separate process-electricity sensitivity. Primary compression factor retained; common expanded factor is an explicitly labelled symmetric-response approximation. Equal external heat and fuel services.',
        no_dynamic_reoptimization=True),
      numerical=dict(solver='COPT inherited V36',two_stages=True,physical_tolerance=2e-4,EENS_secondary_slack_MWh=1e-4,
        threads=4,stage_time_limit_seconds=7200,workers_max=4,launch_min_available_GiB=26,retries_max=3),
      pilot=dict(name=pilot['name'],reference='F1_2035_U1_Central_WS17',EENS_abs_tolerance_MWh=.001,
        objective_relative_tolerance=1e-8,objective_absolute_tolerance_EUR=1.,D_abs_tolerance=1e-5),
      source_sha256=static,reused_source_sha256=bindings,new_jobs=len(js),reused_arrays=len(bindings)//2,
      changes='Amendments must be append-only, with prior results exposure disclosed. No threshold or case exclusion changes to obtain a desired conclusion.')
    dump(HERE/'protocol/cells_v38.json',cells)
    dump(HERE/'protocol/jobs_v38.json',dict(jobs=js))
    registration['cells_sha256']=sha(HERE/'protocol/cells_v38.json')
    registration['jobs_sha256']=sha(HERE/'protocol/jobs_v38.json')
    dump(target,registration)
    print(json.dumps(dict(new_jobs=len(js),by_batch={b:sum(j['batch']==b for j in js) for b in ['QA','M','S']},cells=len(cells),reused_arrays=len(bindings)//2)))
if __name__=='__main__': main()
