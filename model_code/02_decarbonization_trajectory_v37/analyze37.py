"""V37 numeric tables from completed receipts (no interpretation). Safe to run on partial results.

Outputs (report/): case_ledger.csv, year_use_summary.csv, use_accounting_by_year.csv, balance_composition.csv,
decomposition.csv, switching_inputs.csv, value_stack.csv, subset_check.csv, superposition.csv, sensitivities.csv,
criteria_inputs_v37.csv.
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common37 import HERE, V36, DASH, arrays, now, dump
from accounting37 import pair_metrics, finish_u1, FOSSIL, P36
sys.path.insert(0, str(V36 / 'scripts'))
import nonpower

YEARS = [2028, 2030, 2033, 2035]; GWP_BIO = {'GWP100': 27.0, 'GWP20': P36['GWP20']['bio_CH4']}; RHO = .072
GAS_CHAIN = {h: P36['gas_combustion_t_per_MWh'] + nonpower.gas_upstream(P36, h) for h in ['GWP100', 'GWP20']}
ETA_TH = .488; BOILER = .90; VOLL = 12240.; LEAK = [.010, .025, .042]
JOBS = {j['name']: j for j in json.loads((HERE / 'protocol/jobs_v37.json').read_text())['jobs']}


def ok(name):
    p = HERE / 'results' / (name + '.json')
    if not p.exists(): return None
    r = json.loads(p.read_text())
    return r if r.get('success') and r.get('physical_verified') and r.get('stages_saved') == 2 else None


def r0_path(job):
    y, w = job['year'], job['weather']; fx = bool(job.get('flex'))
    if job.get('sens') == 'no_gascut':
        n = f"{'F5' if fx else 'B5'}_gas_{y}_R0_WS{w:02d}"; return HERE / 'results' / (n + '.npz'), ok(n)
    if fx: n = f'F0_{y}_R0_WS{w:02d}'; return HERE / 'results' / (n + '.npz'), ok(n)        # amendment_post_02 main specification
    if y == 2035: return V36 / f'results/R0_WS{w:02d}.npz', json.loads((V36 / f'results/R0_WS{w:02d}.json').read_text())
    return HERE / 'results' / f'B0_{y}_R0_WS{w:02d}.npz', ok(f'B0_{y}_R0_WS{w:02d}')


def prices(year):
    p = pd.read_excel(DASH / 'Commodity Prices.xlsx', sheet_name='Commodity Prices'); p = p[p.Year.eq(year) & p.data_version.eq('ERAA 2025 final')]
    v = {str(r.Fuel).strip(): float(r.Value) * 3.6 for r in p.itertuples()}          # EUR/MWh fuel (CO2: EUR/t x 3.6, unused)
    lig = np.mean([x for k, x in v.items() if k.startswith('Lignite')])
    return {'gas': v['Gas'], 'Lignite': lig, 'Hard coal': v['Hard coal'], 'Light oil': v['Light oil'], 'Heavy oil': v['Heavy oil'], 'CO2_EUR_per_t': v['CO2 price'] / 3.6}


def case_ledger():
    rows = []; comp = []
    for name, job in JOBS.items():
        if job['use'] == 'R0': continue
        r = ok(name)
        if r is None: continue
        p0, r0 = r0_path(job)
        if r0 is None: continue
        u = arrays(HERE / 'results' / (name + '.npz')); z = arrays(p0)
        m = pair_metrics(u, z)
        meta = dict(name=name, batch=job['batch'], year=job['year'], weather=job['weather'], use=job['use'], resource=job.get('resource'),
                    country=job.get('country', 'ALL'), sens=job.get('sens', 'none'), scale=r.get('injection_scale', 1.), EENS_R0_MWh=r0['EENS_MWh'], EENS_MWh=r['EENS_MWh'])
        if job['use'] == 'U1': m.update(finish_u1(m))
        else:
            B = m['biomethane_MWh']; m['D_fossil_per_MWh'] = m['fossil_fuel_displaced_MWh'] / B
            for h in ['GWP100', 'GWP20']: m[f'avoided_{h}_t_per_MWh'] = m[f'avoided_{h}_t'] / B
            m.update(boost_MWh=r['boost_MWh'], compression_electricity_MWh=r['compression_electricity_MWh'], diverted_biomethane_MWh=float(u['other_gas_use'].sum()),
                     base_generation_MWh=float(np.sum(r['base_generation_MWh_by_zone'])))
        pr = prices(job['year']); m['fuel_saving_EUR'] = sum(m[f'delta_fuel_{k}_MWh'] * pr[k] for k in FOSSIL)
        rows.append({**meta, **m})
    return pd.DataFrame(rows)


def net_u(avoided, leak, h): return avoided - leak * RHO * GWP_BIO[h]


def main():
    led = case_ledger(); out = HERE / 'report'
    if led.empty: print('no completed pairs'); return
    led.to_csv(out / 'case_ledger.csv', index=False)
    sysU1 = led[(led.batch == 'B1')]; sysU2 = led[(led.batch == 'B2')]
    # Year-level system summary (36-weather means; equal B per weather, so means of ratios = ratio of sums).
    def summ(df):
        g = df.groupby('year'); s = g[['D_fossil_per_MWh', 'avoided_GWP100_t_per_MWh', 'avoided_GWP20_t_per_MWh', 'delta_EENS_MWh', 'fuel_saving_EUR', 'biomethane_MWh']].mean()
        s['weathers'] = g.size()
        if 's_f' in df: s['s_f'] = g.s_f.mean(); s['m_t_per_MWh_e'] = g.avoided_GWP100_t.sum() / g.fossil_generation_reduction_MWh.sum(); s['MEF_t_per_MWh_e'] = g.MEF_t_per_MWh_e.mean()
        return s
    S1 = summ(sysU1) if not sysU1.empty else pd.DataFrame(); S2 = summ(sysU2) if not sysU2.empty else pd.DataFrame()
    S1.add_prefix('U1_').join(S2.add_prefix('U2_'), how='outer').to_csv(out / 'year_use_summary.csv')
    # Use accounting by year: U1, U2 (dispatch only and with diverted gas credited), injection (U4), transport (U5).
    acc = []
    v36 = pd.read_csv(V36 / 'report/all_use_primary_numeric_ledger.csv')
    v36 = v36[v36.resource_scenario.eq('Central') & v36.cost_scenario.eq('central') & v36.horizon.eq('GWP100') & v36.variant.eq('primary') & v36.site_efficiency.eq(.4)]
    base_cost = {}
    for use in ['U1', 'U2']:
        x = v36[v36.use.eq(use) & v36.leakage.eq(.025)]
        if len(x): base_cost[use] = float((x.net_incremental_cost_excluding_carbon_EUR2024.sum() + x.avoided_fossil_fuel_cost_EUR2024.sum()) / x.biomethane_MWh.sum())
    for y in YEARS:
        if y not in S1.index: continue
        mef = float(S1.loc[y, 'MEF_t_per_MWh_e']); pr = prices(y)
        for h in ['GWP100', 'GWP20']:
            for leak in LEAK:
                for use in ['U1', 'U2']:
                    S = S1 if use == 'U1' else S2
                    if y not in S.index: continue
                    av = float(S.loc[y, f'avoided_{h}_t_per_MWh']); B = float(S.loc[y, 'biomethane_MWh'])
                    saving = float(S.loc[y, 'fuel_saving_EUR']) / B; net = net_u(av, leak, h); cost = base_cost.get(use, np.nan) - saving
                    acc.append(dict(year=y, use=use, horizon=h, leakage=leak, D=float(S.loc[y, 'D_fossil_per_MWh']), avoided=av, net_abatement=net,
                                    net_cost_EUR_per_MWh=cost, abatement_cost_EUR_per_t=cost / net if net > 0 else np.nan, carbon_price_EUR_per_t=pr['CO2_EUR_per_t'],
                                    breakeven_leakage=av / (RHO * GWP_BIO[h]), MEF=mef))
                    if use == 'U2':
                        d2 = sysU2[sysU2.year == y]; div = float(d2.diverted_biomethane_MWh.mean()) / B
                        av2 = av + div * (GAS_CHAIN[h] - .03 * mef)
                        acc.append(dict(year=y, use='U2_diverted_injected', horizon=h, leakage=leak, D=float(S.loc[y, 'D_fossil_per_MWh']) + div, avoided=av2,
                                        net_abatement=net_u(av2, leak, h), breakeven_leakage=av2 / (RHO * GWP_BIO[h]), MEF=mef, diverted_share=div))
                for use in ['U4', 'U5_CNG_diesel', 'U5_LNG_diesel', 'U5_CNG_fossil_CNG']:
                    c = nonpower.calculate(use, mef, 'central', leak, h, P36)
                    saving = c['gross_fossil_displacement_MWh_per_MWh'] * (pr['gas'] if c['displaced_fuel'] == 'gas' else pr['Light oil'])
                    cost = c['cost_before_fossil_saving_EUR_per_MWh'] - saving; net = c['net_abatement_t_per_MWh']
                    av = net + c['biogenic_methane_emissions_t_per_MWh']
                    acc.append(dict(year=y, use=use, horizon=h, leakage=leak, D=c['gross_fossil_displacement_MWh_per_MWh'], avoided=av, net_abatement=net,
                                    net_cost_EUR_per_MWh=cost, abatement_cost_EUR_per_t=cost / net if net > 0 else np.nan, carbon_price_EUR_per_t=pr['CO2_EUR_per_t'],
                                    breakeven_leakage=av / (RHO * GWP_BIO[h]), MEF=mef))
    A = pd.DataFrame(acc); A.to_csv(out / 'use_accounting_by_year.csv', index=False)
    # Energy-balance composition of U1 site electricity (system, 36-weather sums).
    bal = [c for c in sysU1.columns if c.startswith('balance_')]
    comp = sysU1.groupby('year')[bal].sum().div(sysU1.groupby('year').site_generation_MWh.sum(), axis=0)
    comp.columns = [c.replace('balance_', '').replace('_MWh', '') for c in comp.columns]; comp.to_csv(out / 'balance_composition.csv')
    # Two-factor Shapley decomposition of avoided = 0.40 * s_f * m between years.
    dec = []
    for a, b in [(2028, 2035), (2028, 2030), (2030, 2033), (2033, 2035)]:
        if a in S1.index and b in S1.index:
            sa, sb, ma, mb = S1.loc[a, 's_f'], S1.loc[b, 's_f'], S1.loc[a, 'm_t_per_MWh_e'], S1.loc[b, 'm_t_per_MWh_e']
            share = .4 * (sb - sa) * (ma + mb) / 2; inten = .4 * (mb - ma) * (sa + sb) / 2
            dec.append(dict(from_year=a, to_year=b, s_f_from=sa, s_f_to=sb, m_from=ma, m_to=mb, total_change=.4 * (sb * mb - sa * ma),
                            fossil_share_term=share, intensity_term=inten, fossil_share_fraction=share / (share + inten) if share + inten else np.nan))
    pd.DataFrame(dec).to_csv(out / 'decomposition.csv', index=False)
    # Switching inputs: system (B1) and single-country (B3) net abatement and heat threshold h*.
    heat = GAS_CHAIN['GWP100'] * ETA_TH / BOILER; sw = []
    inj = A[(A.use == 'U4') & (A.horizon == 'GWP100') & (A.leakage == .025)].set_index('year')
    def add(scope, df):
        for y, g in df.groupby('year'):
            if y not in inj.index: continue
            av = float(g.avoided_GWP100_t_per_MWh.mean()); ai = float(inj.loc[y, 'avoided'])
            sw.append(dict(scope=scope, year=y, weathers=len(g), D=float(g.D_fossil_per_MWh.mean()), avoided=av, s_f=float(g.s_f.mean()),
                           **{f'net_leak{l}': net_u(av, l, 'GWP100') for l in LEAK}, breakeven_leakage=av / (RHO * 27.), injection_avoided=ai,
                           h_star=(ai - av) / heat, **{'net_GWP20_leak0.025': net_u(float(g.avoided_GWP20_t_per_MWh.mean()), .025, 'GWP20')}))
    add('system_B1_36w', sysU1)
    sub = json.loads((HERE / 'protocol/weather_subsets.json').read_text())['subset8']
    add('system_B1_subset8', sysU1[sysU1.weather.isin(sub)])
    b3 = led[led.batch == 'B3']
    for c, g in b3.groupby('country'): add('country_' + c, g)
    pd.DataFrame(sw).to_csv(out / 'switching_inputs.csv', index=False)
    # Value stack: adequacy vs fuel saving for U1 and U2.
    vs = []
    for use, S in [('U1', S1), ('U2', S2)]:
        for y in S.index:
            vs.append(dict(use=use, year=y, delta_EENS_MWh=S.loc[y, 'delta_EENS_MWh'], adequacy_value_EUR=S.loc[y, 'delta_EENS_MWh'] * VOLL,
                           fuel_saving_EUR=S.loc[y, 'fuel_saving_EUR'], ratio=S.loc[y, 'delta_EENS_MWh'] * VOLL / S.loc[y, 'fuel_saving_EUR'] if S.loc[y, 'fuel_saving_EUR'] else np.nan))
    pd.DataFrame(vs).to_csv(out / 'value_stack.csv', index=False)
    # Subset check (registered) and superposition.
    chk = []
    for y, g in sysU1.groupby('year'):
        full = g.D_fossil_per_MWh.mean(); s8 = g[g.weather.isin(sub)].D_fossil_per_MWh.mean()
        chk.append(dict(year=y, weathers=len(g), D_36=full, D_subset8=s8, rel=s8 / full - 1, abs=s8 - full, passed=bool(len(g) == 36 and (abs(s8 / full - 1) <= .05 or abs(s8 - full) <= .01))))
    pd.DataFrame(chk).to_csv(out / 'subset_check.csv', index=False)
    sp = []
    for y, g in b3.groupby('year'):
        for w, gw in g.groupby('weather'):
            s1 = sysU1[(sysU1.year == y) & (sysU1.weather == w)]
            if len(s1) and len(gw) == 15:
                iso = (gw.fossil_fuel_displaced_MWh / gw.scale).sum(); sp.append(dict(year=y, weather=w, isolated_sum=iso, simultaneous=float(s1.fossil_fuel_displaced_MWh.iloc[0]), ratio=iso / float(s1.fossil_fuel_displaced_MWh.iloc[0])))
    pd.DataFrame(sp).to_csv(out / 'superposition.csv', index=False)
    led[led.batch.isin(['B4', 'B5', 'B6'])].to_csv(out / 'sensitivities.csv', index=False)
    # T1-T5 criterion inputs (no verdicts).
    crit = []
    if len(S1):
        crit.append(dict(item='T1_D_by_year', value=json.dumps({int(k): float(v) for k, v in S1.D_fossil_per_MWh.items()})))
        a, b = sysU1[sysU1.year == 2028].set_index('weather'), sysU1[sysU1.year == 2035].set_index('weather')
        common = a.index.intersection(b.index)
        crit.append(dict(item='T1_paired_positive_2028_minus_2035', value=json.dumps(dict(n=len(common), positive=int(((a.loc[common].D_fossil_per_MWh - b.loc[common].D_fossil_per_MWh) > 0).sum())))))
    for r in sw:
        if r['scope'] == 'system_B1_36w': crit.append(dict(item=f"T2_T5_{r['year']}", value=json.dumps(dict(net_leak0025=r['net_leak0.025'], breakeven=r['breakeven_leakage'], h_star=r['h_star']))))
    for r in dec: crit.append(dict(item=f"T3_{r['from_year']}_{r['to_year']}", value=json.dumps(dict(fossil_share_fraction=r['fossil_share_fraction']))))
    for r in vs:
        if r['use'] == 'U2': crit.append(dict(item=f"T4_{r['year']}", value=json.dumps(dict(ratio=r['ratio']))))
    pd.DataFrame(crit).to_csv(out / 'criteria_inputs_v37.csv', index=False)
    dump(out / 'analysis_run.json', dict(utc=now(), pairs=len(led), by_batch=led.batch.value_counts().to_dict()))
    print('pairs', len(led), led.batch.value_counts().to_dict())


if __name__ == '__main__': main()
