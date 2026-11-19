"""Target-year food-waste biomethane resource by country (registration_v37 resource_rules) and existing-biogas fleet.

R(c,y,s) = P(c,y) * w0(c) * r(y,s) * k(c,y,s) * 0.97 * a * 90 Nm3/t * 0.85 * 0.995 * 0.98 * 0.96 * 9.9722 kWh/Nm3
Outputs: inputs/resource_v37_country_year.csv (scenarios Low/Central/High and fleet), report/resource_lmdi.csv,
qa/resource_reproduction.json, inputs/resource_fleet_ready.json.
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common37 import HERE, ROOT, DS, V36, now, sha, dump

RAW = HERE / 'inputs/raw'
EU14 = ['AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'FR', 'DE', 'GR', 'IT', 'PT', 'RO', 'SI', 'ES']
ESTAT = {c: ('EL' if c == 'GR' else c) for c in EU14}
YEARS = [2028, 2030, 2033, 2035]; BASE_YEAR = 2022
CHAIN = 0.97 * 90 * 0.85 * 0.995 * 0.98 * 0.96 * 9.9722e-9      # TWh per tonne of AD feedstock before the AD share
AD_SHARE = 0.80
# BIC/ZWE (Nov 2024) Table 10, food waste collected / potential generation, 2022 (pdf pp. 16-17).
K0_TABLE = {'AT': .29, 'BE': .29, 'BG': .00, 'HR': .06, 'CZ': .16, 'DK': .49, 'FR': .17, 'DE': .32, 'GR': .01, 'IT': .72,
            'PT': .14, 'RO': .02, 'SI': .14, 'ES': .15, 'GB_BICZWE': .20}
K0_EU27_TABLE = {'CY': .03, 'EE': .03, 'FI': .15, 'HU': .07, 'IE': .11, 'LV': .07, 'LT': .17, 'LU': .34, 'MT': .38, 'NL': .28, 'PL': .09, 'SK': .19, 'SE': .16}
K_UK = 0.170                                                     # registered: WRAP observed household capture
R = {'Central': {2028: .775, 2030: .70, 2033: .70, 2035: .70}, 'High': {2028: .8875, 2030: .85, 2033: .85, 2035: .85},
     'Low': {2028: .775, 2030: .70, 2033: .58, 2035: .50}}


def jsonstat(path, dims):
    j = json.loads(Path(path).read_text()); ids = j['id']; size = j['size']; idx = {d: j['dimension'][d]['category']['index'] for d in ids}
    out = {}
    for k, v in j['value'].items():
        k = int(k); pos = []
        for s in reversed(size): pos.append(k % s); k //= s
        pos = pos[::-1]; key = {}
        for d, p in zip(ids, pos):
            inv = {i: c for c, i in idx[d].items()}; key[d] = inv[p]
        out[tuple(key[d] for d in dims)] = v
    return out


def main():
    pop = jsonstat(RAW / 'eurostat/demo_pjan_2021_2023.json', ['geo', 'time'])
    proj = jsonstat(RAW / 'eurostat/proj_23np_bsl.json', ['geo', 'time'])
    sch = pd.read_csv(DS / '02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv')
    sch = sch[sch.scenario.eq('Central')]
    uk = pd.read_csv(V36 / 'inputs/uk_resource_v36.csv'); uk_base = float(uk[uk['case'].eq('primary_mean_2021_2022')].base_food_waste_t.iloc[0])
    ukpop = json.loads((RAW / 'ons/ukpop_timeseries.json').read_text()); ukmid = {int(y['year']): float(y['value']) for y in ukpop['years']}
    ons = pd.read_excel(RAW / 'ons/ukpppsummary_2024based.xlsx', 'PERSONS', header=None)
    hdr = [str(v).split(' ')[0] for v in ons.iloc[6].tolist()]; end = ons[ons.iloc[:, 0].astype(str).str.startswith('Population at end')].iloc[0]
    ukproj_mid = {int(h): float(end.iloc[i]) * 1000 for i, h in enumerate(hdr) if h.isdigit() and str(end.iloc[i]) != '[z]'}      # mid-year y = end of column y
    ukjan = {y: (ukproj_mid[y - 1] + ukproj_mid[y]) / 2 for y in YEARS}

    kstar = max(list({c: K0_TABLE[c] for c in EU14}.values()) + list(K0_EU27_TABLE.values())); assert kstar <= .85
    rows = []; base = {}
    for c in EU14 + ['GB']:
        if c == 'GB':
            w0 = uk_base / ((ukmid[2021] + ukmid[2022]) / 2); P = {**{BASE_YEAR: ukmid[BASE_YEAR]}, **ukjan}; k0 = K_UK; psrc = 'ONS mid-year estimates; 1 Jan of target year = mean of adjacent mid-year values of the ONS 2024-based principal projection'
        else:
            e = ESTAT[c]; t = sch[sch.country_code.eq(c) & sch.year.between(2021, 2023)].base_food_waste_t.mean()
            p = np.mean([pop[(e, str(y))] for y in (2021, 2022, 2023)]); w0 = t / p
            P = {BASE_YEAR: pop[(e, str(BASE_YEAR))], **{y: proj[(e, str(y))] for y in YEARS}}; k0 = K0_TABLE[c]; psrc = 'Eurostat demo_pjan / proj_23np BSL, 1 January'
        base[c] = dict(P=P[BASE_YEAR], w0=w0, r=1., k=k0, R=P[BASE_YEAR] * w0 * k0 * AD_SHARE * CHAIN)
        k2035 = {'Low': k0, 'Central': k0 + .5 * max(kstar - k0, 0), 'High': max(kstar, k0)}
        for s in ['Low', 'Central', 'High']:
            for y in YEARS:
                k = k0 + (k2035[s] - k0) * (y - BASE_YEAR) / (2035 - BASE_YEAR)
                twh = P[y] * w0 * R[s][y] * k * AD_SHARE * CHAIN
                rows.append(dict(country_code=c, year=y, scenario=s, population=P[y], w0_t_per_capita=w0, r=R[s][y], k=k, k0=k0, k_star=kstar,
                                 ad_share=AD_SHARE, deliverable_TWh=twh, MW_fuel_mean=twh * 1e6 / 8760, population_source=psrc,
                                 k_source='BIC/ZWE 2024 Table 10 (2022)' if c != 'GB' else 'WRAP observed household capture (registered); BIC/ZWE gives 0.20'))
    df = pd.DataFrame(rows)

    # Existing biogas fleet (scale check): EU = Eurostat gross electricity from biogases 2023 / 0.40; UK = DUKES 6.4 biogas fuel used for electricity 2023.
    gep = jsonstat(RAW / 'eurostat/nrg_bal_peh_biogas.json', ['geo', 'time'])
    d64 = pd.read_excel(RAW / 'dukes/DUKES_6.4_2026.xlsx', '6.4', header=None)
    yrs = [str(v).split('.')[0] for v in d64.iloc[4].tolist()]; brow = d64[d64.iloc[:, 0].astype(str).str.strip().eq('Biogas')].iloc[0]
    uk_fuel_ktoe = float(brow.iloc[yrs.index('2023')])
    fleet = []
    for c in EU14 + ['GB']:
        fuel = uk_fuel_ktoe * 11.63 / 1000 if c == 'GB' else gep[(ESTAT[c], '2023')] / .40 / 1000
        for y in [2028, 2035]:
            fleet.append(dict(country_code=c, year=y, scenario='fleet', fleet_fuel_TWh=fuel, deliverable_TWh=np.nan,
                              population_source='', k_source='Eurostat nrg_bal_peh R5300 GEP 2023 / 0.40' if c != 'GB' else 'DUKES 6.4 biogas used to generate electricity 2023, 11.63 GWh/ktoe'))
    df = pd.concat([df, pd.DataFrame(fleet)], ignore_index=True)
    out = HERE / 'inputs/resource_v37_country_year.csv'
    if out.exists(): raise FileExistsError(out)
    df.to_csv(out, index=False)

    # Reproduction: 2022 population, r=0.70, k=0.249 for EU14 vs V36 Central 2.9416 TWh.
    rep = sum(pop[(ESTAT[c], '2022')] * base[c]['w0'] * .70 * .249 * AD_SHARE * CHAIN for c in EU14)
    v36 = 2.9416180841297317; rel = rep / v36 - 1
    dump(HERE / 'qa/resource_reproduction.json', dict(utc=now(), EU14_TWh=rep, V36_Central_TWh=v36, relative_difference=rel, passed=abs(rel) <= .03))

    # LMDI (additive) from the 2022 base (r = 1) to each target year: population, prevention, capture.
    lm = []
    for c in EU14 + ['GB']:
        b = base[c]
        for s in ['Low', 'Central', 'High']:
            for y in YEARS:
                r = df[(df.country_code == c) & (df.year == y) & (df.scenario == s)].iloc[0]
                Ry, Rb = r.deliverable_TWh, b['R']
                L = (Ry - Rb) / np.log(Ry / Rb) if Ry > 0 and Rb > 0 and abs(Ry - Rb) > 1e-15 else Ry
                eff = dict(population=np.log(r.population / b['P']), prevention=np.log(r.r / b['r']), capture=np.log(r.k / b['k'])) if Rb > 0 and Ry > 0 else None
                lm.append(dict(country_code=c, scenario=s, year=y, R_base_2022_TWh=Rb, R_year_TWh=Ry, delta_TWh=Ry - Rb,
                               **({f'{k}_TWh': L * v for k, v in eff.items()} if eff else dict(population_TWh=np.nan, prevention_TWh=np.nan, capture_TWh=Ry - Rb))))
    pd.DataFrame(lm).to_csv(HERE / 'report/resource_lmdi.csv', index=False)
    dump(HERE / 'inputs/resource_fleet_ready.json', dict(utc=now(), file=str(out.relative_to(HERE)), sha256=sha(out)))
    tot = df[df.scenario != 'fleet'].groupby(['scenario', 'year']).deliverable_TWh.sum().unstack()
    print(tot.round(3).to_string()); print('fleet fuel TWh', round(df[(df.scenario == 'fleet') & (df.year == 2028)].fleet_fuel_TWh.sum(), 2))
    print('reproduction', round(rep, 4), 'vs', round(v36, 4), f'{rel:+.2%}', 'k*', kstar)


if __name__ == '__main__': main()
