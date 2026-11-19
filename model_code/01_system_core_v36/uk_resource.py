"""UK food-waste biomethane resource on the same parameter chain as the EU14 resource table.

Boundary (matches the EU14 base: households + restaurants and food services, Eurostat env_wasfw):
  household food waste collected by local authorities (WRAP, four streams; excludes sewer and home
  compost, as sewer discards are a voluntary item outside the EU mandatory total) + hospitality and
  food service (HaFS, WRAP model estimate, 2021 only).
The EU14 chain parameters are identical across countries, so the UK value is base mass x the EU14
TWh-per-Mt factor of each scenario. Inputs are read from the WRAP evidence package (read-only).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DS = HERE.parent
UKP = DS / '48_uk_food_waste_sources_20261103/processed'
EU = DS / '02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv'
V35_PROXY = DS / '47_review_response_v35/protocol/uk_resource_v35.json'
HOURS = 8760


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    nb = pd.read_csv(UKP / 'national_benchmarks.csv').set_index('id')
    nat = pd.read_csv(UKP / 'nations_2022_kerbside.csv').set_index('geo')
    cap = pd.read_csv(UKP / 'capture_2022.csv').iloc[0]
    eu = pd.read_csv(EU)
    eu = eu[eu.base_food_waste_t > 0]
    factor = (eu.deliverable_fuel_TWh_LHV / (eu.base_food_waste_t / 1e6)).groupby(eu.scenario).agg(['min', 'max'])
    assert (factor['max'] - factor['min']).abs().max() < 1e-9, 'EU14 chain parameters must be uniform across countries'
    f = factor['min'].to_dict()                       # TWh fuel (LHV) per Mt food waste, deliverable stage
    # Upgraded-CH4 stage (before plant methane losses and availability): the stage behind the V32-V35 model q.
    up = (eu.upgraded_CH4_Nm3 * 9.9722e-9 / (eu.base_food_waste_t / 1e6)).groupby(eu.scenario).agg(['min', 'max'])
    assert (up['max'] - up['min']).abs().max() < 1e-9
    f_up = up['min'].to_dict()
    capture_central = float(eu[eu.scenario == 'Central'].f_capture.iloc[0])

    hafs = nb.loc['HaFS_2021', 'value_t']
    base = {2021: float(nb.loc['HH_2021_LA_all', 'value_t'] + hafs),
            2022: float(nb.loc['HH_2022_LA_all', 'value_t'] + hafs)}   # 2022 uses the 2021 HaFS estimate (flagged)
    base_mean = sum(base.values()) / len(base)
    home_compost = 160000.0                                     # WRAP Household 2022, Table 2 (diary estimate)
    ni_share = float(nat.loc['GB-NIR', 'total_kerbside_food_waste_t'] / nat.total_kerbside_food_waste_t.sum())

    rows = []
    for name, mass, note in [
            ('primary_mean_2021_2022', base_mean, 'mean of 2021 and 2022 (household LA four streams + HaFS 2021)'),
            ('primary_2021', base[2021], 'same-year 2021'),
            ('primary_2022', base[2022], 'household 2022 + HaFS 2021'),
            ('sens_plus_home_compost', base_mean + home_compost, 'adds 2022 home-composted household food waste'),
            ('sens_all_routes_2022', 7033000.0, 'adds sewer and home compost; not comparable with the EU mandatory total')]:
        for sc in ['Low', 'Central', 'High']:
            twh = mass / 1e6 * f[sc]
            rows.append(dict(case=name, scenario=sc, base_food_waste_t=mass, TWh_fuel_per_yr=twh,
                             TWh_upgraded_stage=mass / 1e6 * f_up[sc],
                             MW_fuel_mean=twh * 1e6 / HOURS, UK00_MW=twh * 1e6 / HOURS * (1 - ni_share),
                             UKNI_MW=twh * 1e6 / HOURS * ni_share, note=note))
    # UK-observed household capture (17.0%) instead of the EU27+NO 24.9% used for every EU14 country.
    twh_obs = base_mean / 1e6 * f['Central'] * float(cap.derived_fraction) / capture_central
    rows.append(dict(case='sens_uk_observed_capture', scenario='Central', base_food_waste_t=base_mean,
                     TWh_fuel_per_yr=twh_obs, MW_fuel_mean=twh_obs * 1e6 / HOURS,
                     UKNI_MW=twh_obs * 1e6 / HOURS * ni_share, UK00_MW=twh_obs * 1e6 / HOURS * (1 - ni_share),
                     note=f'Central chain with UK household capture {cap.derived_fraction:.4f} replacing {capture_central}'))
    df = pd.DataFrame(rows)
    out = HERE / 'inputs/uk_resource_v36.csv'
    df.to_csv(out, index=False)

    proxy = json.loads(V35_PROXY.read_text())
    prim = df[(df.case == 'primary_mean_2021_2022') & (df.scenario == 'Central')].iloc[0]
    eu14 = eu[eu.year.between(2020, 2023)].groupby(['scenario', 'year']).deliverable_fuel_TWh_LHV.sum().groupby('scenario').mean()
    summary = dict(
        method='UK base mass x EU14 TWh-per-Mt factor (uniform EU14 parameter chain)',
        factor_TWh_per_Mt=f, factor_upgraded_stage_TWh_per_Mt=f_up,
        stage_note='TWh_fuel_per_yr is the deliverable stage (after plant methane losses and 96% availability); TWh_upgraded_stage is the stage used for model q in V32-V35 (EU14 central 3.1267 TWh vs 2.9416 TWh deliverable)',
        base_t=base, base_mean_t=base_mean, ni_share_of_kerbside=ni_share,
        central_TWh=float(prim.TWh_fuel_per_yr), central_MW_fuel=float(prim.MW_fuel_mean),
        v35_proxy_TWh=proxy['uk_annual_TWh'], ratio_to_v35_proxy=float(prim.TWh_fuel_per_yr / proxy['uk_annual_TWh']),
        eu14_2020_2023_mean_TWh=eu14.to_dict(),
        flags=['2020 and 2023 UK values not available; mean of 2021 and 2022 used for the 2020-2023 equivalent',
               '2022 combines household 2022 with the 2021 HaFS model estimate (no 2022 HaFS estimate published)',
               'UK household capture 17.0% (WRAP) is below the EU27+NO 24.9% applied to all EU14 countries; reported as sensitivity',
               'UK00/UKNI split by 2022 kerbside household food waste (Northern Ireland share)'],
        inputs_sha256={str(p.relative_to(DS)): sha(p) for p in [UKP / 'national_benchmarks.csv', UKP / 'nations_2022_kerbside.csv',
                                                                UKP / 'capture_2022.csv', EU, V35_PROXY]})
    (HERE / 'report/uk_resource_summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float) + '\n')
    print(df.round(4).to_string())
    print(json.dumps({k: summary[k] for k in ['base_t', 'base_mean_t', 'ni_share_of_kerbside', 'central_TWh', 'v35_proxy_TWh', 'ratio_to_v35_proxy', 'eu14_2020_2023_mean_TWh']}, indent=1))


if __name__ == '__main__':
    main()
