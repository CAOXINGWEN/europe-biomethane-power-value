"""One-factor resource experiments and collection-response tables, without use rankings."""
from pathlib import Path
import sys,math
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,resource_table,dump,now


def main():
    original=pd.read_csv(ROOT/'dataset_one/02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv')
    h=original[original.year.between(2020,2023)&original.deliverable_fuel_TWh_LHV.notna()]
    fields=['f_residual','f_capture','f_contamination','f_AD','Y_CH4_wet_Nm3_per_t','eta_real','eta_upgrade','lambda_non_upgrade','availability']
    history=h.groupby(['country_code','scenario'],as_index=False)[fields+['base_food_waste_t','AD_wet_feedstock_t','deliverable_fuel_TWh_LHV']].mean()
    resource=resource_table();lever=[];collection=[];checks=[]
    for country in sorted(resource.country_code.unique()):
        base=resource[resource.country_code.eq(country)].set_index('scenario').deliverable_fuel_TWh_LHV
        source=history[history.country_code.eq(country)] if country!='GB' else history[history.country_code.eq('AT')]
        central=source[source.scenario.eq('Central')].iloc[0];high=source[source.scenario.eq('High')].iloc[0]
        product=1.
        for key in fields:
            a=1-central[key] if key in ['f_contamination','lambda_non_upgrade'] else central[key]
            b=1-high[key] if key in ['f_contamination','lambda_non_upgrade'] else high[key];ratio=b/a;product*=ratio
            lever.append(dict(country=country,changed_factor=key,central_value=float(central[key]),factor_high_value=float(high[key]),other_factors='held at central values',central_biomethane_MWh=float(base.Central*1e6),single_factor_biomethane_MWh=float(base.Central*1e6*ratio),single_factor_increment_MWh=float(base.Central*1e6*(ratio-1)),multiplier=float(ratio),log_multiplier=math.log(ratio),period='2020–2023 mean' if country!='GB' else '2020–2023 proxy, 2021/2022 mean'))
        checks.append(abs(product-base.High/base.Central)<1e-10)
        for capture in np.unique(np.r_[np.linspace(0,1,21),central.f_capture,min(1.,central.f_capture+.10),.17,.85]):
            ratio=capture/central.f_capture
            collection.append(dict(country=country,collection_fraction=float(capture),central_collection_fraction=float(central.f_capture),central_biomethane_MWh=float(base.Central*1e6),collection_only_biomethane_MWh=float(base.Central*1e6*ratio),increment_over_central_MWh=float(base.Central*1e6*(ratio-1)),is_plus_10_percentage_points=bool(abs(capture-central.f_capture-.1)<1e-12),other_conversion_factors='held at Central',is_dispatch_recalculated=False,period='2020–2023 mean' if country!='GB' else '2020–2023 proxy, 2021/2022 mean'))
    pd.DataFrame(lever).to_csv(HERE/'report/resource_single_factor_outputs.csv',index=False)
    pd.DataFrame(collection).to_csv(HERE/'report/resource_collection_response_grid.csv',index=False)
    assert all(checks)
    dump(HERE/'qa/resource_response_checks.json',dict(passed=True,utc=now(),country_checks=len(checks),single_factor_rows=len(lever),collection_rows=len(collection),test='Product of factor multipliers exactly reconstructs registered High/Central deliverable fuel ratio; no use or hypothesis interpretation'))
    print(dict(passed=True,lever_rows=len(lever),collection_rows=len(collection)))


if __name__=='__main__':main()
