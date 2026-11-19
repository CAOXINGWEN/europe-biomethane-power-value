"""Independent U3 absorption-capped ledgers alongside the unaltered design formula."""
import itertools
import numpy as np,pandas as pd
from common36 import HERE,RAW,dump,now
from nonpower import assumptions,gas_upstream


def export(country,marginal,partial=False):
    if country.empty:return
    c=country[country.use.eq('U3')].copy()
    if c.empty:return
    c.to_csv(HERE/'report/U3_absorption_by_weather.csv',index=False)
    keys=['country','variant','site_efficiency','injection_efficiency','scenario']
    groups=c.groupby(keys).mean(numeric_only=True).reset_index();groups.to_csv(HERE/'report/U3_absorption_means.csv',index=False)
    p=assumptions();price=pd.read_excel(RAW/'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data/Commodity Prices.xlsx',sheet_name='Commodity Prices')
    price=float(price[price.Year.eq(2035)&price.data_version.eq('ERAA 2025 final')&price.Fuel.eq('Gas')].Value.iloc[0])*3.6
    ledger=pd.read_csv(HERE/'report/all_use_accounting_sensitivities.csv');base={}
    for row in ledger[ledger.use.eq('U3')].itertuples():base[(row.country,row.variant,row.resource_scenario,row.injection_efficiency,row.horizon,row.leakage,row.cost_scenario)]=row
    rows=[]
    for v in groups.itertuples():
        m=marginal[marginal.country.eq(v.country)];weights=m.identified_biomethane_weight_fraction
        for cap in ['country_annual','zone_hourly']:
            substituted=getattr(v,'U3_'+cap+'_absorbable_MWh');excess=v.biomethane_MWh-substituted
            for horizon,leakage,costcase in itertools.product(['GWP100','GWP20'],[.01,.025,.042],['low','central','high']):
                column='marginal_fossil_ef_GWP20_identified_hours' if horizon=='GWP20' else 'marginal_fossil_ef_identified_hours'
                mef=float((m[column].fillna(0)*weights).sum()/weights.sum()) if weights.sum()>0 else np.nan
                ef=p['gas_combustion_t_per_MWh']+gas_upstream(p,horizon);B=v.biomethane_MWh
                avoided=getattr(v,'avoided_fossil_emissions_'+horizon+'_t')-excess*ef
                net=avoided-B*leakage*.072*p[horizon]['bio_CH4']-.03*B*mef
                # Keep the entire delivered-resource gate/processing budget; unused gas receives no alternate-use credit.
                base_saving=v.delta_gas_fuel_MWh+substituted
                fuel_saving=base_saving*price*{'low':.5,'central':1.,'high':1.5}[costcase]
                original=base.get((v.country,v.variant,v.scenario,v.injection_efficiency,horizon,leakage,costcase))
                saving_all=original.avoided_fossil_fuel_cost_EUR2024-excess*price*{'low':.5,'central':1.,'high':1.5}[costcase] if original is not None else np.nan
                gross_cost=(p['gate_cost_EUR_per_MWh'][costcase]+p['injection_all_in_EUR_per_MWh'][costcase])*B
                net_cost=gross_cost-saving_all
                rows.append(dict(country=v.country,variant=v.variant,resource_scenario=v.scenario,injection_efficiency=v.injection_efficiency,absorption_constraint=cap,horizon=horizon,leakage=leakage,cost_scenario=costcase,delivered_biomethane_MWh=B,design_unconstrained_substitution_MWh=B,capped_direct_substitution_MWh=substituted,unabsorbed_biomethane_MWh=excess,gross_fossil_displacement_MWh=v.dispatch_fossil_fuel_displacement_MWh+substituted,avoided_fossil_emissions_t=avoided,net_abatement_conditional_MEF_t=net,common_gate_plus_processing_cost_EUR2024=gross_cost,avoided_gas_cost_only_EUR2024=fuel_saving,avoided_all_fossil_fuel_cost_EUR2024=saving_all,net_incremental_cost_excluding_carbon_EUR2024=net_cost,abatement_cost_EUR2024_per_t=net_cost/net if net>0 else np.nan,weather_count=int(c[c.country.eq(v.country)&c.variant.eq(v.variant)&c.scenario.eq(v.scenario)&c.injection_efficiency.eq(v.injection_efficiency)].weather.nunique()),accounting_note='No re-solve; capacity relief and resource denominator unchanged. Absorbed gas capped against the resulting power-plant gas use; no alternate use/credit assigned to excess. Non-gas dispatch cost changes retained from the ordinary use ledger.'))
    pd.DataFrame(rows).to_csv(HERE/'report/U3_absorption_capped_accounting.csv',index=False)
    dump(HERE/'report/U3_absorption_scope.json',dict(created_utc=now(),partial=partial,registered_unconstrained_formula_retained=True,alternative_annual_country_and_hourly_zone_caps=True,causal_resource_origin_attribution=False,unused_gas_receives_other_use_credit=False,imported_gas_network_transport_modelled=False))
