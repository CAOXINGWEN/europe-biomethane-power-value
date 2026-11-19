"""Fixed-dispatch gas heat-rate sensitivity, distinct from re-solved H72 injection arms."""
import numpy as np,pandas as pd
from common36 import HERE,RAW,dump,now
from nonpower import assumptions,gas_upstream


def export(country,marginal,partial=False):
    if country.empty:return
    path=HERE/'report/all_use_accounting_sensitivities.csv'
    if not path.exists():return
    p=assumptions();table=pd.read_csv(path)
    mean=country.groupby(['country','use','variant','site_efficiency','injection_efficiency','scenario']).mean(numeric_only=True)
    gas_delta={key:row.delta_gas_fuel_MWh for key,row in mean.iterrows()}
    share={}
    for ct,g in marginal.groupby('country'):
        weight=g.identified_biomethane_weight_fraction;share[ct]=float((g.gas_source_fraction_identified_weight.fillna(0)*weight).sum()/weight.sum()) if weight.sum() else np.nan
    price=pd.read_excel(RAW/'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data/Commodity Prices.xlsx',sheet_name='Commodity Prices')
    gasprice=float(price[price.Year.eq(2035)&price.data_version.eq('ERAA 2025 final')&price.Fuel.eq('Gas')].Value.iloc[0])*3.6
    output=[];identity_errors=[]
    for r in table.itertuples():
        key=(r.country,r.use,r.variant,r.site_efficiency,r.injection_efficiency,r.resource_scenario)
        if r.use in ['U1','U2','U3'] and key not in gas_delta:raise KeyError(('Missing dispatch group',key))
        old=gas_delta.get(key,0.);B=r.biomethane_MWh;ef=p['gas_combustion_t_per_MWh']+gas_upstream(p,r.horizon)
        process={'U1':0.,'U2':0.,'U3':.03,'U4':.03,'U5_CNG_diesel':.03,'U5_LNG_diesel':.0605,'U5_CNG_fossil_CNG':0.}[r.use]
        for eta in [.4,.5,.58]:
            change=old*(.5/eta-1);mef_change=share[r.country]*ef*(1/eta-1/.5)
            abatement=r.abatement_point_conditional_mef_t+change*ef-process*B*mef_change
            cost=r.net_incremental_cost_excluding_carbon_EUR2024-change*gasprice*{'low':.5,'central':1.,'high':1.5}[r.cost_scenario]
            displacement=r.fossil_displacement_MWh+change
            output.append(dict(country=r.country,use=r.use,variant=r.variant,resource_scenario=r.resource_scenario,site_efficiency=r.site_efficiency,H72_injection_capacity_efficiency=r.injection_efficiency,ordinary_gas_fuel_efficiency=eta,cost_scenario=r.cost_scenario,leakage=r.leakage,horizon=r.horizon,weather_count=r.weather_count,biomethane_MWh=B,fossil_displacement_MWh=displacement,fossil_displacement_per_MWh=displacement/B,net_abatement_conditional_MEF_t=abatement,net_incremental_cost_excluding_carbon_EUR2024=cost,abatement_cost_EUR2024_per_t=cost/abatement if abatement>0 else np.nan,gas_marginal_source_fraction_of_identified_hours=share[r.country],process_marginal_emission_factor_adjustment=mef_change,dispatch_reoptimized=False,merit_order_reoptimized=False,scope='Fuel conversion, known gas-marginal electricity factor and fuel expenditure revaluation of fixed dispatch; separately re-solved H72 capacity sensitivities retain their own variant label'))
            if eta==.5:identity_errors.append(max(abs(abatement-r.abatement_point_conditional_mef_t),abs(cost-r.net_incremental_cost_excluding_carbon_EUR2024),abs(displacement-r.fossil_displacement_MWh)))
    pd.DataFrame(output).to_csv(HERE/'report/fixed_dispatch_gas_efficiency_sensitivity.csv',index=False)
    assert max(identity_errors,default=0)<1e-8
    dump(HERE/'qa/fuel_efficiency_revaluation_checks.json',dict(passed=True,partial=partial,utc=now(),rows=len(output),reference_eta_identity_checks=len(identity_errors),maximum_reference_identity_error=max(identity_errors,default=0),dispatch_held_fixed=True))
