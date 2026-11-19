"""Cost and emissions ledgers, uncertainty grids and resource response tables."""
from pathlib import Path
import sys,json,itertools,math
from functools import lru_cache
import numpy as np,pandas as pd
from common36 import HERE,ROOT,RAW,load_input,design,dump,sha,now
from nonpower import assumptions,calculate,gas_upstream,crf


@lru_cache(maxsize=3)
def engineering_inputs(scenario):
    d=load_input(23,scenario,'R2')
    fields={k:d[k].copy() for k in ['countries','zones','q','eta_bio','beta','compression_kappa']}
    peaks=[d['resource_q_hourly'].max(axis=0)]
    del d
    peaks.append(load_input(13,scenario,'R1')['resource_q_hourly'].max(axis=0))
    fields['peak_gas_by_profile']=np.array(peaks)
    return fields


@lru_cache(maxsize=405)
def assets(country,scenario,costcase,site_efficiency=.4):
    p=assumptions();d=engineering_inputs(scenario);sel=d['countries']==country;q=d['q'][sel]
    ds=design(scenario);P=np.array(ds['power'])[sel];S=np.array(ds['energy'])[sel]
    parameters=json.loads((ROOT/'dataset_one/39_public_validation_system_comparison_v27/economics/cost_scenarios.json').read_text())['cases'][costcase]
    conversion=p['HICP_factors_to_2024']['2020'];rate=parameters['rate'];engine_annuity=crf(rate,parameters['engine_life'])
    peak=float(d['peak_gas_by_profile'][:,sel].sum(axis=1).max())
    u1=site_efficiency*peak*(parameters['engine_capex']*engine_annuity+parameters['engine_fom'])*conversion
    # Reuse only the pure V33 engineering accounting; its files remain read-only.
    old=ROOT/'dataset_one/45_followup_v33/scripts'
    if str(old) not in sys.path:sys.path.insert(0,str(old))
    import v33_c1_costs as cost
    data=dict(q=q,zones=d['zones'][sel],eta_bio=d['eta_bio'],beta=d['beta'],compression_kappa=d['compression_kappa'])
    pair=cost.asset_pair(data,P,S,costcase,'representative_legacy')
    total=pair['total'];u2=total['annual_fixed_total_EUR']*conversion
    # V33 total includes the boost generator. Add the base engine rated to supply B plus compression.
    u2+=float(.4*q.sum()/.97)*(parameters['engine_capex']*engine_annuity+parameters['engine_fom'])*conversion
    detail=dict(total_asset_components_EUR2020=total,total_container_rows=[v for v in pair['container_rows'] if v['side']=='total'],total_inventory_rows=[v for v in pair['inventory_rows'] if v['side']=='total'],parameters_EUR2020=pair['parameters'],HICP_2020_to_2024=conversion,U2_base_engine_annual_EUR2024=float(.4*q.sum()/.97)*(parameters['engine_capex']*engine_annuity+parameters['engine_fom'])*conversion,baseline_for_new_experiment='No food biomethane and no food-biomethane assets; old 16q reference/delta costs are not used')
    return dict(U1_fixed_annual_EUR2024=float(u1),U2_fixed_annual_EUR2024=float(u2),engine_variable_OM_EUR2024_per_MWh=parameters['engine_vom']*conversion,peak_site_capacity_MW=site_efficiency*peak,boost_capacity_MW=float(P.sum()),gas_buffer_MWh=float(S.sum()),asset_policy='V33 representative 2.5 MWfuel sites, complete U2 assets relative to no-biomethane R0, with base engine added; historical price year converted using EA20 HICP',engineering_detail=detail)


def finish(country,marginal,partial=False):
    p=assumptions();out=HERE/'report';rows=[];policy=[]
    # Save resource histories and one-parameter resource lever calculations as numeric inputs.
    path=ROOT/'dataset_one/02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv'
    history=pd.read_csv(path);history.to_csv(out/'resource_country_year_all_stages.csv',index=False)
    lever=[]
    active=history[history.deliverable_fuel_TWh_LHV.notna()&history.year.between(2020,2023)]
    factors=['f_residual','f_capture','f_contamination','f_AD','Y_CH4_wet_Nm3_per_t','eta_real','eta_upgrade','lambda_non_upgrade','availability']
    for (ct,year),group in active.groupby(['country_code','year']):
        central=group[group.scenario.eq('Central')].iloc[0];high=group[group.scenario.eq('High')].iloc[0]
        for factor in factors:
            a=1-central[factor] if factor in ['f_contamination','lambda_non_upgrade'] else central[factor]
            b=1-high[factor] if factor in ['f_contamination','lambda_non_upgrade'] else high[factor]
            lever.append(dict(country=ct,year=int(year),factor=factor,central_value=central[factor],high_value=high[factor],multiplicative_change=b/a,log_change=math.log(b/a)))
    pd.DataFrame(lever).to_csv(out/'resource_lever_factors.csv',index=False)
    if marginal.empty or country.empty:return
    mef={}
    for ct,g in marginal.groupby('country'):
        coverage=float(g.identified_biomethane_weight_fraction.mean());weights=g.identified_biomethane_weight_fraction
        identified=float((g.marginal_fossil_ef_identified_hours.fillna(0)*weights).sum()/weights.sum()) if weights.sum()>0 else np.nan
        identified20=float((g.marginal_fossil_ef_GWP20_identified_hours.fillna(0)*weights).sum()/weights.sum()) if weights.sum()>0 else np.nan
        mef[ct]=dict(point=identified,point20=identified20,coverage=coverage,low=identified*coverage if np.isfinite(identified) else 0.,high=(identified*coverage if np.isfinite(identified) else 0.)+1.5*(1-coverage),low20=identified20*coverage if np.isfinite(identified20) else 0.,high20=(identified20*coverage if np.isfinite(identified20) else 0.)+1.5*(1-coverage),weather_count=len(g))
    pd.DataFrame([dict(country=k,**v) for k,v in mef.items()]).to_csv(out/'country_marginal_factors_mean.csv',index=False)
    pri=pd.read_excel(RAW/'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data/Commodity Prices.xlsx',sheet_name='Commodity Prices')
    pri=pri[pri.Year.eq(2035)&pri.data_version.eq('ERAA 2025 final')];prices={str(x.Fuel).strip():float(x.Value)*3.6 for x in pri.itertuples() if str(x.Fuel).strip()!='CO2 price'}
    dep=pd.read_csv(HERE/'inputs/annual_import_dependency_2014_2023.csv');dep=dep[dep.year.between(2020,2023)].groupby('country').dependency_fraction.mean().to_dict()
    # These are the values present in the preserved ACER Table 10 excerpt; omitted countries remain missing.
    vf=pd.read_csv(HERE/'inputs/country_voll_acer2023.csv')
    voll=vf[vf.VOLL_EUR_per_MWh.notna()].set_index('country').VOLL_EUR_per_MWh.to_dict()
    keys=['country','use','variant','site_efficiency','injection_efficiency','scenario'];cache={}
    for key,g in country.groupby(keys):
        ct,use,variant,site_eta,inj_eta,scenario=key
        if len(g)!=36 and not partial:raise ValueError('Incomplete weather group '+str(key))
        v=g.mean(numeric_only=True);B=float(v.biomethane_MWh);m=mef[ct]
        if not np.isfinite(m['point']):continue
        for costcase,leakage,horizon in itertools.product(['low','central','high'],[.01,.025,.042],['GWP100','GWP20']):
            mp=m['point20'] if horizon=='GWP20' else m['point'];ml=m['low20'] if horizon=='GWP20' else m['low'];mh=m['high20'] if horizon=='GWP20' else m['high']
            akey=(ct,scenario,costcase,site_eta)
            if akey not in cache:cache[akey]=assets(ct,scenario,costcase,site_eta)
            asset=cache[akey];factor={'low':.5,'central':1.,'high':1.5}[costcase]
            group=1 if ct in ['BG','CZ'] else 3 if ct in ['SI','RO'] else 4 if ct=='GR' else 2
            lignite=next(price for name,price in prices.items() if name.startswith('Lignite G'+str(group)))
            savings=(v.delta_gas_fuel_MWh*prices['Gas']+v.delta_Lignite_fuel_MWh*lignite+v.delta_Hard_coal_fuel_MWh*prices['Hard coal']+v.delta_Light_oil_fuel_MWh*prices['Light oil']+v.delta_Heavy_oil_fuel_MWh*prices['Heavy oil']+v.direct_grid_gas_substitution_MWh*prices['Gas'])*factor
            gate=p['gate_cost_EUR_per_MWh'][costcase]*B
            if use in ['U1','U2']:
                fixed=asset[use+'_fixed_annual_EUR2024'];proc=0.
                electric=(site_eta if use=='U1' else .3)*B
                if use=='U2':electric+=float(v.boost_MWh+v.compression_electricity_MWh)
                variable=electric*asset['engine_variable_OM_EUR2024_per_MWh']
                cost=gate+fixed+variable-savings
            else:
                fixed=0.;proc=.03;variable=p['injection_all_in_EUR_per_MWh'][costcase]*B;cost=gate+variable-savings
            leak=B*leakage*.072*p[horizon]['bio_CH4'];avoided=float(v['avoided_fossil_emissions_'+horizon+'_t'])
            point=avoided-leak-proc*B*mp;lower=avoided-leak-proc*B*mh;upper=avoided-leak-proc*B*ml
            rows.append(dict(country=ct,use=use,variant=variant,resource_scenario=scenario,site_efficiency=site_eta,injection_efficiency=inj_eta,cost_scenario=costcase,leakage=leakage,horizon=horizon,weather_count=g.weather.nunique(),biomethane_MWh=B,fossil_displacement_MWh=float(v.gross_fossil_displacement_MWh),fossil_displacement_per_MWh=float(v.displacement_MWh_per_MWh),gas_displacement_per_MWh=float(v.delta_gas_fuel_MWh+v.direct_grid_gas_substitution_MWh)/B,avoided_combustion_emissions_t=float(v.avoided_combustion_emissions_t),abatement_point_conditional_mef_t=point,abatement_lower_mef_bound_t=lower,abatement_upper_mef_bound_t=upper,identified_marginal_hour_weight=m['coverage'],common_gate_cost_EUR2024=gate,asset_fixed_EUR2024=fixed,variable_processing_EUR2024=variable,avoided_fossil_fuel_cost_EUR2024=float(savings),net_incremental_cost_excluding_carbon_EUR2024=float(cost),abatement_cost_EUR2024_per_t=float(cost/point) if point>0 else np.nan,adequacy_value_primary_EUR=float(v.delta_EENS_at_dispatch_location_MWh)*12240,adequacy_value_country_VOLL_EUR=float(v.delta_EENS_at_dispatch_location_MWh)*voll[ct] if ct in voll else np.nan,import_dependency_fraction=dep.get(ct,np.nan),import_displacement_weighted_proxy_MWh=float(v.gross_fossil_displacement_MWh)*dep.get(ct,np.nan),country_attribution=('mixed: producer direct credit plus dispatch-location changes; unconstrained gas absorption' if use=='U3' else 'dispatch location, concurrent deployment'),policy_money_in_cost_numerator=False))
    resource=country[country.variant.eq('primary')][['country','scenario','biomethane_MWh']].drop_duplicates()
    for item in resource.itertuples(index=False):
        ct,scenario,B=item;m=mef[ct]
        if not np.isfinite(m['point']):continue
        for use,costcase,leakage,horizon in itertools.product(['U4','U5_CNG_diesel','U5_LNG_diesel','U5_CNG_fossil_CNG'],['low','central','high'],[.01,.025,.042],['GWP100','GWP20']):
            mp=m['point20'] if horizon=='GWP20' else m['point'];ml=m['low20'] if horizon=='GWP20' else m['low'];mh=m['high20'] if horizon=='GWP20' else m['high']
            k=calculate(use,mp,costcase,leakage,horizon,p)
            price=(prices['Gas'] if k['displaced_fuel']=='gas' else prices['Light oil'])*{'low':.5,'central':1.,'high':1.5}[costcase]
            saved=k['gross_fossil_displacement_MWh_per_MWh']*price
            net=k['cost_before_fossil_saving_EUR_per_MWh']-saved;abat=k['net_abatement_t_per_MWh']
            rows.append(dict(country=ct,use=use,variant='primary',resource_scenario=scenario,site_efficiency=.4,injection_efficiency=.5,cost_scenario=costcase,leakage=leakage,horizon=horizon,weather_count=m['weather_count'],biomethane_MWh=B,fossil_displacement_MWh=k['gross_fossil_displacement_MWh_per_MWh']*B,fossil_displacement_per_MWh=k['gross_fossil_displacement_MWh_per_MWh'],gas_displacement_per_MWh=k['gross_fossil_displacement_MWh_per_MWh'] if k['displaced_fuel']=='gas' else 0.,avoided_combustion_emissions_t=k['gross_fossil_displacement_MWh_per_MWh']*(.20196 if k['displaced_fuel']=='gas' else .26352)*B,abatement_point_conditional_mef_t=abat*B,abatement_lower_mef_bound_t=calculate(use,mh,costcase,leakage,horizon,p)['net_abatement_t_per_MWh']*B,abatement_upper_mef_bound_t=calculate(use,ml,costcase,leakage,horizon,p)['net_abatement_t_per_MWh']*B,identified_marginal_hour_weight=m['coverage'],common_gate_cost_EUR2024=k['common_gate_cost_EUR_per_MWh']*B,asset_fixed_EUR2024=k['station_fixed_cost_EUR_per_MWh']*B,variable_processing_EUR2024=k['conditioning_cost_EUR_per_MWh']*B,avoided_fossil_fuel_cost_EUR2024=saved*B,net_incremental_cost_excluding_carbon_EUR2024=net*B,abatement_cost_EUR2024_per_t=net/abat if abat>0 else np.nan,adequacy_value_primary_EUR=np.nan,import_dependency_fraction=dep.get(ct,np.nan),import_displacement_weighted_proxy_MWh=k['gross_fossil_displacement_MWh_per_MWh']*B*dep.get(ct,np.nan),country_attribution='direct national end-use calculation',policy_money_in_cost_numerator=False))
    table=pd.DataFrame(rows);table.to_csv(out/'all_use_accounting_sensitivities.csv',index=False)
    matched=((table.resource_scenario.eq('Low')&table.leakage.eq(.042))|(table.resource_scenario.eq('Central')&table.leakage.eq(.025))|(table.resource_scenario.eq('High')&table.leakage.eq(.01)))
    primary=table[table.variant.eq('primary')&table.cost_scenario.eq('central')&matched&table.horizon.eq('GWP100')]
    primary.to_csv(out/'all_use_primary_numeric_ledger.csv',index=False)
    for r in primary.itertuples():
        B=r.biomethane_MWh
        for gasprice,carbon,credit in itertools.product([15,25,38.170422,50,75,100],[0,45,100,197.505516,300],[0,20,50,100]):
            # Policy scenarios are numerical budgets; credits are not observed RED certificate prices.
            carbon_base=r.avoided_combustion_emissions_t/B
            qualifies=r.use.startswith('U5') and r.country!='GB'
            nongas=r.avoided_fossil_fuel_cost_EUR2024/B-r.gas_displacement_per_MWh*prices['Gas']
            val=r.gas_displacement_per_MWh*gasprice+nongas+carbon_base*carbon+(credit if qualifies else 0)-(r.common_gate_cost_EUR2024+r.asset_fixed_EUR2024+r.variable_processing_EUR2024)/B
            policy.append(dict(country=r.country,use=r.use,resource_scenario=r.resource_scenario,gas_price_EUR2024_per_MWh=gasprice,carbon_price_EUR2024_per_t=carbon,hypothetical_transport_credit_EUR2024_per_MWh=credit if qualifies else 0,net_value_EUR2024_per_MWh=val,credit_is_hypothetical=True,UK_not_assigned_EU_transport_credit=r.country=='GB',price_axis_note='Gas price varies; other fuel prices fixed to ERAA. Carbon applies only to avoided combustion; price/credit sweeps are hypothetical budgets.'))
    pd.DataFrame(policy).to_csv(out/'policy_price_response_grid.csv',index=False)
    asset_details=[dict(country=k[0],resource_scenario=k[1],cost_scenario=k[2],site_efficiency=k[3],**value) for k,value in cache.items()]
    dump(out/'power_asset_accounting_details.json',asset_details)
    pd.DataFrame([{k:v for k,v in row.items() if k!='engineering_detail'} for row in asset_details]).to_csv(out/'power_asset_summary.csv',index=False)
    # No best-use labels, hypothesis decisions or manuscript statements are generated.
    dump(out/'accounting_scope.json',dict(created_utc=now(),partial=partial,interpretation_produced=False,cost_currency='EUR2024',point_estimate_scope='conditional mean of identified R0 marginal generators; unresolved share propagated as [0,1.5] t/MWh electricity bounds',upstream_sources=dict(gas='JRC142012 Table 2, AR6 inventory recharacterisation',coal_and_lignite='JRC132057 Table 11 LCA minus direct factors, fixed upstream characterisation in GWP20 sensitivity',oil='JRC142012 Table 4'),national_effects='Country of changed dispatch under concurrent deployment, not a single-country causal counterfactual',country_VOLL_missing_retained=True,UK_2014_2019_resource_not_imputed=True,dispatch_reoptimized_for_cost_and_policy_sweeps=False,transport_vehicle_purchase_premium_quantified=False,diesel_fuel_price_proxy='ERAA2035 Light oil commodity price, excluding road-fuel retail taxes'))
