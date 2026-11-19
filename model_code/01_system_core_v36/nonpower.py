"""U4/U5 energy, emissions and fee accounting; no ranking or interpretation."""
from pathlib import Path
import sys,json,itertools,math
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,resource_table,dump,sha,now
from process_sources import decode


def crf(rate,years):return rate*(1+rate)**years/((1+rate)**years-1) if rate else 1/years


def vehicle_factors():
    """Reconstruct gas inventories from JEC's rounded aggregate cells and explicit gas ratios."""
    payload=14.290;cold=.031*325*1.5/640
    out={'diesel_B0':dict(CH4_t_per_MWh=0.,N2O_t_per_MWh=(50.31-48.34)/298*.0036/.6603,printed_CO2_g_per_tkm=48.34,printed_CO2eq_AR4_g_per_tkm=50.31)}
    for tech,energy,co2,equ,bog in [('CNG',.8299,46.61,47.26,0.),('LNG',.8303,46.84,47.68,12.15*1000/116000)]:
        work=((equ-co2)*payload-25*(cold+bog))/(25*.2+298*.01)
        ch4=(.2*work+cold+bog)/payload*.0036/energy;n2o=.01*work/payload*.0036/energy
        assert work>0
        assert abs((25*ch4+298*n2o)*energy/.0036-(equ-co2))<1e-10
        out[tech]=dict(CH4_t_per_MWh=ch4,N2O_t_per_MWh=n2o,reconstructed_positive_engine_work_kWh_per_km=work,printed_CO2_g_per_tkm=co2,printed_CO2eq_AR4_g_per_tkm=equ,printed_difference_rounding_interval_g_per_tkm=[equ-co2-.01,equ-co2+.01],boil_off_CH4_g_per_km=bog)
    return dict(inventories=out,source='JEC TTW v5 PDF page 62 Table 45; PDF page 51 Tables 33/34; PDF page 52 Tables 35/36; PDF page 50 equations; PDF page 17 Table 7',payload_t=payload,engine_power_kW=325,cold_starts_per_day=1.5,longhaul_km_per_day=640,original_GWP=dict(CH4=25,N2O=298),reconstruction_note='Aggregate CO2 and CO2eq printed to .01 g/tkm. Diesel non-CO2 is N2O; gas hot CH4:N2O ratio and cold/boil-off inventories separate CH4 from N2O. Derived inventories inherit printed-cell rounding, not unrounded simulation precision.')


def assumptions():
    h=decode(HERE/'inputs/raw/eurostat/hicp_2011_2025.json');idx=dict(zip(h.time.astype(int),h.value))
    factor=lambda year:idx[2024]/idx[year]
    return dict(currency='EUR2024-equivalent engineering budget; HICP EA20 indexation where a reference year is identified',
        GWP100=dict(bio_CH4=27.,fossil_CH4=29.8,N2O=273.),GWP20=dict(bio_CH4=79.7,fossil_CH4=82.5,N2O=273.),
        methane_mass_t_per_MWh=.072,gas_combustion_t_per_MWh=.20196,
        diesel_combustion_t_per_MWh=73.2*.0036,diesel_upstream_t_per_MWh=21.9*.0036,
        gas_upstream_inventory_g_per_MJ=dict(CO2=5.4,CH4=.2,N2O=.00017),
        grid_compression_MWh_per_MWh=.03,vehicle_CNG_compression_MWh_per_MWh=.03,liquefaction_MWh_per_MWh=.0605,
        vehicle_energy_MJ_per_tkm=dict(diesel_B0=.6603,CNG_PI=.8299,LNG_PI=.8303),
        vehicle_reference='JEC TTW v5, PDF page 62 Table 45, 2025 long haul; alternative regional/urban cases in parameter_source_cells.csv',
        vehicle_nonCO2=vehicle_factors(),vehicle_primary_methane_boundary='Full-chain lambda includes vehicle methane; explicit vehicle methane inventory not subtracted again. N2O and avoided fossil-vehicle methane are separate.',
        gate_cost_EUR_per_MWh=dict(low=54*factor(2021),central=84*factor(2021),high=91*factor(2021)),
        gate_cost_source='BIP TF4 2023: PDF page 14 identifies price year 2021; cost categories 54/84/91 are plant-scale/feedstock proxies, not measured country-specific food-waste costs',
        injection_all_in_EUR_per_MWh=dict(low=3*factor(2019),central=5*factor(2019),high=7*factor(2019)),
        injection_source='BIP TF4 conclusion, Navigant 2019 cited benchmark ~5 EUR/MWh. 3/7 are declared engineering uncertainty endpoints.',
        liquefaction_all_in_EUR_per_MWh=dict(low=9*factor(2021),central=12*factor(2021),high=15*factor(2021)),
        liquefaction_source='BIP TF4 conclusion cites DNV 2021 9–15 EUR/MWh; tariff includes electricity, which is not added again to cost.',
        station_capex_EUR_per_annual_MWh=dict(CNG=750000/10000*factor(2016),LNG=1500000/30000*factor(2016)),
        station_source='IEA Bioenergy Task 37 (2022) PDF page 42, Borjesson et al. 2016 benchmarks; publication year used as an explicit price-year proxy for indexation',
        station_FOM_fraction=.03,station_lifetime_years=20,station_FOM_status='engineering assumption, not a quoted observation',
        discount_rates=dict(low=.04,central=.07,high=.10),electricity_budget_EUR_per_MWh=dict(low=50.,central=100.,high=200.),
        electricity_price_status='Engineering process-electricity budget grid, independent of ETS prices; dispatch duals retained separately',
        HICP_factors_to_2024={str(y):factor(y) for y in [2016,2019,2020,2021]},
        notes=['GWP20 uses corrected AR6 biogenic methane 79.7, not superseded 80.8.',
        'Delivered methane output is not reduced by externally supplied electricity; net fossil-energy accounting and process electricity are separate columns.',
        'Identical compression and an existing shared CNG station cancel versus fossil CNG; dedicated new-station cost boundary retained separately.',
        'All-in injection/liquefaction fees are not charged a second time for their electricity.',
        'Historical station costs, uniform fee assumptions and common gate-cost proxies are supplied as scenario inputs, not national empirical estimates.'])


def gas_upstream(params,horizon):
    v=params['gas_upstream_inventory_g_per_MJ'];g=params[horizon]
    return (v['CO2']+v['CH4']*g['fossil_CH4']+v['N2O']*g['N2O'])*.0036


def calculate(use,mef,scenario='central',leakage=.025,horizon='GWP100',params=None,station_boundary='shared_existing_for_fossil_CNG',methane_boundary='full_chain_includes_vehicle'):
    p=params or assumptions();e=p['vehicle_energy_MJ_per_tkm'];rate=p['discount_rates'][scenario]
    if use=='U4':displacement=1.;process=.03;reference_process=0.;fuel='gas'
    elif use=='U5_CNG_diesel':displacement=e['diesel_B0']/e['CNG_PI'];process=.03;reference_process=0.;fuel='diesel'
    elif use=='U5_LNG_diesel':displacement=e['diesel_B0']/e['LNG_PI'];process=.0605;reference_process=0.;fuel='diesel'
    elif use=='U5_CNG_fossil_CNG':displacement=1.;process=.03;reference_process=.03;fuel='gas'
    else:raise ValueError(use)
    leakage_ef=leakage*p['methane_mass_t_per_MWh']*p[horizon]['bio_CH4']
    refef=(p['gas_combustion_t_per_MWh']+gas_upstream(p,horizon)) if fuel=='gas' else p['diesel_combustion_t_per_MWh']+p['diesel_upstream_t_per_MWh']
    process_net=process-reference_process
    reference_vehicle=0.;bio_vehicle_n2o=0.;bio_vehicle_ch4=0.
    if use.startswith('U5'):
        inventories=p['vehicle_nonCO2']['inventories'];gasveh=inventories['LNG' if 'LNG' in use else 'CNG'];ref=inventories['diesel_B0' if fuel=='diesel' else 'CNG']
        reference_vehicle=displacement*(ref['CH4_t_per_MWh']*p[horizon]['fossil_CH4']+ref['N2O_t_per_MWh']*p[horizon]['N2O'])
        bio_vehicle_n2o=gasveh['N2O_t_per_MWh']*p[horizon]['N2O']
        if methane_boundary=='supply_chain_plus_explicit_vehicle':bio_vehicle_ch4=gasveh['CH4_t_per_MWh']*p[horizon]['bio_CH4']
    if methane_boundary not in ['full_chain_includes_vehicle','supply_chain_plus_explicit_vehicle']:raise ValueError(methane_boundary)
    avoided=displacement*refef+reference_vehicle-leakage_ef-process_net*mef-bio_vehicle_n2o-bio_vehicle_ch4
    gate=p['gate_cost_EUR_per_MWh'][scenario];station=0.;conditioning=0.
    if use=='U4':conditioning=p['injection_all_in_EUR_per_MWh'][scenario]
    else:
        tech='LNG' if 'LNG' in use else 'CNG'
        scale={'low':.75,'central':1.,'high':1.25}[scenario]
        station=p['station_capex_EUR_per_annual_MWh'][tech]*scale*(crf(rate,p['station_lifetime_years'])+p['station_FOM_fraction'])
        conditioning=p['liquefaction_all_in_EUR_per_MWh'][scenario] if tech=='LNG' else process*p['electricity_budget_EUR_per_MWh'][scenario]
    gross_station=station;gross_conditioning=conditioning;avoided_station=0.;avoided_conditioning=0.
    if use=='U5_CNG_fossil_CNG' and station_boundary=='shared_existing_for_fossil_CNG':
        avoided_station=station;avoided_conditioning=conditioning;station=0.;conditioning=0.
    if station_boundary not in ['shared_existing_for_fossil_CNG','dedicated_new_station']:raise ValueError(station_boundary)
    return dict(use=use,gross_fossil_displacement_MWh_per_MWh=displacement,displaced_fuel=fuel,
        bio_process_electricity_MWh_per_MWh=process,avoided_reference_process_electricity_MWh_per_MWh=reference_process,
        net_process_electricity_MWh_per_MWh=process_net,marginal_electricity_tCO2e_per_MWh=mef,
        reference_avoided_emissions_t_per_MWh=displacement*refef+reference_vehicle,biogenic_methane_emissions_t_per_MWh=leakage_ef,
        avoided_reference_vehicle_nonCO2_t_per_MWh=reference_vehicle,bio_vehicle_N2O_tCO2e_per_MWh=bio_vehicle_n2o,bio_vehicle_explicit_CH4_tCO2e_per_MWh=bio_vehicle_ch4,methane_loss_boundary=methane_boundary,
        process_emissions_t_per_MWh=process_net*mef,net_abatement_t_per_MWh=avoided,
        common_gate_cost_EUR_per_MWh=gate,conditioning_cost_EUR_per_MWh=conditioning,
        station_fixed_cost_EUR_per_MWh=station,cost_before_fossil_saving_EUR_per_MWh=gate+conditioning+station,
        gross_station_fixed_cost_EUR_per_MWh=gross_station,reference_station_avoided_cost_EUR_per_MWh=avoided_station,
        gross_conditioning_cost_EUR_per_MWh=gross_conditioning,reference_conditioning_avoided_cost_EUR_per_MWh=avoided_conditioning,
        station_cost_boundary=station_boundary if use=='U5_CNG_fossil_CNG' else 'incremental_gas_station_vs_diesel_or_injection',
        carbon_price_in_abatement_cost_numerator=False,policy_credit_in_abatement_cost_numerator=False,
        electrical_energy_source='external electricity, explicit process emissions; no artificial loss of gas LHV')


def main():
    p=assumptions();dump(HERE/'inputs/nonpower_parameters.json',p);dump(HERE/'inputs/vehicle_nonCO2_parameters.json',p['vehicle_nonCO2'])
    # Complete numeric response surfaces, independent of whether the dispatch queue has finished.
    rows=[]
    for use,scenario,leakage,horizon,mef in itertools.product(['U4','U5_CNG_diesel','U5_LNG_diesel','U5_CNG_fossil_CNG'],['low','central','high'],[.01,.025,.042],['GWP100','GWP20'],np.linspace(0,1.5,31)):
        boundaries=['shared_existing_for_fossil_CNG','dedicated_new_station'] if use=='U5_CNG_fossil_CNG' else ['shared_existing_for_fossil_CNG']
        loss_boundaries=['full_chain_includes_vehicle','supply_chain_plus_explicit_vehicle'] if use.startswith('U5') else ['full_chain_includes_vehicle']
        for boundary,loss_boundary in itertools.product(boundaries,loss_boundaries):rows.append(dict(cost_scenario=scenario,leakage=leakage,horizon=horizon,**calculate(use,float(mef),scenario,leakage,horizon,p,station_boundary=boundary,methane_boundary=loss_boundary)))
    pd.DataFrame(rows).to_csv(HERE/'report/nonpower_response_grid.csv',index=False)
    checks=[abs(calculate('U4',.5,params=p)['gross_fossil_displacement_MWh_per_MWh']-1)<1e-14,
            abs(calculate('U5_CNG_diesel',0,params=p)['gross_fossil_displacement_MWh_per_MWh']-.6603/.8299)<1e-14,
            abs(calculate('U5_CNG_fossil_CNG',0,params=p)['net_abatement_t_per_MWh']-calculate('U5_CNG_fossil_CNG',1,params=p)['net_abatement_t_per_MWh'])<1e-14,
            calculate('U5_LNG_diesel',1,params=p)['net_abatement_t_per_MWh']<calculate('U5_LNG_diesel',0,params=p)['net_abatement_t_per_MWh'],
            abs(calculate('U5_CNG_fossil_CNG',.5,params=p)['cost_before_fossil_saving_EUR_per_MWh']-p['gate_cost_EUR_per_MWh']['central'])<1e-12,
            calculate('U5_CNG_fossil_CNG',.5,params=p,station_boundary='dedicated_new_station')['cost_before_fossil_saving_EUR_per_MWh']>p['gate_cost_EUR_per_MWh']['central']]
    checks.append(abs(calculate('U5_CNG_diesel',.5,params=p)['net_abatement_t_per_MWh']-calculate('U5_CNG_diesel',.5,params=p,methane_boundary='supply_chain_plus_explicit_vehicle')['net_abatement_t_per_MWh']-p['vehicle_nonCO2']['inventories']['CNG']['CH4_t_per_MWh']*27)<1e-12)
    assert all(checks)
    dump(HERE/'qa/nonpower_algebra_checks.json',dict(passed=True,checks=len(checks),response_grid_rows=len(rows),utc=now(),parameter_sha256=sha(HERE/'inputs/nonpower_parameters.json')))
    print(json.dumps(dict(passed=True,rows=len(rows))))


if __name__=='__main__':main()
