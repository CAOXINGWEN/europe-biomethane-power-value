"""Separate waste-treatment counterfactual addends; no assumed national disposal mix."""
from pathlib import Path
import sys,itertools,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,dump,sha,now
from nonpower import assumptions,gas_upstream


def factors():
    p=assumptions();rows=[]
    for horizon in ['GWP100','GWP20']:
        gwp=p[horizon]
        for doc,docf,recovery,oxidation in itertools.product([.08,.15,.20],[.5,.7],[0.,.5,.75,.9],[0.,.1]):
            ch4=doc*docf*.5*16/12*(1-recovery)*(1-oxidation)
            rows.append(dict(treatment='landfill',horizon=horizon,DOC=doc,DOCf=docf,CH4_recovery_fraction=recovery,oxidation_fraction=oxidation,CH4_t_per_t_wet=ch4,N2O_t_per_t_wet=0.,avoided_gross_treatment_tCO2e_per_t_wet=ch4*gwp['bio_CH4'],forgone_recovered_electricity_MWh_per_t_wet=0.,forgone_recovered_heat_MWh_per_t_wet=0.,factor_scope='Committed ultimate emissions of one waste cohort; recovered methane flared, no energy export assumed; not an annual FOD inventory',source='IPCC2006 Table 2.4; IPCC2019 Table 3.0/3.1, Equations 3.1–3.3; recovery grid engineering scenarios; DOCf .7 food-specific and .5 legacy bulk sensitivity'))
        for label,ch4_gkg,n2o_gkg in [('low',.03,.06),('central',4.,.24),('high',8.,.6)]:
            ch4=ch4_gkg/1000;n2o=n2o_gkg/1000
            rows.append(dict(treatment='composting',horizon=horizon,parameter_case=label,CH4_t_per_t_wet=ch4,N2O_t_per_t_wet=n2o,avoided_gross_treatment_tCO2e_per_t_wet=ch4*gwp['bio_CH4']+n2o*gwp['N2O'],forgone_recovered_electricity_MWh_per_t_wet=0.,forgone_recovered_heat_MWh_per_t_wet=0.,factor_scope='Direct treatment gases; fertiliser and soil-carbon credits absent',source='Corrected IPCC2006 Chapter 4 (2015 correction), Table 4.1 wet basis; central N2O is .24 g/kg, not superseded .3'))
        for tech,ch4_gt,n2o_gt in [('continuous_stoker_MSW_proxy',.2,47.),('continuous_fluidised_bed_MSW_proxy',0.,67.)]:
            for electric,heat in itertools.product([0.,.1,.3,.6],[0.,.5]):
                ch4=ch4_gt/1e6;n2o=n2o_gt/1e6
                rows.append(dict(treatment='incineration',horizon=horizon,parameter_case=tech,CH4_t_per_t_wet=ch4,N2O_t_per_t_wet=n2o,avoided_gross_treatment_tCO2e_per_t_wet=ch4*gwp['bio_CH4']+n2o*gwp['N2O'],forgone_recovered_electricity_MWh_per_t_wet=electric,forgone_recovered_heat_MWh_per_t_wet=heat,factor_scope='MSW technology proxy for non-CO2 gases; food biogenic CO2 excluded; electricity/heat export assumptions are scenario endpoints, not food-specific measurements',source='IPCC2019 Chapter 5 Tables 5.3 and 5.4, PDF pages 13/16; energy recovery grid engineering assumptions'))
    frame=pd.DataFrame(rows);frame.insert(0,'counterfactual_id',[f'WC{i:03d}' for i in range(len(frame))]);return frame


def main():
    p=assumptions();h=pd.read_csv(ROOT/'dataset_one/02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv')
    h=h[h.year.between(2020,2023)&h.deliverable_fuel_TWh_LHV.notna()]
    ct=h.groupby(['country_code','scenario'],as_index=False)[['AD_wet_feedstock_t','deliverable_fuel_TWh_LHV']].mean().rename(columns={'country_code':'country','deliverable_fuel_TWh_LHV':'biomethane_TWh'})
    uk=pd.read_csv(HERE/'inputs/uk_resource_v36.csv');uk=uk[uk['case'].eq('primary_mean_2021_2022')]
    for row in uk.itertuples():
        proxy=h[h.scenario.eq(row.scenario)].iloc[0]
        wet=row.base_food_waste_t*proxy.f_residual*proxy.f_capture*(1-proxy.f_contamination)*proxy.f_AD
        ct=pd.concat([ct,pd.DataFrame([dict(country='GB',scenario=row.scenario,AD_wet_feedstock_t=wet,biomethane_TWh=row.TWh_fuel_per_yr)])],ignore_index=True)
    frame=factors();frame.to_csv(HERE/'report/waste_treatment_factor_grid.csv',index=False);rows=[]
    for resource in ct.itertuples():
        B=resource.biomethane_TWh*1e6;wet=resource.AD_wet_feedstock_t
        for f in frame.itertuples():
            gas_heat=(p['gas_combustion_t_per_MWh']+gas_upstream(p,f.horizon))/.9
            gross=f.avoided_gross_treatment_tCO2e_per_t_wet*wet
            heat=f.forgone_recovered_heat_MWh_per_t_wet*wet*gas_heat
            elec=f.forgone_recovered_electricity_MWh_per_t_wet*wet
            rows.append(dict(country=resource.country,resource_scenario=resource.scenario,counterfactual_id=f.counterfactual_id,treatment=f.treatment,horizon=f.horizon,AD_wet_feedstock_t=wet,biomethane_MWh=B,avoided_gross_treatment_committed_tCO2e=gross,forgone_heat_recovery_emissions_tCO2e=heat,forgone_electric_recovery_MWh=elec,net_common_addend_at_MEF0_tCO2e=gross-heat,net_common_addend_at_MEF1p5_tCO2e=gross-heat-1.5*elec,combine_formula='Use matching-horizon country electricity MEF: gross treatment minus foregone heat emissions minus foregone electricity times MEF; add once to each end use of the same feedstock',is_national_empirical_treatment_mix=False))
    pd.DataFrame(rows).to_csv(HERE/'report/waste_counterfactual_resource_addends.csv',index=False)
    files=['IPCC2006_waste_data.pdf','IPCC2006_biological_treatment.pdf','IPCC2019_solid_waste_disposal.pdf','IPCC2019_incineration.pdf']
    dump(HERE/'inputs/waste_counterfactual_parameters.json',dict(created_utc=now(),source_sha256={f:sha(HERE/'inputs/raw/ipcc'/f) for f in files},landfill_DOC_food=.15,landfill_DOC_range=[.08,.2],landfill_DOCf_food=.7,landfill_DOCf_legacy_bulk=.5,landfill_MCF=1.,landfill_CH4_fraction=.5,compost_wet_CH4_g_per_kg=4.,compost_wet_N2O_g_per_kg=.24,incineration_nonCO2_factor_status='MSW proxy, not food-specific factors',national_treatment_mix_assigned=False,incineration_heat_replacement_efficiency=.9,heat_efficiency_status='engineering assumption',notes=['Avoided treatment is kept separate from end-use abatement, never automatically used as RED credit.', 'Landfill emissions are committed ultimate emissions of a waste cohort, not contemporaneous annual emissions.','The same AD feedstock supplies all uses. Collection/transport/digestate-soil effects and compost-product credits are not quantified here.','Foregone recovery grids are assumptions, not measured national energy recovery.']))
    checks=[abs(4/1000*27+.24/1000*273-.17352)<1e-12,
            len(ct)==45,
            abs(.15*.7*.5*16/12-.07)<1e-12,
            all(frame.avoided_gross_treatment_tCO2e_per_t_wet>=0),
            all(ct.AD_wet_feedstock_t>0)]
    assert all(checks)
    dump(HERE/'qa/waste_counterfactual_checks.json',dict(passed=True,checks=len(checks),factor_rows=len(frame),resource_addend_rows=len(rows),no_country_disposal_mix_invented=True))
    print(json.dumps(dict(factor_rows=len(frame),resource_addend_rows=len(rows),passed=True)))


if __name__=='__main__':main()
