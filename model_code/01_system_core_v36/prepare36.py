"""Reconcile thermal capacities and freeze technical dispatch inputs before runs."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,V32,RAW,sha,dump,arrays,resource_table,now


def main():
    d=arrays(V32/'inputs/2035_WS13.npz');zones=d['zones'];Z=len(zones);T=8760
    rawfile=RAW/'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data/GenerationCapacities.csv'
    raw=pd.read_csv(rawfile);raw=raw[raw.data_version.eq('ERAA 2025 final')&raw['Target year'].eq(2035)]
    technologies=['Nuclear','Lignite','Hard coal','Light oil','Heavy oil','Biofuel']
    cap=np.zeros((T,Z,len(technologies)));gas=np.zeros((T,Z))
    for k,(z,tech) in enumerate(zip(d['audit_thermal_zone'],d['audit_thermal_technology'])):
        if tech=='Hydrogen':continue
        if tech=='Natural gas':gas[:,z]+=d['audit_thermal_available'][:,k]
        else:cap[:,z,technologies.index(str(tech))]+=d['audit_thermal_available'][:,k]
    mappings=[]
    for z,zone in enumerate(zones):
        extra=float(raw[raw.Market_Node.eq(zone)&raw.Technology.eq('Natural gas')&raw.Operational_Status.str.contains('Inelastic',na=False)].Value.sum())*.92
        gas[:,z]+=extra
        residual=d['other_thermal_cap'][:,z]-cap[:,z,:].sum(axis=1)
        if abs(residual).max()>1e-7:
            if abs(residual-residual[0]).max()>1e-7:raise ValueError('Time-varying unexplained other capacity '+zone)
            if residual[0]>1e-7:raise ValueError('Unexpected positive other capacity '+zone)
            cap[:,z,technologies.index('Biofuel')]+=residual
        mappings.append(dict(zone=str(zone),inelastic_gas_addition_MW=extra,legacy_biomass_pool_subtraction_MW=float(-residual.mean())))
    gaserror=float(abs(gas-d['gas_cap']).max());othererror=float(abs(cap.sum(axis=2)-d['other_thermal_cap']).max())
    if max(gaserror,othererror)>1e-7 or cap.min()<-1e-7:raise ValueError('Thermal reconciliation failed')
    cap=np.maximum(cap,0)
    commodityfile=RAW/'ERAA_2025_Dashboard_RawData/ERAA_2025_DashboardRawData1/Dashboard Raw Data/Commodity Prices.xlsx'
    prices=pd.read_excel(commodityfile,sheet_name='Commodity Prices');prices=prices[prices.Year.eq(2035)&prices.data_version.eq('ERAA 2025 final')]
    price={str(r.Fuel).strip():float(r.Value) for r in prices.itertuples()};carbon=price['CO2 price']
    # Standard ERAA efficiencies; existing gas mix is not disaggregated, so use the registered .50 fallback.
    eta={'Nuclear':.33,'Lignite':.4,'Hard coal':.4,'Light oil':.35,'Heavy oil':.4,'Biofuel':.4}
    ef={'Nuclear':0.,'Lignite':.3636,'Hard coal':.34056,'Light oil':.26676,'Heavy oil':.27864,'Biofuel':0.}
    vom={'Nuclear':9.,'Lignite':3.3,'Hard coal':3.3,'Light oil':1.1,'Heavy oil':3.3,'Biofuel':25.}
    columns=[];cost=[];tz=[];tn=[];eff=[];emissions=[];ledger=[]
    for z,zone in enumerate(zones):
        country=str(d['countries'][z]);group=1 if country in ['BG','CZ'] else 3 if country in ['SI','RO'] else 4 if country=='GR' else 2
        lignite=next(v for k,v in price.items() if k.startswith('Lignite G'+str(group)))
        for k,name in enumerate(technologies):
            if cap[:,z,k].max()<1e-8:continue
            fuel=lignite if name=='Lignite' else price.get(name,0.)
            mc=fuel*3.6/eta[name]+ef[name]/eta[name]*carbon+vom[name]
            columns.append(cap[:,z,k]);cost.append(mc);tz.append(z);tn.append(name);eff.append(eta[name]);emissions.append(ef[name])
            ledger.append(dict(zone=str(zone),technology=name,mean_available_MW=float(cap[:,z,k].mean()),efficiency=eta[name],combustion_tCO2_per_MWh_fuel=ef[name],fuel_price_EUR2024_per_MWh_fuel=fuel*3.6,carbon_EUR2024_per_t=carbon,variable_OM_EUR2024_per_MWh=vom[name],dispatch_cost_EUR2024_per_MWh=mc,efficiency_source='ERAA Common Data standard technology category; aggregate existing fleet mapping assumption'))
    data=dict(zones=zones,thermal_cap=np.array(columns).T,thermal_zone=np.array(tz,dtype=int),thermal_technology=np.array(tn,dtype='U20'),thermal_efficiency=np.array(eff),thermal_combustion_ef=np.array(emissions),thermal_cost=np.array(cost),gas_efficiency=np.full(Z,.50),gas_cost=np.full(Z,(price['Gas']*3.6+.20196*carbon)/.50+1.6))
    np.savez_compressed(HERE/'inputs/thermal_split_2035.npz',**data)
    pd.DataFrame(ledger).to_csv(HERE/'inputs/thermal_parameters.csv',index=False)
    resource_table().to_csv(HERE/'inputs/country_resource_2020_2023_mean.csv',index=False)
    gross=np.zeros(Z);sourcehash={}
    for w in range(1,37):
        ip=V32/'inputs'/f'2035_WS{w:02d}.npz'
        with np.load(ip,allow_pickle=False) as a:
            gross+=a['gross_demand'].mean(axis=0)/36
            for k in ['gas_cap','other_thermal_cap','audit_thermal_available']:
                if not np.array_equal(a[k],d[k]):raise ValueError(f'Weather-dependent thermal input {w}/{k}')
        sourcehash[str(ip.relative_to(ROOT))]=sha(ip)
    weights=np.zeros(Z)
    for country in np.unique(d['countries']):
        ix=d['countries']==country;weights[ix]=gross[ix]/gross[ix].sum()
    # Registered UK-specific household-derived allocation.
    u=pd.read_csv(HERE/'inputs/uk_resource_v36.csv');u=u[u['case'].eq('primary_mean_2021_2022')&u.scenario.eq('Central')].iloc[0]
    weights[zones=='UKNI']=u.UKNI_MW/u.MW_fuel_mean;weights[zones=='UK00']=u.UK00_MW/u.MW_fuel_mean
    np.savez_compressed(HERE/'inputs/resource_zone_weights.npz',zones=zones,countries=d['countries'],weights=weights,mean_gross_demand_MW=gross)
    sourcehash.update({str(p.relative_to(ROOT)):sha(p) for p in [rawfile,commodityfile,RAW/'ERAA_2025_CommonData/Common data/Common Data.xlsx',V32/'scripts/model.py',V32/'scripts/solver_adapter.py',ROOT/'dataset_one/02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv',HERE/'inputs/uk_resource_v36.csv']})
    dump(HERE/'protocol/source_hashes.json',sourcehash)
    dump(HERE/'qa/thermal_split_reconciliation.json',dict(passed=True,created_utc=now(),gas_max_absolute_difference_MW=gaserror,other_max_absolute_difference_MW=othererror,hydrogen_kept_separate=True,mapping_corrections=mappings,existing_gas_efficiency=dict(value=.5,source='Design 3.4 permitted fallback: public model input aggregates existing natural-gas units without their fine technology mix',sensitivity=[.4,.58]),other_efficiencies='ERAA Common Data standard thermal characteristics; existing aggregate category mapping is explicit in thermal_parameters.csv',source_hashes=sourcehash))
    print(json.dumps(dict(passed=True,gas_max_error_MW=gaserror,other_max_error_MW=othererror,thermal_zone_technology_columns=len(columns)),indent=1))


if __name__=='__main__':main()
