"""Parse inert public data, preserve missing values and record parameter cells."""
from pathlib import Path
import sys,json,re
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd,fitz
from common36 import HERE,ROOT,dump,sha,now


def decode(path):
    v=json.loads(path.read_text());ids=v['id'];size=v['size']
    labels={k:{pos:code for code,pos in v['dimension'][k]['category']['index'].items()} for k in ids};out=[]
    for index,value in v['value'].items():
        position=np.unravel_index(int(index),size)
        row={key:labels[key][int(p)] for key,p in zip(ids,position)}
        row.update(value=value,status=v.get('status',{}).get(index,''));out.append(row)
    return pd.DataFrame(out)


def evidence(key,value,unit,path,page=None,table=None,printed=None,note=''):
    p=HERE/path
    if page is not None:
        with fitz.open(p) as pdf:text=pdf[page-1].get_text()
        if printed is not None and printed not in text:raise ValueError(f'Missing source cell {key}: {printed}')
    return dict(parameter=key,value=value,unit=unit,source_file=path,source_sha256=sha(p),pdf_page=page,table=table,printed_cell=printed,note=note)


def main():
    raw=HERE/'inputs/raw';rows=[]
    balances=decode(raw/'eurostat/nrg_bal_c.json');balances.to_csv(HERE/'inputs/eurostat_gas_balance_long.csv',index=False)
    dependency=decode(raw/'eurostat/nrg_ind_id.json');dependency.to_csv(HERE/'inputs/eurostat_gas_import_dependency.csv',index=False)
    ukpath=raw/'dukes/DUKES_4.1_2026.xlsx';f=pd.read_excel(ukpath,sheet_name='4.1',header=None);uk=[]
    for j in range(1,f.shape[1]):
        match=re.match(r'(\d{4})',str(f.iat[5,j]));year=int(match.group(1)) if match else 0
        if not 2014<=year<=2023:continue
        for i in range(6,len(f)):
            value=f.iat[i,j]
            if isinstance(value,(int,float)) and np.isfinite(value):
                uk.append(dict(country='GB',year=year,flow=str(f.iat[i,0]),value_GWh_GCV=float(value),value_GWh_NCV=float(value)*.9,NCV_GCV_ratio=.9,source='DUKES 2026 Table 4.1',source_sheet='4.1',source_excel_row=i+1,source_excel_column=j+1))
    uk=pd.DataFrame(uk);uk.to_csv(HERE/'inputs/uk_gas_balance_2014_2023.csv',index=False)
    mapping={'FC_IND_E':['Industry'],'FC_OTH_HH_E':['Domestic'],'FC_OTH_CP_E':['Public administration','Commercial'],'FC_TRA_E':['Transport'],'FC_TRA_ROAD_E':['Road'],'TI_EHG_E':['Electricity generation','Heat generation [note 6]'],'IMP':['Imports'],'EXP':['Exports'],'GAE':['Total supply']}
    gas=[]
    for row in balances.itertuples():
        if row.geo=='UK' or row.nrg_bal not in mapping:continue
        gas.append(dict(country={'EL':'GR'}.get(row.geo,row.geo),year=int(row.time),flow=row.nrg_bal,value_MWh_NCV=row.value*1000,source='Eurostat nrg_bal_c',status=row.status,scope='Natural gas, net calorific value'))
    udep=[]
    for year,part in uk.groupby('year'):
        get=lambda key:float(part[part.flow.eq(key)].iloc[0].value_GWh_GCV)
        for flow,names in mapping.items():
            # Electricity generation appears again as energy-sector own use; select the first transformation row.
            value=sum(float(part[part.flow.eq(name)].iloc[0].value_GWh_NCV) for name in names)
            if flow=='EXP':value=-value
            gas.append(dict(country='GB',year=int(year),flow=flow,value_MWh_NCV=value*1000,source='DUKES 2026 Table 4.1; NCV/GCV=.9',status='',scope='Natural gas; harmonised approximate NCV'))
        net=get('Imports')+get('Exports');gae=get('Production')+net+get('Stock change [note 2]')
        udep.append(dict(country='GB',year=int(year),dependency_fraction=net/gae,source='DUKES derived (imports-exports)/(production+imports-exports+stock change); excludes transfers',method='Net import dependence; signed and not clipped'))
    pd.DataFrame(gas).to_csv(HERE/'inputs/annual_sector_gas_2014_2023.csv',index=False)
    dep=[dict(country={'UK':'GB','EL':'GR'}.get(r.geo,r.geo),year=int(r.time),dependency_fraction=r.value/100,source='Eurostat nrg_ind_id',method='Net import dependence; signed and not clipped') for r in dependency.itertuples() if r.geo!='UK']+udep
    pd.DataFrame(dep).to_csv(HERE/'inputs/annual_import_dependency_2014_2023.csv',index=False)
    for source,target in [('total_sector_balances.json','total_sector_energy.csv'),('population.json','population_2020_2023.csv'),('hicp_2020_2025.json','hicp_2020_2025.csv')]:
        p=raw/'eurostat'/source
        if p.exists():decode(p).to_csv(HERE/'inputs'/target,index=False)
    ep='inputs/raw/jec/JEC_TTW_HDV_v5.pdf'
    for label,values,page,table in [
        ('regional_2025',(2.707,3.334,3.286),61,'39'),
        ('longhaul_2025',(.6603,.8299,.8303),62,'45'),
        ('regional_2016',(2.989,3.655,3.602),60,'38'),
        ('longhaul_2016',(.7292,.9097,.9103),62,'44'),
        ('urban_rigid_2025',(3.065,3.749,3.714),71,'51')]:
        for fuel,value in zip(['diesel_B0','CNG_PI','LNG_PI'],values):
            printed=f'{value:.4f}' if value<1 else f'{value:.3f}'
            rows.append(evidence(label+'_'+fuel,value,'MJ/tkm',ep,page,table,printed,note='Payload weighted; conventional ICE; urban group is a rigid truck, not a bus'))
    jp='inputs/raw/jrc/JRC142012_2026_emission_values.pdf'
    rows.extend([
        evidence('gas_supply_CO2',5.4,'g/MJ fuel',jp,18,'2','5.4'),
        evidence('gas_supply_CH4',.2,'g/MJ fuel',jp,18,'2','0.2'),
        evidence('gas_supply_N2O',.00017,'g/MJ fuel',jp,18,'2','1.7E-04'),
        evidence('diesel_supply_GHG',21.9,'gCO2e/MJ fuel',jp,20,'4','21.9'),
        evidence('HFO_supply_GHG',5.7,'gCO2e/MJ fuel',jp,20,'4','5.7'),
        evidence('liquefaction_electricity',.0605,'MWh electricity/MWh liquid methane',jp,43,'section 5.1.9','0.0605'),
        evidence('CNG_compression_default_GHG',2.4,'gCO2e/MJ fuel',jp,43,'section 5.1.9','2.4',note='Reference default only; primary calculations use explicit electricity and country marginal factors, never both.'),
    ])
    pd.DataFrame(rows).to_csv(HERE/'inputs/parameter_source_cells.csv',index=False)
    for folder in ['jec','jrc','costs','ipcc']:
        for p in (raw/folder).glob('*.pdf'):
            with fitz.open(p) as doc:
                target=HERE/'inputs/extracted'/(p.stem+'.txt');target.parent.mkdir(exist_ok=True)
                target.write_text('\n'.join(f'[PDF page {i+1}]\n'+page.get_text() for i,page in enumerate(doc)))
    dump(HERE/'qa/external_data_checks.json',dict(passed=True,utc=now(),parameter_cells=len(rows),sector_gas_rows=len(gas),dependency_rows=len(dep),uk_years=sorted(map(int,uk.year.unique())),uk_calorific_conversion=.9,negative_and_over_100_percent_dependencies_retained=True,missing_values_not_filled=True))
    print(json.dumps(dict(passed=True,parameter_cells=len(rows),sector_gas_rows=len(gas),dependency_rows=len(dep))))


if __name__=='__main__':main()
