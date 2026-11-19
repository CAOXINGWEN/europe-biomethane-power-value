"""Freeze the existing evidence for the four-section Results and relocate source tables."""
import sys,re,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *

def main():
 doc=ROOT/'manuscript/paper2_use_value/revision_v46';dest=doc/'supplement_source_data';dest.mkdir(parents=True,exist_ok=True)
 mapping={
  'TableS1_resource_country_year.csv':V37/'inputs/resource_v37_country_year.csv',
  'TableS2_uses_costs_and_capacity.csv':V37/'report/preliminary_value_cost_by_year.csv',
  'TableS3_flexible_reserve_by_weather.csv':V37/'report/preliminary_F2_by_weather.csv',
  'TableS4_country_response_by_case.csv':V37/'report/preliminary_F3_by_case.csv',
  'TableS5_legacy_sensitivities.csv':V37/'report/final_F4F5_summary.csv',
  'TableS6_biogas_end_uses_all_feedstocks.csv':V37/'report/biogas_end_use_eurostat.csv',
  'TableS7_registered_criteria.csv':V37/'report/final_criteria_v37.csv',
  'TableS8_process_parameters_source_cells.csv':V36/'inputs/parameter_source_cells.csv',
 }
 prov={}
 for name,src in mapping.items():
  shutil.copyfile(src,dest/name);assert sha(src)==sha(dest/name)
  prov[name]=dict(source=relative(src),sha256=sha(src),copied_utc=now(),unchanged=True)
 dump(dest/'source_manifest.json',prov)
 f=pd.read_csv(V37/'report/preliminary_F1_by_weather.csv')
 g=pd.read_csv(ROOT/'manuscript/paper2_use_value/source_data/arrays/counterfactual_by_weather.csv')
 m=f.merge(g,on=['year','weather'],validate='one_to_one')
 cr=[dict(group='pooled_4years',n=len(m),r=m.D.corr(m.curtail_rate))]
 for y,t in m.groupby('year'):cr.append(dict(group=str(y),n=len(t),r=t.D.corr(t.curtail_rate)))
 dm=m[['D','curtail_rate']]-m.groupby('year')[['D','curtail_rate']].transform('mean')
 cr.append(dict(group='year_demeaned',n=len(dm),r=dm.D.corr(dm.curtail_rate)))
 pd.DataFrame(cr).to_csv(HERE/'report/correlation_structure.csv',index=False)
 inp=V37/'inputs/raw/dea_el_dh/technology_data_el_dh_2026-08.xlsx'
 import openpyxl
 w=openpyxl.load_workbook(inp,data_only=True,read_only=True);s=w['06 Gas engines, biogas'];heat=[]
 for col in 'CDEF':
  ee=s[f'{col}9'].value;cb=s[f'{col}10'].value
  heat.append(dict(catalog_year=s[f'{col}3'].value,annual_net_electrical_efficiency=ee,Cb_power_to_heat=cb,
     derived_heat_efficiency=ee/cb,source_sheet=s.title,electricity_cell=f'{col}9',Cb_cell=f'{col}10'))
 pd.DataFrame(heat).to_csv(HERE/'report/DEA_heat_parameter_audit.csv',index=False)
 dump(HERE/'report/DEA_heat_parameter_provenance.json',dict(source=relative(inp),sha256=sha(inp),
    primary_v37_eta_th=.488,derivation='2015 C9/C10, rounded; not a directly observed 2035 heat efficiency',
    h_definition='Net externally useful heat / catalog potential heat, after internal heat and heat delivery losses'))
 b=pd.read_csv(HERE/'report/boundary_year_summary.csv');rows=[]
 gas=.243023076
 for t in b.itertuples():
  for eth in [.40,.45,.488,.50,.55]:
   heat=gas*eth/.9
   rows.append(dict(year=t.year,thermal_efficiency=eth,net_zero_heat_fraction=-t.net_primary/heat,
      heat_fraction_to_match_injection=(t.injection_net_primary-t.net_primary)/heat,
      status='declared engineering range; fixed electrical dispatch, not a CHP operating curve'))
 pd.DataFrame(rows).to_csv(HERE/'report/CHP_thermal_efficiency_sensitivity.csv',index=False)
 # Methane leakage differences, separate from the main equal-leakage comparison.
 leak=[]
 for t in b.itertuples():
  for l1 in [.010,.025,.042]:
   for li in [.010,.025,.042]:
    net1=t.avoided_domestic-l1*.072*27
    neti=gas-.03*t.MEF_domestic-li*.072*27
    leak.append(dict(year=t.year,site_leakage=l1,injection_leakage=li,site_net=net1,injection_net=neti,difference=neti-net1,
         h_match_injection=(neti-net1)/(.488*gas/.9)))
 pd.DataFrame(leak).to_csv(HERE/'report/asymmetric_leakage_scenarios.csv',index=False)
 print(json.dumps(dict(copied_source_tables=len(mapping),correlations=cr,thermal_efficiency_rows=len(rows),asymmetric_leakage_rows=len(leak))),flush=True)
if __name__=='__main__':main()
