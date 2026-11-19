"""Post-review accounting audit: re-read every V37 F0/F1 pair; no LP or old-file writes.

Separate domestic dispatch emissions from external product credits. Hydrogen energy
basis is not verified in the ERAA CSV; report both explicit LHV/HHV interpretations.
Import intensities and heat destinations are scenarios, not measured marginal factors.
"""
import sys,itertools,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *
import accounting37 as ac
import nonpower

OUT=HERE/'report'; QA=HERE/'qa'
LEAKAGES=[.010,.025,.042]; H=[0.,.25,.5,1.]
IEA_H2=[0.,10.,11.,12.] # kg CO2eq per kg unabated natural-gas hydrogen; GWP100 only.
IMPORT=[0.,.1,.3,.5] # t CO2eq/MWh external electricity, declared scenarios.
H2_KWH_KG={'LHV_assumption':33.33,'HHV_assumption':39.41}
P=ac.P36;RHO=P['methane_mass_t_per_MWh'];GWP=P['GWP100']['bio_CH4']
ETA_TH=.488;ETA_BOILER=.90
GAS=ac.EF['GWP100']['gas'];HEAT=GAS*ETA_TH/ETA_BOILER

def main():
 OUT.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
 src=V37/'report/preliminary_F1_by_weather.csv'; old=pd.read_csv(src)
 assert len(old)==144 and not old.duplicated(['year','weather']).any()
 rows=[];provenance={relative(src):sha(src),relative(Path(__file__)):sha(Path(__file__))};checks=[]
 for t in old.itertuples():
  paths=[V37/'results'/f'F1_{t.year}_U1_Central_WS{t.weather:02d}.npz',V37/'results'/f'F0_{t.year}_R0_WS{t.weather:02d}.npz']
  aa=[]
  for p in paths:
   r=json.loads(p.with_suffix('.json').read_text());hs=sha(p)
   assert r.get('success') and r.get('physical_verified') and r['stages_saved']==2 and r['arrays_sha256']==hs
   provenance[relative(p)]=hs;provenance[relative(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
   aa.append(arrays(p))
  d=ac.pair_metrics(*aa);f=ac.finish_u1(d);B=d['biomethane_MWh']
  # Existing CSV response columns, including h2, use added electricity as denominator.
  chk=dict(year=int(t.year),weather=int(t.weather),D_error=f['D_fossil_per_MWh']-t.D,
    avoided_error=f['avoided_GWP100_t_per_MWh']-t.avoided,
    H2_conversion_error=d['extra_hydrogen_MWh']/B-.4*t.h2,
    import_conversion_error=d['balance_reduced_external_import_MWh']/B-.4*t.reduced_external_import,
    balance_relative=f['balance_residual_relative'])
  assert max(abs(v) for k,v in chk.items() if k not in ('year','weather'))<1e-8
  checks.append(chk)
  rows.append(dict(year=int(t.year),weather=int(t.weather),B_MWh=B,D=f['D_fossil_per_MWh'],
     avoided_domestic=f['avoided_GWP100_t_per_MWh'],avoided20=f['avoided_GWP20_t_per_MWh'],
     extra_H2_reported_MWh_per_MWh=d['extra_hydrogen_MWh']/B,
     avoided_external_import_MWhe_per_MWh=d['balance_reduced_external_import_MWh']/B,
     extra_P2H_MWhe_per_MWh=d['balance_increased_power_to_heat_consumption_MWh']/B,
     reduced_H2_power_MWhe_per_MWh=d['balance_reduced_hydrogen_MWh']/B))
  if t.weather==36:print('verified year',t.year,flush=True)
 b=pd.DataFrame(rows);b['MEF_domestic']=b.avoided_domestic/.4
 b['net_primary']=b.avoided_domestic-.025*RHO*GWP
 b['injection_net_primary']=GAS-.03*b.MEF_domestic-.025*RHO*GWP
 b['h_zero_primary']=-b.net_primary/HEAT
 b['h_match_injection_primary']=(b.injection_net_primary-b.net_primary)/HEAT
 b['lambda_zero_primary']=b.avoided_domestic/(RHO*GWP)
 b['import_EF_zero_primary']=-b.net_primary/b.avoided_external_import_MWhe_per_MWh
 b.to_csv(OUT/'boundary_base_by_weather.csv',index=False)
 joint=[]
 for r in b.itertuples():
  for leak,h,efh,efi,basis,proc in itertools.product(LEAKAGES,H,IEA_H2,IMPORT,H2_KWH_KG,['registered_domestic','expanded_symmetric_approximation']):
   hc=r.extra_H2_reported_MWh_per_MWh*efh/H2_KWH_KG[basis]
   ic=r.avoided_external_import_MWhe_per_MWh*efi
   upstream=r.avoided_domestic+hc+ic+h*HEAT
   # The symmetric case assumes infinitesimal compression demand reverses the U1
   # system response. It is an accounting approximation, not a re-solved injection case.
   process_mef=r.MEF_domestic if proc=='registered_domestic' else (r.avoided_domestic+hc+ic)/.4
   inj=GAS-.03*process_mef-leak*RHO*GWP
   cng=nonpower.calculate('U5_CNG_diesel',process_mef,'central',leak,'GWP100',P)['net_abatement_t_per_MWh']
   u=upstream-leak*RHO*GWP
   joint.append(dict(year=r.year,weather=r.weather,leakage=leak,h=h,H2_kgCO2_per_kg=efh,H2_energy_basis=basis,
      import_tCO2_per_MWhe=efi,compression_accounting=proc,H2_credit=hc,import_credit=ic,CHP_heat_credit=h*HEAT,
      process_MEF=process_mef,site_net=u,injection_net=inj,CNG_diesel_net=cng,injection_minus_site=inj-u,
      lambda_zero=upstream/(RHO*GWP),h_match_injection=(inj+leak*RHO*GWP-r.avoided_domestic-hc-ic)/HEAT))
 j=pd.DataFrame(joint);j.to_csv(OUT/'boundary_joint_scenarios.csv',index=False)
 group=['year','leakage','h','H2_kgCO2_per_kg','H2_energy_basis','import_tCO2_per_MWhe','compression_accounting']
 values=['site_net','injection_net','CNG_diesel_net','injection_minus_site','lambda_zero','h_match_injection']
 g=j.groupby(group)[values].agg(['mean','min','max']);g.columns=['_'.join(x) for x in g.columns];g=g.reset_index()
 g.to_csv(OUT/'boundary_joint_summary.csv',index=False)
 # Independent alternatives for the displaced HEAT service; no heat-pump LP is implied.
 heatrows=[]
 for r in b.itertuples():
  for name,ef in [('gas_boiler_90pct',GAS/.9),('zero_carbon_heat',0.),('electric_heat_pump_COP3_domestic_MEF',r.MEF_domestic/3)]:
   for h in H:
    heatrows.append(dict(year=r.year,weather=r.weather,heat_counterfactual=name,h=h,heat_EF_t_per_MWhth=ef,
        exported_heat_MWhth=h*ETA_TH,net=r.net_primary+h*ETA_TH*ef,
        injection_minus_site=r.injection_net_primary-r.net_primary-h*ETA_TH*ef))
 heat=pd.DataFrame(heatrows);heat.to_csv(OUT/'CHP_heat_counterfactuals.csv',index=False)
 # Report P2H and hydrogen-fired generation as additional, unclosed product boundaries.
 other=[]
 for r in b.itertuples():
  for conversion in [1.,3.]:
   other.append(dict(year=r.year,weather=r.weather,P2H_assumed_heat_per_electricity=conversion,
     P2H_assumed_gas_heat_credit=r.extra_P2H_MWhe_per_MWh*conversion*GAS/.9,
     status='illustrative service expansion only: technology, sink and uptake are not identified'))
 oth=pd.DataFrame(other);oth.to_csv(OUT/'P2H_unresolved_boundary_scenarios.csv',index=False)
 # Zero net threshold uses ratios of mean physical flows (not a mean of weather thresholds).
 yr=b.drop(columns='weather').groupby('year').mean(numeric_only=True).reset_index()
 yr['import_EF_zero_primary']=-yr.net_primary/yr.avoided_external_import_MWhe_per_MWh
 for lab,k in H2_KWH_KG.items():
  for ef in [10.,12.]:
   cred=yr.extra_H2_reported_MWh_per_MWh*ef/k
   yr[f'H2_{lab}_{int(ef)}_site_net']=yr.net_primary+cred
   yr[f'H2_{lab}_{int(ef)}_lambda_zero']=(yr.avoided_domestic+cred)/(RHO*GWP)
 yr['net_h025']=yr.net_primary+.25*HEAT
 yr.to_csv(OUT/'boundary_year_summary.csv',index=False)
 # Meaningful invariants: reproduce frozen table, full factor units, credit differences,
 # and show that equal leakage changes cancel in the site/injection difference.
 vc=pd.read_csv(V37/'report/preliminary_value_cost_by_year.csv')
 refs=vc[vc.use=='injection'].set_index('year').net_t
 assert np.allclose(yr.set_index('year').injection_net_primary,refs,atol=1e-10)
 diff=j.groupby([x for x in group if x!='leakage']+['weather']).injection_minus_site.agg(['min','max'])
 assert (diff['max']-diff['min']).abs().max()<1e-12
 assert len(j)==4*36*3*4*4*4*2*2
 # Preserve data provenance, including all source arrays and this analysis source.
 param=V36/'inputs/nonpower_parameters.json';provenance[relative(param)]=sha(param)
 provenance[relative(V37/'scripts/accounting37.py')]=sha(V37/'scripts/accounting37.py')
 provenance[relative(V36/'scripts/nonpower.py')]=sha(V36/'scripts/nonpower.py')
 dump(OUT/'boundary_provenance.json',provenance)
 receipt=dict(passed=True,utc=now(),paired_cases=144,source_arrays=288,joint_scenario_rows=len(j),
    scenario_rows_are_not_new_LP_cases=True,max_reproduction_error=float(pd.DataFrame(checks).drop(columns=['year','weather']).abs().max().max()),
    injection_primary_reproduced=True,equal_leakage_difference_invariant=True,
    limitations=['H2 efficiency energy basis unverified: both explicit interpretations reported','External import intensities are scenarios, not measurements',
    'P2H product destination and H2 turbine upstream fuel are not closed','Expanded compression response is an approximation, not an LP',
    'GWP100 IEA H2 intensity is not recycled into GWP20'],
    constants=dict(gas_chain=GAS,heat_output_eff=ETA_TH,boiler_eff=ETA_BOILER,heat_credit_per_h=HEAT,
      eta_th_source='DEA 06 Gas engines, biogas cells C9/C10 = 0.40/0.82 = 0.487804878; 2015 derived engineering benchmark rounded to 0.488, held constant',
      H2_kgCO2_per_kg_source='IEA Global Hydrogen Review 2024 Chapter 7, 10-12 kg CO2eq/kg for unabated natural gas; not a European marginal H2 observation',
      heat_rounding_difference_from_v37=(GAS-.24302)*ETA_TH/.9))
 dump(QA/'boundary_checks.json',receipt)
 with pd.ExcelWriter(OUT/'V38_boundaries.xlsx',engine='openpyxl') as x:
  yr.to_excel(x,sheet_name='year_summary',index=False);b.to_excel(x,sheet_name='paired_weather',index=False)
  g.to_excel(x,sheet_name='joint_summary',index=False);heat.to_excel(x,sheet_name='heat_counterfactual',index=False)
  oth.to_excel(x,sheet_name='unclosed_P2H',index=False);pd.DataFrame(checks).to_excel(x,sheet_name='reproduction',index=False)
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
