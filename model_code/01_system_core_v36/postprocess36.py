"""Export numerical experiment tables and source arrays without interpreting H1–H4."""
from pathlib import Path
import sys,json,argparse,math
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,V32,now,dump,sha,arrays,load_input,resource_table
from nonpower import assumptions,calculate,gas_upstream

FOSSIL=['gas','Lignite','Hard coal','Light oil','Heavy oil']
UPSTREAM={'gas':None,'Lignite':.010,'Hard coal':.038,'Light oil':.07884,'Heavy oil':.02052}


def fuel_arrays(a,d):
    out={'gas':a['gas']/d['gas_efficiency'][None,:]}
    for tech in ['Nuclear','Biofuel','Lignite','Hard coal','Light oil','Heavy oil']:
        value=np.zeros_like(a['gas'])
        for k,(z,t) in enumerate(zip(d['thermal_zone'],d['thermal_technology'])):
            if t==tech:value[:,z]+=a['thermal_dispatch'][:,k]/d['thermal_efficiency'][k]
        out[tech]=value
    return out


def factors(d,p,horizon='GWP100'):
    ef={'gas':p['gas_combustion_t_per_MWh']+gas_upstream(p,horizon)}
    for tech in FOSSIL[1:]:
        ids=np.flatnonzero(d['thermal_technology']==tech)
        combustion=float(d['thermal_combustion_ef'][ids[0]]) if len(ids) else {'Lignite':.3636,'Hard coal':.34056,'Light oil':.26676,'Heavy oil':.27864}[tech]
        ef[tech]=combustion+UPSTREAM[tech]
    return ef


def marginal(a,d,p):
    """Price-matched interior-generator witnesses; unresolved values stay missing."""
    dual=a['nodal_dual'];T,Z=dual.shape;code=np.zeros((T,Z),dtype=np.int16);ef=np.full((T,Z),np.nan);fu=np.full((T,Z),np.nan);count=np.zeros((T,Z),dtype=np.int16)
    values=factors(d,p);labels={0:'unresolved/storage/network shadow',1:'gas',2:'Lignite',3:'Hard coal',4:'Light oil',5:'Heavy oil',6:'Nuclear',7:'Biofuel',8:'external_import',9:'hydrogen',10:'curtailment',11:'shortage'}
    def add(mask,k,carbon,fuel):
        nonlocal code,ef,fu,count
        mask=mask&(code!=k)
        new=mask&(count==0);code[new]=k;ef[new]=carbon;fu[new]=fuel;count[mask]+=1
    gascap=d['gas_cap']*(1-.3*d['event'][:,None]);interior=(a['gas']>1e-5)&(a['gas']<gascap-1e-5)
    for cost in np.unique(d['gas_cost']):
        cols=np.isclose(d['gas_cost'],cost);witness=interior[:,cols].any(axis=1)
        add(witness[:,None]&(abs(dual-cost)<.005),1,values['gas']/.5,2.)
    for tech,k in [(x,y) for y,x in labels.items() if x in ['Lignite','Hard coal','Light oil','Heavy oil','Nuclear','Biofuel']]:
        for cost in np.unique(d['thermal_cost'][d['thermal_technology']==tech]):
            ix=(d['thermal_technology']==tech)&np.isclose(d['thermal_cost'],cost)
            active=((a['thermal_dispatch'][:,ix]>1e-5)&(a['thermal_dispatch'][:,ix]<d['thermal_cap'][:,ix]-1e-5)).any(axis=1)
            eta=float(d['thermal_efficiency'][np.flatnonzero(ix)[0]])
            add(active[:,None]&(abs(dual-cost)<.005),k,values.get(tech,0)/eta,(1/eta if tech in FOSSIL else 0.))
    active=((a['external_import']>1e-5)&(a['external_import']<d['external_cap']*.5-1e-5)).any(axis=1)
    add(active[:,None]&(abs(dual-100)<.005),8,np.nan,np.nan)
    hc=np.broadcast_to(d.get('hydrogen_cap',0.),(T,Z));hp=np.broadcast_to(d.get('hydrogen_cost_eur_per_MWh',154.496),(T,Z))
    hi=(a['hydrogen']>1e-5)&(a['hydrogen']<hc-1e-5)
    for z in range(Z):add(hi[:,z,None]&(abs(dual-hp[:,z,None])<.005),9,0.,0.)
    # Existing curtailment proves that extra local load can use otherwise-spilled renewable output.
    add((a['curtailment']>1e-5)&(abs(dual)<.005),10,0.,0.)
    multiple=count>1;code[multiple]=0;ef[multiple]=np.nan;fu[multiple]=np.nan
    shortage=a['unserved']>.001;code[shortage]=11;ef[shortage]=np.nan;fu[shortage]=np.nan
    ef20=ef.copy();gas=code==1
    ef20[gas]+=(gas_upstream(p,'GWP20')-gas_upstream(p,'GWP100'))/.5
    return dict(nodal_dual=dual,source_code=code,source_candidate_count=count,marginal_fossil_ef=ef,marginal_fossil_ef_GWP20=ef20,marginal_fossil_fuel=fu),labels


def main(partial=False):
    manifest=json.loads((HERE/'protocol/jobs_v36.json').read_text());jobs=manifest['jobs'];success={}
    for j in jobs:
        path=HERE/'results'/(j['name']+'.json')
        if path.exists():
            r=json.loads(path.read_text())
            if r.get('success') and r.get('physical_verified'):success[j['name']]=(j,r)
    if not partial and len(success)!=len(jobs):raise RuntimeError('Registered dispatch jobs incomplete')
    p=assumptions();metrics=[];country=[];marginal_rows=[];evidence=[];frontier=[]
    for name,(j,r) in success.items():
        pn=HERE/'results'/(name+'.npz')
        if sha(pn)!=r['arrays_sha256']:raise ValueError('Result array hash mismatch '+name)
        metrics.append(dict(name=name,batch=j['batch'],variant=j.get('variant','primary'),scenario=j.get('scenario','Central'),weather=j['weather'],minimum_EENS_MWh=r['minimum_EENS_MWh'],final_EENS_MWh=r['EENS_MWh'],objective_cost_EUR=r['objective_cost_eur'],cost_optimized=r['stages_saved']==2,solve_seconds=r['solve_seconds'],physical_verified=r['physical_verified'],stages_saved=r['stages_saved'],arrays_file=str(pn.relative_to(HERE)),arrays_sha256=r['arrays_sha256']))
        binding=dict(name=name,arrays_file=str(pn.relative_to(HERE)),arrays_sha256=r['arrays_sha256'],receipt_file='results/'+name+'.json',receipt_sha256=sha(HERE/'results'/(name+'.json')))
        first=HERE/'results'/(name+'_s1.npz');first_json=HERE/'results'/(name+'_s1.json')
        if first.exists() and first_json.exists():binding.update(first_stage_arrays_file=str(first.relative_to(HERE)),first_stage_arrays_sha256=sha(first),first_stage_receipt_file=str(first_json.relative_to(HERE)),first_stage_receipt_sha256=sha(first_json))
        evidence.append(binding)
        if j['batch']=='R5':
            base=success.get(f"R0_WS{j['weather']:02d}")
            if base:
                be=base[1]['minimum_EENS_MWh'];frontier.append(dict(name=name,weather=j['weather'],beta=j['beta'],power_GW=j['power_GW'],buffer_h=j['buffer_h'],weight=j['stratum_weight'],baseline_EENS_MWh=be,reduction_MWh=be-r['minimum_EENS_MWh'],reduction_fraction=(be-r['minimum_EENS_MWh'])/be))
    dump(HERE/'report/result_array_index.json',evidence)
    for w in range(1,37):
        if f'R0_WS{w:02d}' not in success:continue
        base=success[f'R0_WS{w:02d}'][1];ba=arrays(HERE/'results'/f'R0_WS{w:02d}.npz');d=load_input(w,'Central','R0');bf=fuel_arrays(ba,d)
        ma,labels=marginal(ba,d,p);np.savez_compressed(HERE/'report'/f'marginal_sources_WS{w:02d}.npz',**ma,zones=d['zones'],countries=d['countries'],hour_index=np.arange(8760),event_H72=d['event'])
        for ct in np.unique(d['countries']):
            ix=d['countries']==ct;weight=ba['resource_q_hourly'][:,ix];valid=np.isfinite(ma['marginal_fossil_ef'][:,ix]);den=float(weight[valid].sum())
            marginal_rows.append(dict(country=str(ct),weather=w,identified_biomethane_weight_fraction=den/float(weight.sum()),gas_source_fraction_identified_weight=float(weight[valid&(ma['source_code'][:,ix]==1)].sum()/den) if den else np.nan,marginal_fossil_ef_identified_hours=float((ma['marginal_fossil_ef'][:,ix][valid]*weight[valid]).sum()/den) if den else np.nan,marginal_fossil_ef_GWP20_identified_hours=float((ma['marginal_fossil_ef_GWP20'][:,ix][valid]*weight[valid]).sum()/den) if den else np.nan,marginal_fossil_fuel_identified_hours=float((ma['marginal_fossil_fuel'][:,ix][valid]*weight[valid]).sum()/den) if den else np.nan,point_estimate_scope='conditional on uniquely identified generator/curtailment witness; unresolved hours retained in NPZ'))
        for name,(j,r) in success.items():
            if j['weather']!=w or j['batch'] not in ['R1','R2','R3']:continue
            a=arrays(HERE/'results'/(name+'.npz'));df=fuel_arrays(a,d);delta={k:bf[k]-df[k] for k in FOSSIL}
            for ct in np.unique(d['countries']):
                ix=d['countries']==ct;B=float(np.asarray(r['resource_MWh_by_zone'])[ix].sum());direct=B if j['batch']=='R3' else 0.
                row=dict(name=name,country=str(ct),weather=w,use={'R1':'U1','R2':'U2','R3':'U3'}[j['batch']],variant=j.get('variant','primary'),site_efficiency=j.get('site_efficiency',.4),injection_efficiency=j.get('gas_efficiency',.5),scenario=j['scenario'],biomethane_MWh=B,direct_grid_gas_substitution_MWh=direct,dispatch_fossil_fuel_displacement_MWh=sum(float(v[:,ix].sum()) for v in delta.values()),delta_EENS_at_dispatch_location_MWh=float((ba['unserved'][:,ix]-a['unserved'][:,ix]).sum()),allocation_basis='Country of changed dispatch; concurrent 15-country deployment, not causal resource-origin attribution')
                row['gross_fossil_displacement_MWh']=row['dispatch_fossil_fuel_displacement_MWh']+direct;row['displacement_MWh_per_MWh']=row['gross_fossil_displacement_MWh']/B
                for k,v in delta.items():row['delta_'+k.replace(' ','_')+'_fuel_MWh']=float(v[:,ix].sum())
                for horizon in ['GWP100','GWP20']:
                    ef=factors(d,p,horizon);row['avoided_fossil_emissions_'+horizon+'_t']=sum(float(v[:,ix].sum())*ef[k] for k,v in delta.items())+direct*ef['gas']
                combustion={'gas':.20196,'Lignite':.3636,'Hard coal':.34056,'Light oil':.26676,'Heavy oil':.27864}
                row['avoided_combustion_emissions_t']=sum(float(v[:,ix].sum())*combustion[k] for k,v in delta.items())+direct*.20196
                row['boost_MWh']=float(a['boost'][:,ix].sum())
                row['compression_electricity_MWh']=float(a.get('gas_charge',np.zeros_like(a['gas']))[:,ix].sum())*.03
                row['hydrogen_generation_change_MWh']=float((a['hydrogen'][:,ix]-ba['hydrogen'][:,ix]).sum())
                row['external_import_change_MWh']=float((a['external_import'][:,ix]-ba['external_import'][:,ix]).sum())
                row['curtailment_change_MWh']=float((a['curtailment'][:,ix]-ba['curtailment'][:,ix]).sum())
                row['biomethane_diverted_MWh']=float(a.get('other_gas_use',np.zeros_like(a['gas']))[:,ix].sum())
                if j['batch']=='R3':
                    row['allocation_basis']='Direct substitution credited to biomethane producer under unconstrained gas-grid absorption; changed dispatch assigned to generation location; capped ledgers supplied separately'
                    gas_fuel=np.maximum(df['gas'][:,ix],0.);row['U3_power_plant_gas_fuel_raw_MWh']=float(df['gas'][:,ix].sum());row['U3_power_plant_gas_fuel_MWh']=float(gas_fuel.sum())
                    row['U3_country_annual_absorbable_MWh']=min(B,float(gas_fuel.sum()))
                    row['U3_zone_hourly_absorbable_MWh']=float(np.minimum(a['resource_q_hourly'][:,ix],gas_fuel).sum())
                    row['U3_excess_over_country_annual_gas_MWh']=max(0.,B-float(gas_fuel.sum()))
                    row['U3_direct_substitution_scope']='Registered unconstrained 1:1 assumption; separate annual-country and hourly-zone absorption-capped ledgers exported'
                country.append(row)
    pd.DataFrame(metrics).to_csv(HERE/'report/dispatch_case_metrics.csv',index=False)
    pd.DataFrame(country).to_csv(HERE/'report/country_use_by_weather.csv',index=False)
    pd.DataFrame(marginal_rows).to_csv(HERE/'report/country_marginal_factors_by_weather.csv',index=False)
    dump(HERE/'report/marginal_source_codes.json',dict(codes=labels if marginal_rows else {},method='Price matching with an interior dispatch witness and .005 EUR/MWh network-price allowance; multiple/unknown/storage/external cases are not force-assigned.',raw_duals_preserved=True))
    if country:
        cf=pd.DataFrame(country);keys=['country','use','variant','site_efficiency','injection_efficiency','scenario'];n=cf.groupby(keys).weather.nunique()
        means=cf.drop(columns='weather').groupby(keys).mean(numeric_only=True).reset_index();means['weather_count']=means.set_index(keys).index.map(n);means.to_csv(HERE/'report/country_use_means.csv',index=False)
    if frontier:
        ff=pd.DataFrame(frontier);ff.to_csv(HERE/'report/frontier_by_weather.csv',index=False)
        out=[]
        exact_baseline=sum(success[f'R0_WS{w:02d}'][1]['minimum_EENS_MWh'] for w in range(1,37))/36
        for (beta,power,buffer),part in ff.groupby(['beta','power_GW','buffer_h']):
            numerator=float((part.reduction_MWh*part.weight).sum()/36);den=float((part.baseline_EENS_MWh*part.weight).sum()/36)
            out.append(dict(beta=beta,power_GW=power,buffer_h=buffer,sampled_weathers=len(part),weighted_mean_reduction_MWh=numerator,weighted_sample_mean_baseline_EENS_MWh=den,exact_36_weather_mean_baseline_EENS_MWh=exact_baseline,sampled_to_exact_baseline_ratio=den/exact_baseline,reduction_fraction=numerator/exact_baseline))
        frontier_means=pd.DataFrame(out)
        frontier_means.to_csv(HERE/'report/frontier_weighted.csv',index=False)
        from frontier_export36 import export_frontier
        export_frontier(frontier_means)
    # Numerical accounting continues in a separate, independently rerunnable module.
    from accounting36 import finish
    finish(pd.DataFrame(country),pd.DataFrame(marginal_rows),partial)
    from absorption36 import export as export_absorption
    export_absorption(pd.DataFrame(country),pd.DataFrame(marginal_rows),partial)
    from fuel_efficiency36 import export as export_efficiency
    export_efficiency(pd.DataFrame(country),pd.DataFrame(marginal_rows),partial)
    from capacity_export36 import export as export_capacity
    export_capacity()
    dump(HERE/'qa'/('postprocessing_code_partial_manifest.json' if partial else 'postprocessing_code_manifest.json'),dict(created_utc=now(),partial=partial,script_sha256={str(v.relative_to(HERE)):sha(v) for v in sorted((HERE/'scripts').glob('*.py'))},input_parameter_sha256={str(v.relative_to(HERE)):sha(v) for v in sorted((HERE/'inputs').iterdir()) if v.is_file() and v.suffix in ['.json','.csv','.npz']}))
    from workbook36 import export_workbook
    export_workbook(partial)
    from qa36 import check_package
    qa=check_package(partial)
    if not partial:
        progress=json.loads((HERE/'report/progress.json').read_text())
        progress.update(updated_utc=now(),state='complete',accounting_and_QA_complete=True,completion_receipt='qa/completion_receipt.json',successful=qa['successful_cases'],LP_stages_saved=qa['LP_stages_saved'])
        dump(HERE/'report/progress.json',progress)
    print(json.dumps(dict(completed_cases=len(success),registered=len(jobs),partial=partial,interpretation_produced=False)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();main(args.partial)
