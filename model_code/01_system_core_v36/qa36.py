"""Independent output-contract checks and a machine-readable completion receipt."""
from pathlib import Path
import sys,json,argparse,zipfile,os
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,ROOT,V32,now,dump,sha,load_input


def check_package(partial=False):
    manifest=json.loads((HERE/'protocol/jobs_v36.json').read_text());checks=[];successful=0;stages=0;profiles={};splits={}
    def check(name,ok,**detail):checks.append(dict(name=name,passed=bool(ok),**detail))
    for name in ['model_extension_tests.json','thermal_split_reconciliation.json','reproduction_extended_main_eval_rule_eens_WS13_checkpoint_v2.json','external_data_checks.json','nonpower_algebra_checks.json','waste_counterfactual_checks.json','protocol_history_binding.json','postprocessing_known_answer_tests.json','resource_response_checks.json','voll_source_recheck.json','fuel_efficiency_revaluation_checks.json']:
        r=json.loads((HERE/'qa'/name).read_text());check('prerequisite '+name,r.get('passed') is True)
    for j in manifest['jobs']:
        name=j['name'];pj=HERE/'results'/(name+'.json');pn=HERE/'results'/(name+'.npz')
        if not pj.exists():
            if not partial:check('case present '+name,False)
            continue
        r=json.loads(pj.read_text());check('case success '+name,r.get('success') is True)
        if not r.get('success'):continue
        successful+=1;stages+=r['stages_saved']
        check('physical certificate '+name,r.get('physical_verified') is True)
        check('NPZ binding '+name,pn.exists() and sha(pn)==r['arrays_sha256'])
        registry=HERE/'protocol/registration_history'/(r['registry_sha256']+'.json')
        check('registered protocol binding '+name,registry.exists() and sha(registry)==r['registry_sha256'])
        with np.load(pn,allow_pickle=False) as a:
            if j['weather'] not in splits:
                with np.load(V32/'inputs'/f"2035_WS{j['weather']:02d}.npz",allow_pickle=False) as original:splits[j['weather']]=str(original['gas_split'])
            profile_key=(j.get('scenario','Central'),splits[j['weather']])
            if profile_key not in profiles:profiles[profile_key]=load_input(j['weather'],j.get('scenario','Central'),'R0')['resource_q_hourly']
            check('gas profile source regeneration '+name,np.max(abs(a['resource_q_hourly']-profiles[profile_key]))<1e-8)
            ens=float(a['unserved'].sum());check('EENS array sum '+name,abs(ens-r['EENS_MWh'])<2e-4)
            check('negative unserved '+name,a['unserved'].min()>=-2e-4)
            check('finite numerical arrays '+name,all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fc'))
            if r['stages_saved']==2:
                check('two-stage EENS cap '+name,r['EENS_MWh']<=r['minimum_EENS_MWh']+1e-4+1e-6)
                s1=HERE/'results'/(name+'_s1.npz');s1j=HERE/'results'/(name+'_s1.json')
                check('first stage saved '+name,s1.exists() and s1j.exists())
                if s1j.exists():
                    first=json.loads(s1j.read_text());check('first stage optimal value '+name,abs(first['EENS_MWh']-r['minimum_EENS_MWh'])<=1e-6)
                    check('first stage NPZ binding '+name,s1.exists() and sha(s1)==first.get('arrays_sha256'))
                    check('first stage physical certificate '+name,first.get('physical_verified') is True)
            if j['batch'] in ['R2','R5']:
                B=a['resource_q_hourly'].sum(axis=0);base=np.array(r['base_generation_MWh_by_zone'])/.4
                consumed=a['boost'].sum(axis=0)/.4+a['other_gas_use'].sum(axis=0)+a['gas_charge'].sum(axis=0)*.03/.4+base
                check('annual gas conservation '+name,abs(B-consumed).max()<=2e-4)
                stock=a['gas_inventory'];charge=a['gas_charge'];withdraw=a['gas_withdraw']
                check('hourly gas conservation '+name,abs(stock-np.roll(stock,1,axis=0)-charge+withdraw).max()<=2e-4)
            if j['batch']=='R0':
                check('no R0 bio boost '+name,abs(a['boost']).max()<=2e-4)
                check('nodal duals retained '+name,'nodal_dual' in a)
    if not partial:
        check('all registered physical cases complete',successful==len(manifest['jobs']))
        check('R5 full 192-case grid',sum(j['batch']=='R5' for j in manifest['jobs'])==192)
        for use in ['R1','R2']:
            p=HERE/'report'/f'ELCC_{use}.json';check('ELCC receipt '+use,p.exists())
            if p.exists():
                s=json.loads(p.read_text());check('ELCC convergence '+use,s.get('converged',s.get('target_mean_reduction_MWh',1)<=1e-8))
        for name in ['country_use_by_weather.csv','country_use_means.csv','all_use_primary_numeric_ledger.csv','all_use_accounting_sensitivities.csv','nonpower_response_grid.csv','policy_price_response_grid.csv','frontier_weighted.csv','frontier_interpolated_surface.csv','frontier_threshold_coordinates.csv','waste_treatment_factor_grid.csv','waste_counterfactual_resource_addends.csv','U3_absorption_capped_accounting.csv','ELCC_search_points.csv','ELCC_allocated_country_capacity.csv','resource_single_factor_outputs.csv','resource_collection_response_grid.csv']:
            p=HERE/'report'/name;check('table exists '+name,p.exists() and p.stat().st_size>100)
            if p.exists() and p.stat().st_size>100:check('table readable '+name,len(pd.read_csv(p))>0)
        table=pd.read_csv(HERE/'report/all_use_primary_numeric_ledger.csv')
        check('primary ledger 15 countries x 3 resources x 7 end-use arms',len(table)==315 and table.country.nunique()==15 and table.resource_scenario.nunique()==3 and table.use.nunique()==7)
        groups=pd.read_csv(HERE/'report/country_use_means.csv');check('all dispatch weather groups complete',(groups.weather_count==36).all())
        frontier=pd.read_csv(HERE/'report/frontier_weighted.csv');check('all frontier grid cells represented',len(frontier)==24 and (frontier.sampled_weathers==8).all())
        interp=json.loads((HERE/'report/frontier_interpolation_metadata.json').read_text());check('frontier interpolation complete',interp.get('partial') is False)
        workbook=json.loads((HERE/'qa/workbook_checks.json').read_text());wp=HERE/workbook['workbook']
        check('numeric workbook integrity',workbook.get('passed') is True and wp.exists() and sha(wp)==workbook['sha256'])
    capfile=HERE/'report/U3_absorption_by_weather.csv'
    if capfile.exists():
        cap=pd.read_csv(capfile)
        check('U3 annual absorption within available gas and biomethane',((cap.U3_country_annual_absorbable_MWh<=cap.U3_power_plant_gas_fuel_MWh+1e-6)&(cap.U3_country_annual_absorbable_MWh<=cap.biomethane_MWh+1e-6)).all())
        check('U3 nonnegative absorbed gas',(cap.U3_zone_hourly_absorbable_MWh>=0).all() and (cap.U3_country_annual_absorbable_MWh>=0).all())
        check('U3 hourly absorption within annual cap',(cap.U3_zone_hourly_absorbable_MWh<=cap.U3_country_annual_absorbable_MWh+1e-6).all())
    for path,digest in json.loads((HERE/'protocol/source_hashes.json').read_text()).items():check('protected source unchanged '+path,sha(ROOT/path)==digest)
    for binding in json.loads((HERE/'protocol/inherited_accounting_sources.json').read_text())['binding']:
        check('inherited accounting source binding '+binding['original'],sha(ROOT/binding['original'])==binding['sha256'] and sha(HERE/binding['snapshot'])==binding['sha256'])
    binding=json.loads((HERE/'protocol/gas_profile_source_binding.json').read_text())
    check('hourly gas source unchanged and retained',sha(ROOT/binding['original'])==binding['sha256'] and sha(HERE/binding['snapshot'])==binding['sha256'])
    resource_countries=set(pd.read_csv(HERE/'inputs/country_resource_2020_2023_mean.csv').country_code)
    check('VOLL coverage follows actual model countries',set(pd.read_csv(HERE/'inputs/country_voll_acer2023.csv').country)==resource_countries)
    for name in ['annual_sector_gas_2014_2023.csv','annual_import_dependency_2014_2023.csv']:
        table=pd.read_csv(HERE/'inputs'/name)
        check('external table country/year coverage '+name,set(table.country)==resource_countries and len(table[['country','year']].drop_duplicates())==150 and set(table.year)==set(range(2014,2024)))
    failures=[v for v in checks if not v['passed']]
    result=dict(passed=not failures,partial=partial,completed_utc=now(),registered_physical_cases=len(manifest['jobs']),successful_cases=successful,LP_stages_saved=stages,checks=len(checks),failures=failures,check_details=checks,interpretation_produced=False)
    dump(HERE/'qa'/('partial_output_checks.json' if partial else 'completion_receipt.json'),result)
    if not failures and not partial:
        # Compact hand-off archive; large raw NPZs remain indexed alongside it.
        archive=HERE/'report/V36_numeric_handoff.zip'
        temporary=archive.with_name(archive.stem+f'.{os.getpid()}.tmp.zip')
        members=[p for folder in ['scripts','protocol','report','qa','inputs','results'] for p in (HERE/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.xlsx'] and p!=archive and not p.name.endswith('_partial.xlsx') and p.name not in ['computation_complete.json','progress.json','supervisor_status.json']]
        members.append(HERE/'README_zh.md')
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in members:z.write(path,arcname=str(path.relative_to(HERE)))
            z.writestr('report/computation_complete.json',json.dumps(dict(success=True,completed_utc=now(),successful_cases=successful,LP_stages_saved=stages,data_QA_passed=True,raw_arrays_index='report/result_array_index.json',archive_self_hash_in_external_project_receipt=True,no_hypothesis_interpretation_or_manuscript=True),indent=1)+'\n')
        with zipfile.ZipFile(temporary) as z:check('archive integrity',z.testzip() is None)
        result['checks']=len(checks);result['failures']=[v for v in checks if not v['passed']];result['passed']=not result['failures']
        dump(HERE/'qa/completion_receipt.json',result)
        if not result['passed']:raise RuntimeError('Handoff archive integrity failed')
        temporary.replace(archive)
        dump(HERE/'report/computation_complete.json',dict(success=True,completed_utc=now(),successful_cases=successful,LP_stages_saved=stages,numeric_handoff_archive=str(archive.relative_to(HERE)),archive_sha256=sha(archive),raw_arrays_index='report/result_array_index.json',no_hypothesis_interpretation_or_manuscript=True))
    if failures:raise RuntimeError(json.dumps(failures,ensure_ascii=False))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();r=check_package(args.partial);print(json.dumps({k:r[k] for k in ['passed','partial','successful_cases','checks']}))
