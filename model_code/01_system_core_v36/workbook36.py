"""Write a numerical workbook from exported CSVs; raw hourly arrays remain in NPZ."""
import csv,json,math,os
from pathlib import Path
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill
from common36 import HERE,dump,now,sha


TABLES=[('Cases','report/dispatch_case_metrics.csv'),('CountryWeather','report/country_use_by_weather.csv'),
 ('CountryMeans','report/country_use_means.csv'),('UseLedger','report/all_use_primary_numeric_ledger.csv'),
 ('AccountingSensitivity','report/all_use_accounting_sensitivities.csv'),('MarginalWeather','report/country_marginal_factors_by_weather.csv'),
 ('PowerAssets','report/power_asset_summary.csv'),
 ('GasEfficiencyResponse','report/fixed_dispatch_gas_efficiency_sensitivity.csv'),
 ('MarginalMeans','report/country_marginal_factors_mean.csv'),('PolicyGrid','report/policy_price_response_grid.csv'),
 ('NonpowerGrid','report/nonpower_response_grid.csv'),('FrontierWeather','report/frontier_by_weather.csv'),
 ('FrontierGrid','report/frontier_weighted.csv'),('FrontierThresholds','report/frontier_threshold_coordinates.csv'),
 ('WasteFactors','report/waste_treatment_factor_grid.csv'),('WasteResource','report/waste_counterfactual_resource_addends.csv'),
 ('U3AbsorptionWeather','report/U3_absorption_by_weather.csv'),('U3AbsorptionSensitivity','report/U3_absorption_capped_accounting.csv'),
 ('ELCCSearch','report/ELCC_search_points.csv'),('ELCCAllocated','report/ELCC_allocated_country_capacity.csv'),
 ('ResourceHistory','report/resource_country_year_all_stages.csv'),('ResourceLevers','report/resource_lever_factors.csv'),
 ('ResourceSingleFactor','report/resource_single_factor_outputs.csv'),('CollectionResponse','report/resource_collection_response_grid.csv'),
 ('UKResource','inputs/uk_resource_v36.csv'),('GasBalances','inputs/annual_sector_gas_2014_2023.csv'),
 ('ImportDependency','inputs/annual_import_dependency_2014_2023.csv'),('CountryVOLL','inputs/country_voll_acer2023.csv'),('ThermalParameters','inputs/thermal_parameters.csv'),
 ('SourceCells','inputs/parameter_source_cells.csv'),('WasteSourceCells','inputs/waste_parameter_source_cells.csv'),('VehicleSourceCells','inputs/vehicle_parameter_source_cells.csv')]


def value(s):
    if s in ['', 'nan','NaN']:return None
    if s in ['True','False']:return s=='True'
    try:
        x=float(s)
        if not math.isfinite(x):return None
        return int(x) if x.is_integer() and abs(x)<1e15 else x
    except ValueError:return s


def export_workbook(partial=False):
    out=HERE/'report'/('V36_numeric_results_partial.xlsx' if partial else 'V36_numeric_results.xlsx')
    wb=Workbook(write_only=True);ws=wb.create_sheet('Readme')
    ws.append(['V36 numerical experiment outputs'])
    ws.append(['created_utc',now()]);ws.append(['partial',partial])
    ws.append(['completion_rule','Only report/computation_complete.json with success=true indicates final calculation and QA completion.'])
    ws.append(['hourly_raw_arrays','results/*.npz; SHA256 and receipt locations in report/result_array_index.json'])
    ws.append(['metadata','report/HANDOVER_FOR_ANALYSIS_zh.md; no hypothesis verdicts or manuscript included'])
    index=[]
    for title,file in TABLES:
        path=HERE/file
        if not path.exists() or path.stat().st_size<3:continue
        with path.open(newline='',encoding='utf-8-sig') as stream:
            reader=csv.reader(stream);header=next(reader,None)
            if not header:continue
            sheet=wb.create_sheet(title);sheet.freeze_panes='A2';sheet.sheet_view.showGridLines=False
            styled=[]
            for text in header:
                cell=WriteOnlyCell(sheet,text);cell.font=Font(name='Arial',size=10,bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='264653');styled.append(cell)
            sheet.append(styled);count=0
            for row in reader:sheet.append([value(v) for v in row]);count+=1
            sheet.auto_filter.ref=f'A1:{__import__("openpyxl").utils.get_column_letter(len(header))}{count+1}'
            index.append(dict(sheet=title,source=file,rows=count,columns=len(header),source_sha256=sha(path)))
    elcc=wb.create_sheet('ELCC');elcc.append(['use','capacity_MW','mean_reduction_MWh','mode','target_mean_reduction_MWh','converged'])
    for use in ['R1','R2']:
        path=HERE/'report'/f'ELCC_{use}.json'
        if path.exists():
            state=json.loads(path.read_text())
            for point in state.get('evaluations',[]):elcc.append([use,point['capacity_MW'],point['mean_reduction_MWh'],point['mode'],state['target_mean_reduction_MWh'],state.get('converged')])
    temp=out.with_name(out.stem+f'.{os.getpid()}.tmp.xlsx');wb.save(temp);temp.replace(out)
    verify=load_workbook(out,read_only=True,data_only=True)
    for entry in index:
        # Streaming worksheets may omit dimension metadata; count actual rows.
        assert sum(1 for _ in verify[entry['sheet']].iter_rows())==entry['rows']+1,entry['sheet']
    verify.close()
    dump(HERE/'qa'/('workbook_partial_checks.json' if partial else 'workbook_checks.json'),dict(passed=True,partial=partial,created_utc=now(),workbook=str(out.relative_to(HERE)),sha256=sha(out),table_index=index))
    return out
