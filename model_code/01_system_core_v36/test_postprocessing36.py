"""Known-answer tests for R5 interpolation and the shared 36-weather ELCC search."""
from pathlib import Path
import sys,tempfile,json
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from common36 import HERE,dump,now
from frontier_export36 import export_frontier,crossings
import adaptive36 as adaptive


def main():
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(dict(name=name,passed=True))
    with tempfile.TemporaryDirectory(prefix='synthetic_postprocessing_',dir=HERE/'qa') as folder:
        out=Path(folder);rows=[]
        for b in [0.,.75]:
            for power in [.26,1.,4.,16.]:
                for storage in [22,168,720]:rows.append(dict(beta=b,power_GW=power,buffer_h=storage,sampled_weathers=8,reduction_fraction=.002*power+.00001*storage))
        export_frontier(pd.DataFrame(rows),out);surface=pd.read_csv(out/'frontier_interpolated_surface.csv');threshold=pd.read_csv(out/'frontier_threshold_coordinates.csv')
        check('bilinear interpolation reproduces analytic physical-coordinate plane',np.max(abs(surface.interpolated_reduction_fraction-(.002*surface.power_GW+.00001*surface.buffer_h)))<1e-12)
        part=threshold[threshold.power_GW.notna()]
        check('threshold roots match analytic plane',np.max(abs(part.power_GW-(part.target_fraction-.00001*part.buffer_h)/.002))<1e-9)
        check('unreachable 5 and 10 percent targets retained',threshold[threshold.target_fraction.isin([.05,.1])].status.eq('below_target_throughout_tested_power_interval').all())
        check('nonmonotone crossing branches preserved',np.allclose(crossings([.26,1,4,16],[0,.1,0,.1],.05),[.63,2.5,10],atol=1e-12))
        # Synthetic receipts live only under this temporary QA folder, never formal results/.
        adaptive.HERE=out
        for sub in ['results','protocol','report']:(out/sub).mkdir(exist_ok=True)
        for w in range(1,37):
            base={1:1000.,2:2000.}.get(w,0.);delta={1:100.,2:150.}.get(w,0.)
            for use,value in [('R0',base),('R1',base-delta),('R2',base-delta)]:
                name=f'R0_WS{w:02d}' if use=='R0' else f'{use}_Central_WS{w:02d}'
                dump(out/'results'/(name+'.json'),dict(success=True,physical_verified=True,minimum_EENS_MWh=value,EENS_MWh=value))
        adaptive.load_input=lambda *a,**k:dict(q=np.array([100.,100.]))
        adaptive.design=lambda *a,**k:dict(power=[10.,10.])
        manifest=dict(jobs=[],amendments=[])
        for _ in range(20):
            adaptive.update_elcc(manifest)
            state=json.loads((out/'protocol/elcc_state.json').read_text())
            if all(v['complete'] for v in state.values()):break
            for j in manifest['jobs']:
                path=out/'results'/(j['name']+'.json')
                if not path.exists():
                    w=j['weather'];base={1:1000.,2:2000.}[w];value=max(0.,base-{1:2.,2:3.}[w]*j['firm_MW'])
                    dump(path,dict(success=True,physical_verified=True,minimum_EENS_MWh=value,EENS_MWh=value))
        check('both global ELCC searches converge to known 50 MW',all(v.get('converged') and abs(v['ELCC_MW']-50)<1e-10 for v in state.values()))
        check('ELCC uses all-weather divisor 36',all(abs(v['target_mean_reduction_MWh']-250/36)<1e-12 for v in state.values()))
        check('ELCC capacity common across weather states',all(len(set(j['firm_MW'] for j in manifest['jobs'] if j['use']==u and j['iteration']==i))==1 for u in ['R1','R2'] for i in range(5)))
        import capacity_export36
        capacity_export36.HERE=out;capacity_export36.load_input=lambda *a,**k:dict(countries=np.array(['AA','BB']))
        capacity_export36.export();allocated=pd.read_csv(out/'report/ELCC_allocated_country_capacity.csv')
        check('country allocations conserve the known system ELCC',len(allocated)==4 and np.allclose(allocated.groupby('use').allocated_system_equivalent_firm_capacity_MW.sum(),50))
    dump(HERE/'qa/postprocessing_known_answer_tests.json',dict(passed=True,synthetic_QA_only=True,formal_case_results_untouched=True,checks=checks,utc=now()))
    print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
