"""Export paired factorial results, exact Shapley effects and closure QA; partial-safe."""
import sys,math,itertools,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *
import accounting37

def shapley(values,n):
    out=np.zeros(n)
    for i in range(n):
        for bits in itertools.product([0,1],repeat=n):
            if bits[i]: continue
            k=sum(bits); other=list(bits);other[i]=1
            out[i]+=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*(values[tuple(other)]-values[bits])
    return out

def main():
    # Known-answer check includes interaction; no expected direction is imposed on real results.
    synthetic={b:2*b[0]+3*b[1]+4*b[0]*b[1] for b in itertools.product([0,1],repeat=2)}
    assert np.allclose(shapley(synthetic,2),[4,5])
    cells=json.loads((HERE/'protocol/cells_v38.json').read_text()); rows=[];missing=[]
    for c in cells:
        pair={}
        for use in ['R0','U1']:
            p=ROOT/c[use]['path']; rp=p.with_suffix('.json')
            if not p.exists() or not rp.exists(): break
            r=json.loads(rp.read_text())
            if not r.get('success') or not r.get('physical_verified'): break
            assert sha(p)==r['arrays_sha256']
            pair[use]=arrays(p)
        if len(pair)!=2: missing.append(c);continue
        d=accounting37.pair_metrics(pair['U1'],pair['R0']);m=accounting37.finish_u1(d)
        row={k:v for k,v in c.items() if k not in ['R0','U1']}
        row.update(D=m['D_fossil_per_MWh'],avoided=m['avoided_GWP100_t_per_MWh'],s_f=m['s_f'],intensity=m['m_t_per_MWh_e'],
                   balance_relative=m['balance_residual_relative'],B_MWh=d['biomethane_MWh'],delta_EENS_MWh=d['delta_EENS_MWh'])
        row.update({k:v/d['biomethane_MWh'] for k,v in d.items() if k.startswith('balance_')})
        rows.append(row)
    allrows=pd.DataFrame(rows)
    sh=[];inter=[]
    for kind,n,factors in [('mechanism',3,['background_year','vre_year','price_year']),('scale',2,['mix','total'])]:
        frame=allrows[allrows.kind==kind] if len(allrows) else pd.DataFrame()
        frame.to_csv(HERE/'report'/f'{kind}_by_weather.csv',index=False)
        if not len(frame): continue
        for w,g in frame.groupby('weather'):
            if len(g)!=2**n: continue
            for metric in ['D','avoided']:
                values={tuple(int(getattr(r,k)==2035) if kind=='mechanism' else int(getattr(r,k)==('fleet' if k=='mix' else 'large')) for k in factors):getattr(r,metric) for r in g.itertuples()}
                effects=shapley(values,n);total=values[(1,)*n]-values[(0,)*n]
                assert abs(sum(effects)-total)<1e-10
                for name,eff in zip(factors,effects): sh.append(dict(kind=kind,weather=int(w),metric=metric,factor=name,effect=eff,total_change=total))
                for subset in itertools.product([0,1],repeat=n):
                    if sum(subset)<2: continue
                    members=[b for b in values if all(x<=y for x,y in zip(b,subset))]
                    value=sum((-1)**(sum(subset)-sum(b))*values[b] for b in members)
                    inter.append(dict(kind=kind,weather=int(w),metric=metric,factors='+'.join(k for k,b in zip(factors,subset) if b),effect=value))
    pd.DataFrame(sh).to_csv(HERE/'report/factor_shapley_by_weather.csv',index=False)
    pd.DataFrame(inter).to_csv(HERE/'report/factor_interactions_by_weather.csv',index=False)
    if sh: pd.DataFrame(sh).groupby(['kind','metric','factor']).effect.agg(['mean','min','max','count']).to_csv(HERE/'report/factor_shapley_summary.csv')
    successful=sum(bool(receipt_ok(j['name'])) for j in jobs())
    reg=registry(); unchanged=all(sha(ROOT/p)==h for p,h in {**reg['source_sha256'],**reg['reused_source_sha256']}.items())
    maxres=float(allrows.balance_relative.abs().max()) if len(allrows) else None
    complete=not missing and successful==len(jobs())
    pilot_path=HERE/'qa/reproduction.json'
    pilot_passed=bool(pilot_path.exists() and json.loads(pilot_path.read_text()).get('passed'))
    passed=unchanged and (maxres is not None and maxres<1e-8) and pilot_passed
    receipt=dict(utc=now(),partial=not complete,passed=passed,successful_new_cases=successful,registered_new_cases=len(jobs()),
                 cells_complete=len(rows),cells_registered=len(cells),sources_unchanged=unchanged,max_balance_relative=maxres,
                 known_answer_shapley_passed=True,pilot_passed=pilot_passed,missing_cells=len(missing))
    dump(HERE/'report/analysis_progress.json',receipt)
    with pd.ExcelWriter(HERE/'report/V38_results.xlsx',engine='openpyxl') as x:
        allrows.to_excel(x,sheet_name='paired_results',index=False)
        pd.DataFrame(sh).to_excel(x,sheet_name='Shapley',index=False)
        pd.DataFrame(inter).to_excel(x,sheet_name='interactions',index=False)
        pd.DataFrame([receipt]).to_excel(x,sheet_name='completion',index=False)
    if complete and passed:
        dump(HERE/'qa/completion_receipt.json',receipt)
        dump(HERE/'report/computation_complete.json',dict(success=True,**receipt))
        z=HERE/'report/V38_numeric_handoff.zip'
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as out:
            for folder in ['scripts','protocol','report','qa','literature']:
                for p in (HERE/folder).rglob('*'):
                    if p.is_file() and p!=z and p.suffix not in ['.pyc']:out.write(p,p.relative_to(HERE))
        with zipfile.ZipFile(z) as check: assert check.testzip() is None
    print(json.dumps(receipt),flush=True)
    if not passed: raise SystemExit(1)
if __name__=='__main__': main()
