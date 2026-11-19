"""Structural checks before any new solve; no experiment outputs are overwritten."""
import sys,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *

def main():
    checks=[]
    for year in (2028,2035):
        d,_=legacy.load_input(dict(year=year,weather=17,use='R0',resource='none'))
        tc,gc=thermal_prices(d,year)
        checks.append(dict(name=f'own_year_price_{year}',passed=bool(np.allclose(tc,d['thermal_cost'],rtol=0,atol=1e-10) and np.allclose(gc,d['gas_cost'],rtol=0,atol=1e-10))))
    reference=None
    for b,v,c in itertools.product([2028,2035],repeat=3):
        j=dict(year=b,vre_year=v,price_year=c,weather=17,use='U1',mix='food',total='small')
        d=build_input(j); r=build_input(dict(j,use='R0'))
        if reference is None: reference=d['resource_q_hourly'].copy()
        checks.append(dict(name=f'fixed_resource_and_paired_inputs_{b}_{v}_{c}',passed=bool(
            np.array_equal(reference,d['resource_q_hourly']) and
            np.allclose(r['demand']-d['demand'],.4*d['resource_q_hourly'],atol=1e-10,rtol=0) and
            all(np.array_equal(d[k],r[k],equal_nan=True) if d[k].dtype.kind not in 'US' else np.array_equal(d[k],r[k]) for k in d if k!='demand'))))
    for mix,total in itertools.product(['food','fleet'],['small','large']):
        j=dict(year=2035,vre_year=2035,price_year=2035,weather=17,use='U1',mix=mix,total=total)
        d=build_input(j); by=resource_country(mix,total)
        actual={c:float(d['resource_q_hourly'][:,d['countries']==c].sum()/1e6) for c in by.index}
        checks.append(dict(name=f'resource_mass_{mix}_{total}',passed=all(abs(actual[c]-by[c])<1e-9 for c in by.index)))
    j=dict(year=2035,vre_year=2035,price_year=2035,weather=17,use='U1',mix='food',total='small')
    d=build_input(j); old,_=legacy.load_input(dict(year=2035,weather=17,use='U1',resource='Central'))
    checks.append(dict(name='2035_endpoint_matches_V37',passed=all(
        np.allclose(d[k],old[k],rtol=1e-13,atol=1e-10,equal_nan=True) if d[k].dtype.kind not in 'US' else np.array_equal(d[k],old[k]) for k in old)))
    dump(HERE/'qa/input_preflight.json',dict(utc=now(),passed=all(c['passed'] for c in checks),checks=checks))
    print(json.dumps(dict(checks=len(checks),passed=all(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']])))
    if not all(c['passed'] for c in checks): raise SystemExit(1)
if __name__=='__main__': main()
