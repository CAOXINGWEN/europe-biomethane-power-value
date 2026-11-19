"""V36 paths, atomic receipts, resource inputs and reproducible allocation."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
V32=ROOT/'dataset_one/44_followup_corrected_system_v32/stage_b'
RAW=ROOT/'dataset_one/07_eraa_preprocessed/01_extracted_raw'
PY='/home/cxw/anaconda3/envs/pv/bin/python'
TRAIN=[23,11,9]
SCENARIOS=['Low','Central','High']
POWER=25.96676631231336


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+'\n');tmp.replace(path)
def arrays(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def available_gib():
    return next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
def child_env():
    v=os.environ.copy()
    for k in list(v):
        if k.startswith(('V28_','V31_','V32_')):v.pop(k)
    v.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',V32_SOLVER_MEMORY_MB='12288',V36_STAGE_METHODS='1,2')
    return v


def capped_proportional(weights,caps,total):
    weights=np.asarray(weights,float);caps=np.asarray(caps,float);out=np.zeros_like(caps)
    if total>caps.sum()+1e-8:raise ValueError('Budget exceeds caps')
    for _ in range(len(caps)+2):
        rem=total-out.sum();room=np.maximum(caps-out,0);active=room>1e-12
        if rem<=1e-10:break
        w=np.where(active,weights,0)
        if w.sum()<=0:w=active.astype(float)
        out+=np.minimum(room,rem*w/w.sum())
    if abs(out.sum()-total)>1e-7:raise ValueError('Allocation did not close')
    return out


def resource_table():
    f=pd.read_csv(ROOT/'dataset_one/02_processed/biomethane_country_year_scenarios_15country_schema_2014_2023.csv')
    f=f[f.year.between(2020,2023)&f.deliverable_fuel_TWh_LHV.notna()]
    f=f.groupby(['country_code','scenario'],as_index=False).deliverable_fuel_TWh_LHV.mean()
    u=pd.read_csv(HERE/'inputs/uk_resource_v36.csv');u=u[u['case'].eq('primary_mean_2021_2022')]
    u=u.rename(columns={'TWh_fuel_per_yr':'deliverable_fuel_TWh_LHV'});u['country_code']='GB'
    f=pd.concat([f,u[f.columns]],ignore_index=True)
    return f


def resource_vector(d,scenario):
    f=resource_table();by=f[f.scenario.eq(scenario)].set_index('country_code').deliverable_fuel_TWh_LHV
    shares=np.load(HERE/'inputs/resource_zone_weights.npz',allow_pickle=False)['weights']
    return np.array([by[c]*1e6/8760*shares[j] for j,c in enumerate(d['countries'])])


def load_input(weather,scenario='Central',use='R0',beta=.75):
    d=arrays(V32/'inputs'/f'2035_WS{weather:02d}.npz')
    legacy_base=d['base'].copy();fixed=d['fixed_other']-legacy_base[None,:]
    q=resource_vector(d,scenario);d['resource_q']=q.copy()
    # Exact source waveform, extended to the UK by the same zone-hash phase rule.
    f=pd.read_csv(ROOT/'dataset_one/39_public_validation_system_comparison_v27/station/processed/luo_measured_production_hourly.csv')
    x=f.observed_biogas_m3_customer_h.to_numpy()[slice(0,240) if str(d['gas_split'])=='train' else slice(240,480)]
    phase=np.array([int(hashlib.sha256(str(z).encode()).hexdigest()[:8],16)%len(x) for z in d['zones']])
    raw=x[(np.arange(8760)[:,None]+phase)%len(x)];hourly=raw/raw.mean(axis=0)*q[None,:]
    d['resource_q_hourly']=hourly.copy();d['beta']=np.array(beta)
    d['fixed_other']=fixed
    if use in ['R2','R5']:
        d['q']=q;d['q_hourly']=hourly;d['base']=beta*.4*q
        d['fixed_other']=fixed+d['base'][None,:]
    else:
        d['q']=np.zeros_like(q);d['q_hourly']=np.zeros_like(hourly);d['base']=np.zeros_like(q)
    if use=='R1':d['demand']=d['demand']-.4*hourly
    therm=arrays(HERE/'inputs/thermal_split_2035.npz')
    for k in therm:
        if k!='zones':d[k]=therm[k]
    if use=='R3':d['gas_relief']=hourly*d['gas_efficiency'][None,:]*d['event'][:,None]
    return d


def design(scenario):
    p=HERE/'inputs'/f'design_{scenario}.json'
    if p.exists():return json.loads(p.read_text())
    weights=np.zeros(24);sources=[]
    for w in TRAIN:
        path=HERE/'results'/f'R0_WS{w:02d}.json';r=json.loads(path.read_text())
        if not r.get('success'):raise ValueError('Training R0 failed')
        weights+=np.array(r['zone_EENS_MWh']);sources.append(dict(path=str(path.relative_to(HERE)),sha256=sha(path)))
    d=load_input(TRAIN[0],scenario,'R2');q=d['q']
    caps=.4*(12-.75)*q;power=capped_proportional(weights,caps,POWER*q.sum()/resource_vector(d,'Central').sum())
    value=dict(scenario=scenario,zones=d['zones'].tolist(),power=power.tolist(),energy=(22*q).tolist(),weights=weights.tolist(),train_weather=TRAIN,training_sources=sources,holdout_used_for_fitting=False)
    dump(p,value);return value
