"""Predeclared R5 stratification and common-size, 36-weather ELCC search."""
import json
import numpy as np
from common36 import HERE,now,dump,sha,load_input,design


def good(name):
    p=HERE/'results'/(name+'.json')
    if not p.exists():return False
    r=json.loads(p.read_text())
    return r.get('success') is True and r.get('physical_verified') is True


def receipt(name):return json.loads((HERE/'results'/(name+'.json')).read_text())
def minimum(name):
    r=receipt(name);return float(r.get('minimum_EENS_MWh',r['EENS_MWh']))


def add_job(manifest,j):
    if not any(x['name']==j['name'] for x in manifest['jobs']):manifest['jobs'].append(j);return True
    return False


def update_frontier(manifest):
    if (HERE/'inputs/frontier_weather.json').exists():return False
    if not all(good(f'R0_WS{w:02d}') for w in range(1,37)):return False
    positive=sorted([w for w in range(1,37) if minimum(f'R0_WS{w:02d}')>.001],key=lambda w:(minimum(f'R0_WS{w:02d}'),w))
    groups=np.array_split(positive,8);sample=[int(g[len(g)//2]) for g in groups]
    rows=[dict(weather=w,weight_positive_weathers=len(g),represented_weathers=list(map(int,g)),rank=positive.index(w)+1) for w,g in zip(sample,groups)]
    dump(HERE/'inputs/frontier_weather.json',dict(selected_utc=now(),sorted_positive_weathers=positive,positive_count=len(positive),strata=rows,all_weather_divisor=36,zero_risk_weathers=[w for w in range(1,37) if w not in positive],selection_only_uses_R0=True))
    for beta in [.75,0.]:
        for power in [.26,1.,4.,16.]:
            for buffer in [22,168,720]:
                for row in rows:
                    w=row['weather'];name=f'R5_b{beta:g}_p{power:g}_s{buffer}_WS{w:02d}'
                    add_job(manifest,dict(name=name,batch='R5',weather=w,scenario='Central',beta=beta,power_GW=power,buffer_h=buffer,stratum_weight=row['weight_positive_weathers']))
    manifest['amendments'].append(dict(utc=now(),type='predeclared_adaptive_job_expansion',batch='R5',positive_weather_count=len(positive),jobs_added=192,rank_list=[r['rank'] for r in rows],criterion_changed=False))
    return True


def update_elcc(manifest):
    statepath=HERE/'protocol/elcc_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
    changed=False
    for use in ['R1','R2']:
        if not all(good(f'{use}_Central_WS{w:02d}') and good(f'R0_WS{w:02d}') for w in range(1,37)):continue
        if use not in state:
            target=sum(minimum(f'R0_WS{w:02d}')-minimum(f'{use}_Central_WS{w:02d}') for w in range(1,37))/36
            risk=[w for w in range(1,37) if minimum(f'R0_WS{w:02d}')>.001]
            d=load_input(23,'Central','R2');capacity=.4*d['q'] if use=='R1' else .75*.4*d['q']+np.array(design('Central')['power'])
            state[use]=dict(target_mean_reduction_MWh=target,risk_weathers=risk,shares=(capacity/capacity.sum()).tolist(),lo_MW=0.,hi_MW=float(2*capacity.sum()),iteration=0,mode='upper_bracket',evaluations=[],complete=target<=1e-8)
            changed=True
        s=state[use]
        if s['complete']:continue
        if s.get('pending_names'):
            if not all(good(x) for x in s['pending_names']):continue
            reduction=sum(minimum(f'R0_WS{w:02d}')-minimum(name) for w,name in zip(s['risk_weathers'],s['pending_names']))/36
            point=dict(capacity_MW=s['pending_MW'],mean_reduction_MWh=reduction,job_names=s.pop('pending_names'),mode=s['mode'])
            s['evaluations'].append(point);error=reduction-s['target_mean_reduction_MWh']
            s['absolute_error_MWh']=abs(error);s['last_capacity_MW']=point['capacity_MW'];s['last_mean_reduction_MWh']=reduction
            tolerance=max(1.,.005*abs(s['target_mean_reduction_MWh']))
            if s['mode']=='upper_bracket' and error<0:
                s['lo_MW']=point['capacity_MW'];s['hi_MW']*=2
                if len(s['evaluations'])>8:raise ValueError('ELCC failed to bracket after eight expansions')
            else:
                if error>=0:s['hi_MW']=point['capacity_MW']
                else:s['lo_MW']=point['capacity_MW']
                if abs(error)<=tolerance or (s['mode']=='bisection' and s['iteration']>=12):
                    s['complete']=True;s['converged']=abs(error)<=tolerance;s['tolerance_MWh']=tolerance
                    s['ELCC_MW']=point['capacity_MW'];s['country_allocation_warning']='Shares allocate a system ELCC; they are not independently identified country causal ELCCs.'
                    dump(HERE/'report'/f'ELCC_{use}.json',s);changed=True;continue
                s['mode']='bisection';s['iteration']+=1
            changed=True
        if not s.get('pending_names') and not s['complete']:
            cap=s['hi_MW'] if s['mode']=='upper_bracket' else .5*(s['lo_MW']+s['hi_MW'])
            ix=len(s['evaluations']);names=[]
            for w in s['risk_weathers']:
                name=f'R4_{use}_iter{ix:02d}_WS{w:02d}';names.append(name)
                changed=add_job(manifest,dict(name=name,batch='R4',weather=w,scenario='Central',use=use,firm_MW=cap,firm_shares=s['shares'],iteration=ix)) or changed
            s['pending_names']=names;s['pending_MW']=cap;changed=True
    if changed:dump(statepath,state)
    return changed


def update(manifest):
    a=update_frontier(manifest);b=update_elcc(manifest)
    if a or b:dump(HERE/'protocol/jobs_v36.json',manifest)
    return a or b
