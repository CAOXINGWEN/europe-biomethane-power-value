"""Detached resumable V38 queue; pilot gate, resource gate, retained attempts, automatic numerical export."""
import sys,time,subprocess,fcntl,os,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common38 import *
STATE=HERE/'logs/queue_state.json'

def alive(info):
    try: args=Path(f"/proc/{info['pid']}/cmdline").read_bytes().split(b'\0')
    except FileNotFoundError: return False
    return any(a.endswith(b'run_case38.py') for a in args) and info['name'].encode() in args

def archive(name,reason):
    stamp=now().replace(':','').replace('+','_')
    for suffix in ['.json','.npz','_s1.json','_s1.npz']:
        p=HERE/'results'/(name+suffix)
        if p.exists(): p.rename(p.with_name(name+'.attempt_failed_'+stamp+suffix))
    with (HERE/'logs/attempt_history.jsonl').open('a') as out:
        out.write(json.dumps(dict(name=name,reason=reason,utc=now()))+'\n')

def main():
    lock=(HERE/'logs/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((HERE/'qa/input_preflight.json').read_text())['passed']
    old=json.loads(STATE.read_text()) if STATE.exists() else {}
    attempts=old.get('attempts',{}); active={}; processes={}
    for name,info in old.get('active',{}).items():
        if alive(info): active[name]=info
        elif not receipt_ok(name): archive(name,'controller_restart')
    js=sorted(jobs(),key=lambda j:(j['priority'],WEATHERS.index(j['weather']),j['name']))
    last_launch=0.
    while True:
        for name,info in list(active.items()):
            proc=processes.get(name)
            running=(proc.poll() is None) if proc else alive(info)
            if running and time.time()-info['started_epoch']>15000:
                if proc:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: proc.kill();proc.wait()
                else:
                    os.kill(info['pid'],15)
                running=False
            if not running:
                active.pop(name);processes.pop(name,None)
                if not receipt_ok(name): archive(name,'worker_exit_or_timeout')
        gate=HERE/'qa/reproduction.json'
        gp=json.loads(gate.read_text()).get('passed') if gate.exists() else None
        ok={j['name']:bool(receipt_ok(j['name'])) for j in js}
        pilot=next(j for j in js if j['batch']=='QA')
        if gp is None and ok[pilot['name']] and pilot['name'] not in active:
            dump(gate,dict(passed=False,error='Successful pilot receipt without verification gate',utc=now()))
            gp=False
        failed=[j['name'] for j in js if not ok[j['name']] and attempts.get(j['name'],0)>=3]
        by={b:dict(registered=sum(j['batch']==b for j in js),successful=sum(ok[j['name']] for j in js if j['batch']==b)) for b in ['QA','S','M']}
        state='running' if active else 'waiting_memory'
        if gp is False: state='pilot_failed'
        if all(ok.values()): state='dispatch_complete'
        if failed and not active: state='failed_cases'
        dump(STATE,dict(attempts=attempts,active=active,controller_pid=os.getpid(),utc=now()))
        dump(HERE/'report/progress.json',dict(state=state,updated_utc=now(),registered=len(js),successful=sum(ok.values()),
             by_batch=by,running=list(active.values()),available_GiB=round(available(),2),pilot_passed=gp,exhausted=failed,controller_pid=os.getpid()))
        if state=='dispatch_complete':
            dump(HERE/'report/dispatch_complete.json',dict(success=True,utc=now(),cases=len(js)))
            with (HERE/'logs/postprocess.log').open('a') as log:
                rc=subprocess.call([PY,'-I','-B',str(HERE/'scripts/postprocess38.py')],env=env(),stdout=log,stderr=subprocess.STDOUT)
            dump(HERE/'report/controller_complete.json',dict(utc=now(),postprocess_exitcode=rc,success=rc==0))
            break
        if state in ['pilot_failed','failed_cases']: break
        eligible=[j for j in js if not ok[j['name']] and j['name'] not in active and attempts.get(j['name'],0)<3 and (j['batch']=='QA' or gp is True)]
        if eligible and len(active)<4 and available()>=26 and time.time()-last_launch>=45:
            j=eligible[0];name=j['name']
            if (HERE/'results'/(name+'.json')).exists(): archive(name,'retry_previous_receipt')
            attempts[name]=attempts.get(name,0)+1
            child=env()
            if attempts[name]>1: child['V36_STAGE_METHODS']='1,1'
            with (HERE/'logs'/(name+'.log')).open('a') as out:
                p=subprocess.Popen([PY,'-I','-B',str(HERE/'scripts/run_case38.py'),'--job',name],env=child,
                     stdout=out,stderr=subprocess.STDOUT,start_new_session=True,cwd=HERE)
            active[name]=dict(name=name,pid=p.pid,started_epoch=time.time(),started_utc=now(),attempt=attempts[name])
            processes[name]=p;last_launch=time.time()
            dump(STATE,dict(attempts=attempts,active=active,controller_pid=os.getpid(),utc=now()))
        time.sleep(15)
    print('QUEUE EXIT',now(),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        dump(HERE/'qa/controller_error.json',dict(utc=now(),error=traceback.format_exc()))
        raise
