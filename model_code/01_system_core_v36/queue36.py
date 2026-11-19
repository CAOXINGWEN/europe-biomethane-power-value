"""Restartable memory-gated V36 queue with immutable failed-attempt receipts."""
from pathlib import Path
import sys,os,json,subprocess,time,threading,fcntl,traceback
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common36 import HERE,PY,TRAIN,now,dump,available_gib,child_env,sha,design
from adaptive36 import update,good

LOCK=threading.RLock();LAUNCH=threading.Lock();TAKEN=set();ACTIVE={};EXTERNAL={};STOP=threading.Event()
STATE=HERE/'logs/queue_state.json'
ATTEMPTS={};WORKERS=3;MIN_GIB=20.


def alive(name,info):
    """Bind adopted PIDs to the exact script and case, including PID-reuse protection."""
    path=Path(f"/proc/{info['pid']}/cmdline")
    try:args=path.read_bytes().split(b'\0')
    except FileNotFoundError:return False
    return str(HERE/'scripts/run_case.py').encode() in args and name.encode() in args


def manifest():return json.loads((HERE/'protocol/jobs_v36.json').read_text())
def success(name):return good(name)


def launch_ready():
    with LOCK:return available_gib()>=MIN_GIB and len(ACTIVE)<WORKERS


def ready(j):
    if j['batch']=='R0':return True
    if j['batch'] in ['R1','R3']:return success(f"R0_WS{j['weather']:02d}")
    if j['batch'] in ['R2','R5']:
        if not all(success(f'R0_WS{w:02d}') for w in TRAIN+[j['weather']]):return False
        design(j['scenario']);return True
    return j['batch']=='R4'


def status():
    with LOCK:
        jobs=manifest()['jobs'];by={}
        for batch in ['R0','R1','R2','R3','R4','R5']:
            js=[j for j in jobs if j['batch']==batch];by[batch]=dict(registered=len(js),successful=sum(success(j['name']) for j in js),running=sum(j['name'] in ACTIVE for j in js))
        complete=[json.loads((HERE/'results'/(j['name']+'.json')).read_text()) for j in jobs if success(j['name'])]
        record=dict(updated_utc=now(),controller_pid=os.getpid(),registered=len(jobs),successful=sum(x['successful'] for x in by.values()),LP_stages_saved=sum(r['stages_saved'] for r in complete),running=list(ACTIVE),by_batch=by,available_GiB=available_gib(),workers=WORKERS,minimum_available_GiB=MIN_GIB,failed_exhausted=[k for k,v in ATTEMPTS.items() if v>=3 and not success(k)],state='stopped_failure' if STOP.is_set() else 'running',adaptive_R4_R5_registration=True)
        dump(HERE/'report/progress.json',record);dump(STATE,dict(attempts=ATTEMPTS,active=ACTIVE,updated_utc=now(),controller_pid=os.getpid()))


def next_job():
    with LOCK:
        preference=HERE/'inputs/queue_priority_cases.json'
        pilots=json.loads(preference.read_text())['pilot_cases'] if preference.exists() else []
        def priority(j):
            if j['name'] in pilots:return pilots.index(j['name'])
            if j['batch']=='R0':return 3
            if j.get('scenario')=='Central' and j.get('variant','primary')=='primary' and j['batch'] in ['R1','R2','R3']:return 10+int(j['batch'][1])
            if j['batch']=='R4':return 14
            if j['batch']=='R5':return 40
            return 20 if j.get('variant','primary')=='primary' else 30
        for j in sorted(manifest()['jobs'],key=priority):
            name=j['name']
            if name in TAKEN or success(name) or ATTEMPTS.get(name,0)>=3:continue
            if ready(j):TAKEN.add(name);return j
    return None


def archive_failed(name,returncode):
    stamp=now().replace(':','').replace('+','_');result=HERE/'results'/(name+'.json')
    if not result.exists():dump(result,dict(success=False,job=name,error='Worker exited without a successful receipt',returncode=returncode,completed_utc=now()))
    for ending in ['.json','.npz','_s1.json','_s1.npz']:
        p=HERE/'results'/(name+ending)
        if p.exists():p.rename(p.with_name(name+'.attempt_failed_'+stamp+ending))


def worker(i):
    while not STOP.is_set():
        job=next_job()
        if job is None:time.sleep(3);continue
        name=job['name']
        try:
            while not launch_ready():
                if STOP.wait(15):return
            with LAUNCH:
                while not launch_ready():
                    if STOP.wait(15):return
                with LOCK:
                    ATTEMPTS[name]=ATTEMPTS.get(name,0)+1
                    if (HERE/'results'/(name+'.json')).exists():archive_failed(name,'existing_failed_receipt')
                logpath=HERE/'logs'/(name+'.log')
                with logpath.open('a') as output:
                    output.write(f'\nAttempt {ATTEMPTS[name]} at {now()}\n');output.flush()
                    process=subprocess.Popen([PY,'-I','-B',str(HERE/'scripts/run_case.py'),'--job',name],stdout=output,stderr=subprocess.STDOUT,env=child_env(),start_new_session=True)
                with LOCK:ACTIVE[name]=dict(pid=process.pid,worker=i,started_utc=now());status()
                time.sleep(6)
            try:rc=process.wait(timeout=7600)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                rc='timeout'
            with LOCK:
                ACTIVE.pop(name,None)
                with (HERE/'logs/attempt_history.csv').open('a') as f:f.write(f'{name},{ATTEMPTS[name]},{rc},{now()}\n')
                if not success(name):
                    archive_failed(name,rc)
                    if ATTEMPTS[name]>=3:STOP.set()
                TAKEN.discard(name);status()
        except Exception:
            dump(HERE/'qa'/f'queue_worker_{i}_error.json',dict(utc=now(),job=job,error=traceback.format_exc()));STOP.set()


def main():
    global WORKERS,MIN_GIB,ATTEMPTS
    WORKERS=int(sys.argv[1]) if len(sys.argv)>1 else 3
    MIN_GIB=float(sys.argv[2]) if len(sys.argv)>2 else 20.
    lock=(HERE/'logs/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for file in ['model_extension_tests.json','thermal_split_reconciliation.json','reproduction_extended_main_eval_rule_eens_WS13_checkpoint_v2.json']:
        assert json.loads((HERE/'qa'/file).read_text())['passed'],file
    proof=json.loads((HERE/'qa/reproduction_extended_main_eval_rule_eens_WS13_checkpoint_v2.json').read_text())
    assert proof['model36_sha256']==sha(HERE/'scripts/model36.py')
    for file,digest in manifest()['code_sha256'].items():assert sha(HERE/file)==digest,('Frozen code changed',file)
    if STATE.exists():
        old=json.loads(STATE.read_text());ATTEMPTS=old.get('attempts',{})
        # Adopt orphaned child processes from an interrupted controller; never duplicate an active case.
        for name,info in old.get('active',{}).items():
            if success(name):continue
            if alive(name,info):
                TAKEN.add(name);ACTIVE[name]=info;EXTERNAL[name]=info
            else:archive_failed(name,'controller_restart')
    pilot=HERE/'logs/external_jobs.json'
    if pilot.exists():
        for name,info in json.loads(pilot.read_text()).items():
            if not success(name) and alive(name,info):
                TAKEN.add(name);ACTIVE[name]=info;EXTERNAL[name]=info;ATTEMPTS[name]=max(1,ATTEMPTS.get(name,0))
    threads=[threading.Thread(target=worker,args=(i,),daemon=True) for i in range(WORKERS)]
    for t in threads:t.start()
    while not STOP.wait(15):
        with LOCK:
            for name,info in list(EXTERNAL.items()):
                if success(name) or not alive(name,info):
                    ACTIVE.pop(name,None);EXTERNAL.pop(name,None);TAKEN.discard(name)
                    if not success(name):
                        archive_failed(name,'adopted_worker_exit')
                        if ATTEMPTS.get(name,0)>=3:STOP.set()
            v=manifest();update(v);status()
            done=all(success(j['name']) for j in v['jobs']) and not ACTIVE
            adaptive=(HERE/'inputs/frontier_weather.json').exists()
            es=HERE/'protocol/elcc_state.json'
            elcc=json.loads(es.read_text()) if es.exists() else {}
            if done and adaptive and all(elcc.get(x,{}).get('complete') for x in ['R1','R2']):
                dump(HERE/'report/dispatch_complete.json',dict(success=True,utc=now(),jobs=len(v['jobs']),by_batch=json.loads((HERE/'report/progress.json').read_text())['by_batch']))
                STOP.set();break
    for t in threads:t.join(timeout=1)
    status()
    if (HERE/'report/dispatch_complete.json').exists():
        v=json.loads((HERE/'report/progress.json').read_text());v['state']='dispatch_complete';dump(HERE/'report/progress.json',v)
        post=HERE/'scripts/postprocess36.py'
        if post.exists():subprocess.run([PY,'-I','-B',str(post)],check=True,env=child_env())
    print('QUEUE EXIT',now(),flush=True)


if __name__=='__main__':main()
