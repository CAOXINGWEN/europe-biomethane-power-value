"""Session-independent queue/postprocessing recovery with bounded retries and PID binding."""
from pathlib import Path
import sys,os,json,subprocess,time,fcntl,traceback
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common36 import HERE,PY,child_env,now,dump


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def controller_alive(progress):
    pid=progress.get('controller_pid')
    if not pid:return False
    try:args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return False
    return str(HERE/'scripts/queue36.py').encode() in args


def main():
    lock=(HERE/'logs/supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(HERE/'logs/supervisor_state.json');restarts=state.get('queue_restarts',0);posts=state.get('postprocess_attempts',0)
    while True:
        progress=read(HERE/'report/progress.json');complete=read(HERE/'report/computation_complete.json')
        status='running';error=None
        if complete.get('success'):status='complete'
        elif progress.get('failed_exhausted'):status='stopped_exhausted_case_retries'
        elif not controller_alive(progress):
            if (HERE/'report/dispatch_complete.json').exists():
                if posts>=3:status='stopped_postprocessing_failure'
                else:
                    posts+=1
                    with (HERE/'logs/postprocess_recovery.log').open('a') as f:
                        result=subprocess.run([PY,'-I','-B',str(HERE/'scripts/postprocess36.py')],stdout=f,stderr=subprocess.STDOUT,env=child_env())
                    error=None if result.returncode==0 else f'postprocessing return code {result.returncode}'
            elif restarts>=3:status='stopped_controller_restart_limit'
            else:
                restarts+=1
                with (HERE/'logs/queue.log').open('a') as f:
                    proc=subprocess.Popen([PY,'-I','-B',str(HERE/'scripts/queue36.py'),'6','24'],stdout=f,stderr=subprocess.STDOUT,env=child_env(),start_new_session=True)
                dump(HERE/'logs/supervisor_last_restart.json',dict(utc=now(),controller_pid=proc.pid,attempt=restarts))
        state=dict(updated_utc=now(),supervisor_pid=os.getpid(),state=status,queue_restarts=restarts,postprocess_attempts=posts,last_error=error,successful_cases=progress.get('successful'),registered_cases=progress.get('registered'),controller_pid=progress.get('controller_pid'),server_reboot_requires_manual_relaunch=True)
        dump(HERE/'logs/supervisor_state.json',state);dump(HERE/'report/supervisor_status.json',state)
        if status!='running':return
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception:
        dump(HERE/'qa/supervisor_error.json',dict(utc=now(),error=traceback.format_exc()));raise
