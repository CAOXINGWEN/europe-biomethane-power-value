"""Restartable memory-gated V37 queue (V36 queue36 pattern): RLock, immutable failed receipts, PID adoption.
Usage: queue37.py WORKERS MIN_GIB
"""
from pathlib import Path
import sys, os, json, subprocess, time, threading, fcntl, traceback
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common37 import HERE, PY, TRAIN, now, dump, available_gib, child_env

LOCK = threading.RLock(); LAUNCH = threading.Lock(); TAKEN = set(); ACTIVE = {}; EXTERNAL = {}; STOP = threading.Event()
STATE = HERE / 'logs/queue_state.json'; ATTEMPTS = {}; WORKERS = 6; MIN_GIB = 24.
ORDER = {'REPRO': 0, 'B0': 1, 'F0': 1.5, 'F1': 2, 'B1': 2.5, 'F2': 3, 'F3': 4, 'F4': 5, 'F5': 6}      # amendment_post_02
V36R = HERE.parent / '49_use_value_v36/results'
YEAR_ORDER = {2028: 0, 2035: 1, 2030: 2, 2033: 3}      # B3 endpoints first


def manifest(): return json.loads((HERE / 'protocol/jobs_v37.json').read_text())


def success(name):
    p = HERE / 'results' / (name + '.json')
    if not p.exists(): return False
    try: r = json.loads(p.read_text())
    except Exception: return False
    return bool(r.get('success')) and bool(r.get('physical_verified')) and r.get('stages_saved') == 2


def r0_ok(year, w, flex=False):
    if flex: return success(f'F0_{year}_R0_WS{w:02d}')
    if year == 2035: return (V36R / f'R0_WS{w:02d}.json').exists()
    return success(f'B0_{year}_R0_WS{w:02d}')


def ready(j):
    if j['batch'] in ('REPRO', 'B0', 'F0'): return True
    if (HERE / 'protocol/hold.json').exists() and j['batch'] in json.loads((HERE / 'protocol/hold.json').read_text()).get('batches', []): return False
    if j.get('resource') not in ('none', 'V36Central') and not (HERE / 'inputs/resource_v37_country_year.csv').exists(): return False
    if j.get('resource') == 'fleet' and not (HERE / 'inputs/resource_fleet_ready.json').exists(): return False
    fx = bool(j.get('flex'))
    if j['use'] == 'U2' and not all(r0_ok(j['year'], w, fx) for w in TRAIN + [j['weather']]): return False
    if j.get('sens') == 'nuclear_fixed' and not r0_ok(j['year'], j['weather'], fx): return False
    return True


def status():
    with LOCK:
        jobs = manifest()['jobs']; by = {}
        for b in ORDER:
            js = [j for j in jobs if j['batch'] == b]
            by[b] = dict(registered=len(js), successful=sum(success(j['name']) for j in js), running=sum(j['name'] in ACTIVE for j in js))
        dump(HERE / 'report/progress.json', dict(updated_utc=now(), controller_pid=os.getpid(), registered=len(jobs),
             successful=sum(v['successful'] for v in by.values()), running=list(ACTIVE), by_batch=by, available_GiB=round(available_gib(), 1)))
        dump(STATE, dict(attempts=ATTEMPTS, active=ACTIVE, updated_utc=now(), controller_pid=os.getpid()))


def next_job():
    with LOCK:
        for j in sorted(manifest()['jobs'], key=lambda j: (ORDER[j['batch']], YEAR_ORDER[j['year']] if j['batch'] in ('B3', 'F3') else j['year'], j['weather'], j['name'])):
            n = j['name']
            if n in TAKEN or success(n) or ATTEMPTS.get(n, 0) >= 3: continue
            if ready(j): TAKEN.add(n); return j
    return None


def archive_failed(name, rc):
    stamp = now().replace(':', '').replace('+', '_'); res = HERE / 'results' / (name + '.json')
    if not res.exists(): dump(res, dict(success=False, job=name, error='Worker exited without a successful receipt', returncode=rc, completed_utc=now()))
    for end in ['.json', '.npz', '_s1.json', '_s1.npz']:
        p = HERE / 'results' / (name + end)
        if p.exists(): p.rename(p.with_name(name + '.attempt_failed_' + stamp + end))


def alive(name, info):
    try: args = Path(f"/proc/{info['pid']}/cmdline").read_bytes().split(b'\0')
    except FileNotFoundError: return False
    # Match by script file name and exact case name (robust to relative or absolute launch paths).
    return any(a.endswith(b'run_case37.py') for a in args) and name.encode() in args


def launch_ready():
    with LOCK: return available_gib() >= MIN_GIB and len(ACTIVE) < WORKERS


def worker(i):
    while not STOP.is_set():
        job = next_job()
        if job is None:
            if STOP.wait(20): return
            continue
        name = job['name']
        try:
            with LAUNCH:
                while not launch_ready():
                    if STOP.wait(15): return
                with LOCK:
                    ATTEMPTS[name] = ATTEMPTS.get(name, 0) + 1
                    if (HERE / 'results' / (name + '.json')).exists(): archive_failed(name, 'existing_failed_receipt')
                with (HERE / 'logs' / (name + '.log')).open('a') as out:
                    out.write(f'\nAttempt {ATTEMPTS[name]} at {now()}\n'); out.flush()
                    proc = subprocess.Popen([PY, '-B', str(HERE / 'scripts/run_case37.py'), '--job', name], stdout=out, stderr=subprocess.STDOUT,
                                            env=child_env(), start_new_session=True, cwd=str(HERE))
                with LOCK: ACTIVE[name] = dict(pid=proc.pid, worker=i, started_utc=now()); status()
                time.sleep(45)          # let the new LP reach its memory peak before the next launch check
            try: rc = proc.wait(timeout=16000)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                rc = 'timeout'
            with LOCK:
                ACTIVE.pop(name, None)
                with (HERE / 'logs/attempt_history.csv').open('a') as f: f.write(f'{name},{ATTEMPTS[name]},{rc},{now()}\n')
                if not success(name): archive_failed(name, rc)
                TAKEN.discard(name); status()
        except Exception:
            dump(HERE / 'qa' / f'queue_worker_{i}_error.json', dict(utc=now(), job=job, error=traceback.format_exc())); STOP.set()


def main():
    global WORKERS, MIN_GIB, ATTEMPTS
    WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6; MIN_GIB = float(sys.argv[2]) if len(sys.argv) > 2 else 24.
    lock = (HERE / 'logs/controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists():
        old = json.loads(STATE.read_text()); ATTEMPTS = old.get('attempts', {})
        for name, info in old.get('active', {}).items():
            if success(name): continue
            if alive(name, info): TAKEN.add(name); ACTIVE[name] = info; EXTERNAL[name] = info
            else: archive_failed(name, 'controller_restart')
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    for t in threads: t.start()
    while not STOP.wait(30):
        with LOCK:
            for name, info in list(EXTERNAL.items()):
                if success(name) or not alive(name, info):
                    ACTIVE.pop(name, None); EXTERNAL.pop(name, None); TAKEN.discard(name)
                    if not success(name): archive_failed(name, 'adopted_worker_exit')
            status()
            jobs = manifest()['jobs']
            if all(success(j['name']) or ATTEMPTS.get(j['name'], 0) >= 3 for j in jobs) and not ACTIVE:
                dump(HERE / 'report/dispatch_complete.json', dict(utc=now(), jobs=len(jobs), successful=sum(success(j['name']) for j in jobs),
                     exhausted=[j['name'] for j in jobs if not success(j['name'])])); STOP.set(); break
    for t in threads: t.join(timeout=1)
    status(); print('QUEUE EXIT', now(), flush=True)


if __name__ == '__main__': main()
