"""Two-worker queue for V35. Each case runs in its own process; existing receipts are never overwritten."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

from common35 import HERE, TRAIN, PY, now, dump, available_GiB, child_env, design_file, baseline_files

REG = json.loads((HERE / 'protocol/jobs_v35.json').read_text())
JOBS = REG['jobs']
LOCK = threading.RLock()  # re-entrant: ready() may create designs while next_job() holds the lock
TAKEN = set()
ATTEMPTS = {}
MAX_ATTEMPTS = 3
MIN_GiB = float(sys.argv[1]) if len(sys.argv) > 1 else 14.
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def running_names():
    out = subprocess.run(['pgrep', '-af', 'run_case.py --job'], capture_output=True, text=True).stdout
    return {line.split('--job')[-1].strip() for line in out.splitlines() if '--job' in line}


def done(name):
    p = HERE / 'results' / (name + '.json')
    return p.exists()


def ok(name):
    p = HERE / 'results' / (name + '.json')
    return p.exists() and json.loads(p.read_text()).get('success') is True


def ready(job):
    if job['kind'] == 'baseline' or job['scenario'] == 'main':
        return True
    sc = job['scenario']
    if not all(ok(f'{sc}_baseline_WS{w:02d}') for w in TRAIN):
        return False
    if not ok(f"{sc}_baseline_WS{job['weather']:02d}"):
        return False
    if not design_file(sc, job['rule']).exists():
        with LOCK:
            if not design_file(sc, job['rule']).exists():
                subprocess.run([PY, str(HERE / 'scripts/designs.py'), sc], check=True, cwd=HERE / 'scripts')
    return True


def next_job():
    with LOCK:
        busy = running_names()
        for j in JOBS:
            if j['name'] in TAKEN or done(j['name']) or j['name'] in busy:
                continue
            if ready(j):
                TAKEN.add(j['name'])
                return j
    return None


def worker(i):
    while True:
        j = next_job()
        if j is None:
            pending = [x for x in JOBS if not done(x['name']) and x['name'] not in TAKEN]
            if not pending:
                return
            time.sleep(20)
            continue
        while available_GiB() < MIN_GiB:
            time.sleep(30)
        log = HERE / 'logs' / (j['name'] + '.log')
        ATTEMPTS[j['name']] = ATTEMPTS.get(j['name'], 0) + 1
        with log.open('a') as f:
            f.write(f'\n=== attempt {ATTEMPTS[j["name"]]} {now()}\n'); f.flush()
            try:
                rc = subprocess.run([PY, str(HERE / 'scripts/run_case.py'), '--job', j['name']], stdout=f,
                                    stderr=subprocess.STDOUT, env=child_env(), cwd=HERE / 'scripts', timeout=2700).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            f.write(f'=== exit {rc} {now()}\n')
        if not done(j['name']):
            # Killed without a receipt (for example by the OOM killer): record and retry.
            with (HERE / 'logs/attempt_history.csv').open('a') as h:
                h.write(f"{j['name']},{ATTEMPTS[j['name']]},{rc},{now()}\n")
            with LOCK:
                if ATTEMPTS[j['name']] < MAX_ATTEMPTS:
                    TAKEN.discard(j['name'])
                else:
                    dump(HERE / 'results' / (j['name'] + '.json'),
                         dict(job=j, success=False, error=f'no receipt after {MAX_ATTEMPTS} attempts, last exit {rc}', completed_utc=now()))
        status()


def status():
    with LOCK:
        n = len(JOBS)
        d = sum(done(j['name']) for j in JOBS)
        good = sum(ok(j['name']) for j in JOBS)
        dump(HERE / 'report/progress.json', dict(updated_utc=now(), registered=n, finished=d, successful=good,
                                                 failed=d - good, available_GiB=available_GiB()))


if __name__ == '__main__':
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for t in threads:
        t.start()
        time.sleep(40)
    for t in threads:
        t.join()
    status()
    print('QUEUE FINISHED', now(), flush=True)
