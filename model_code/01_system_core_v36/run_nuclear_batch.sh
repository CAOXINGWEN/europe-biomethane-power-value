#!/bin/bash
# Runs the remaining R1n weathers, 6 at a time; each case waits until at least 24 GiB is available.
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_one() {
  w=$1; name=$(printf "R1n_Central_WS%02d" $w)
  [ -f ../results/$name.json ] && exit 0
  while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 24 ]; do sleep 30; done
  for attempt in 1 2 3; do
    timeout 5400 /home/cxw/anaconda3/envs/pv/bin/python -B run_nuclear_fixed.py --weather $w >> ../logs/$name.out 2>&1
    [ -f ../results/$name.json ] && exit 0
    echo "$name,$attempt,$(date -u +%FT%TZ)" >> ../logs/R1n_attempt_history.csv
    sleep 20
  done
}
export -f run_one
seq 1 36 | grep -v -x 13 | xargs -P 6 -I{} bash -c 'run_one {}'
echo "R1n batch finished $(date -u +%FT%TZ)" >> ../logs/R1n_batch.out
