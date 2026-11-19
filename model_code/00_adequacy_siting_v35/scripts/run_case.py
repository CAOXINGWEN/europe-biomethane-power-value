"""Run one registered V35 case: baseline, layout evaluation or no-foresight evaluation."""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from common35 import (HERE, SCENARIOS, ZERO_FLOOR_MWh, HOUR_THRESHOLD_MW, now, sha, dump, arrays,
                      load_input, input_path, baseline_files, design_file)

JOBS = HERE / 'protocol/jobs_v35.json'


def get_job(name):
    return next(j for j in json.loads(JOBS.read_text())['jobs'] if j['name'] == name)


def load_baseline(scenario, weather):
    pj, pn = baseline_files(scenario, weather)
    br = json.loads(pj.read_text())
    assert br['success'] and br.get('physical_verified', True)
    return br, (None if br['EENS_MWh'] <= ZERO_FLOOR_MWh else arrays(pn)), sha(pj)


def load_design(scenario, rule, d):
    p = design_file(scenario, rule)
    v = json.loads(p.read_text())
    P, S = np.asarray(v['power'], float), np.asarray(v['energy'], float)
    assert list(v['zones']) == d['zones'].tolist()
    caps = float(d['eta_bio']) * (12 - float(d['beta'])) * d['q']
    assert np.all(P <= caps + 2e-4) and abs(S.sum() - 22 * d['q'].sum()) < 2e-4
    return P, S, sha(p)


def configure(p, d, base_arr, P):
    """Identical to V32 runner.configure: DSR and iDSR fixed to the paired baseline."""
    b = p['builder']
    for name in ['dsr', 'idsr_consumption']:
        ids = b.blocks[name].ravel()
        values = base_arr[name].ravel()
        p['bounds'][ids, 0] = values
        p['bounds'][ids, 1] = values
    p['bounds'][b.blocks['gas_withdraw'].ravel(), 1] = np.tile(
        float(d['beta']) * d['q'] + P / float(d['eta_bio']), len(d['demand']))


def run(job):
    import model
    import solver_adapter

    def linprog4(*args, **kw):
        # Same LP and tolerances as V32; only the COPT thread count is raised (CPU contention).
        kw['options'] = dict(kw.get('options') or {}, threads=4)
        return solver_adapter.linprog(*args, **kw)
    model.linprog = linprog4
    sc = SCENARIOS[job['scenario']]
    w = job['weather']
    d = load_input(job['scenario'], w)
    started = time.time()
    rec = dict(job=job, started_utc=now(), input_sha256=sha(input_path(w)),
               scenario_definition=sc, q_total_MW=float(d['q'].sum()))
    if job['kind'] == 'zero_baseline':
        from common35 import V32B
        if job['scenario'] == 'ext075':
            src = V32B.parent / 'results' / f'WS{w:02d}_full_mapped_cut0.3_ext0.75_baseline.json'
        else:
            src = V32B / 'results' / f'baseline_WS{w:02d}.json'
            from greedy import protective_floor
            uk = np.flatnonzero(np.isin(d['zones'], ['UK00', 'UKNI']))
            for z in uk:
                f, _ = protective_floor(d['q_hourly'][:, z], float(d['beta']) * d['q'][z], 16 * d['q'][z],
                                        d['q'][z], float(d['compression_kappa']), float(d['eta_bio']))
                assert f.max() <= 16 * d['q'][z] + 1e-9, 'UK base gas infeasible with 16q'
            rec['uk_base_gas_floor_max_hours'] = [float(protective_floor(d['q_hourly'][:, z], float(d['beta']) * d['q'][z],
                16 * d['q'][z], d['q'][z], float(d['compression_kappa']), float(d['eta_bio']))[0].max() / d['q'][z]) for z in uk]
        src_r = json.loads(src.read_text())
        assert src_r['success'] and src_r['EENS_MWh'] <= ZERO_FLOOR_MWh
        rec.update(success=True, physical_verified=True, analytical_zero=True, EENS_MWh=float(src_r['EENS_MWh']),
                   system_shortage_hours=0, zone_EENS_MWh=[0.] * len(d['q']), zones=d['zones'].tolist(),
                   evidence_file=str(src.relative_to(HERE.parents[1])), evidence_sha256=sha(src))
    elif job['kind'] == 'reuse_baseline':
        # Amendment 01: the V32 A-stage baseline solved the identical electrical LP (gas_cut 0.3, ext 0.75).
        from common35 import V32B
        src = V32B.parent / 'results' / f'WS{w:02d}_full_mapped_cut0.3_ext0.75_baseline'
        sr = json.loads(open(str(src) + '.json').read())
        assert sr['success'] and sr['physical_verified'] and sr['EENS_MWh'] > ZERO_FLOOR_MWh
        a = arrays(str(src) + '.npz')
        un = np.maximum(a['unserved'], 0)
        assert abs(float(a['unserved'].sum()) - float(sr['EENS_MWh'])) < 2e-4
        rec.update(success=True, physical_verified=True, reused_v32_A_stage=True, EENS_MWh=float(sr['EENS_MWh']),
                   system_shortage_hours=int((a['unserved'].sum(axis=1) > HOUR_THRESHOLD_MW).sum()),
                   zone_EENS_MWh=un.sum(axis=0).tolist(), zones=d['zones'].tolist(),
                   zone_shortage_hours_by_zone=(a['unserved'] > HOUR_THRESHOLD_MW).sum(axis=0).tolist(),
                   evidence_file=str(src.relative_to(HERE.parents[1])) + '.json',
                   evidence_sha256=sha(str(src) + '.json'), evidence_arrays_sha256=sha(str(src) + '.npz'))
        np.savez_compressed(HERE / 'results' / (job['name'] + '.npz'), unserved=a['unserved'],
                            dsr=a['dsr'], idsr_consumption=a['idsr_consumption'], flow=a['flow'])
    elif job['kind'] == 'baseline':
        P, S = np.zeros_like(d['q']), 16 * d['q']
        p = model.build_problem(d, tech='none', power=P, energy=S, gas_cut=sc['cut'],
                                external_availability=sc['ext'])
        r = model.solve(p, time_limit=1800, method='copt', secondary=False)
        a = r.pop('arrays')
        un = np.maximum(a['unserved'], 0)
        rec.update({k: r[k] for k in ['success', 'EENS_MWh', 'system_shortage_hours', 'zone_shortage_hours',
                                      'equality_error_MWh', 'inequality_error', 'bound_error', 'calls', 'n_variables']})
        rec.update(zone_EENS_MWh=un.sum(axis=0).tolist(), zones=d['zones'].tolist(),
                   zone_shortage_hours_by_zone=(a['unserved'] > HOUR_THRESHOLD_MW).sum(axis=0).tolist(),
                   physical_verified=bool(max(r['equality_error_MWh'], r['inequality_error'], r['bound_error']) <= 2e-4))
        np.savez_compressed(HERE / 'results' / (job['name'] + '.npz'), unserved=a['unserved'],
                            dsr=a['dsr'], idsr_consumption=a['idsr_consumption'], flow=a['flow'])
    else:
        br, ba, bsha = load_baseline(job['scenario'], w)
        P, S, dsha = load_design(job['scenario'], job['rule'], d)
        rec.update(baseline_sha256=bsha, design_sha256=dsha, baseline_EENS_MWh=float(br['EENS_MWh']))
        if br['EENS_MWh'] <= ZERO_FLOOR_MWh:
            rec.update(success=True, physical_verified=True, analytical_zero_service=True,
                       EENS_MWh=float(br['EENS_MWh']), net_reduction_MWh=0., boost_MWh=0.)
        else:
            allowed = ba['unserved'].sum(axis=1) > HOUR_THRESHOLD_MW
            p = model.build_problem(d, tech='biomethane', power=P, energy=S, gas_cut=sc['cut'],
                                    external_availability=sc['ext'], allow_boost_hours=allowed,
                                    zero_hours=~allowed)
            configure(p, d, ba, P)
            if job['kind'] == 'greedy':
                from greedy import profile
                prof, inv, diag = profile(d, P, S, allowed)
                ids = p['builder'].blocks['boost']
                p['bounds'][ids.ravel(), 0] = prof.ravel()
                p['bounds'][ids.ravel(), 1] = prof.ravel()
                rec.update(greedy_boost_MWh=float(prof.sum()), greedy_diagnostics=diag)
            r = model.solve(p, time_limit=1800, method='copt', secondary=False)
            a = r.pop('arrays')
            net = float(br['EENS_MWh'] - r['EENS_MWh'])
            assert net >= -2e-4, 'Augmentation worsens paired baseline'
            assert r.get('boost_outside_allowed_hours_MWh', 0.) <= 1e-6
            rec.update({k: r[k] for k in ['success', 'EENS_MWh', 'system_shortage_hours', 'boost_MWh',
                                          'equality_error_MWh', 'inequality_error', 'bound_error',
                                          'annual_gas_balance_error_MWh', 'calls', 'allowed_boost_hours']})
            rec.update(net_reduction_MWh=net, power_vector_MW=P.tolist(), buffer_vector_MWh=S.tolist(),
                       zone_boost_MWh=a['boost'].sum(axis=0).tolist(),
                       zone_EENS_MWh=np.maximum(a['unserved'], 0).sum(axis=0).tolist(),
                       physical_verified=bool(max(r['equality_error_MWh'], r['inequality_error'],
                                                  r['bound_error'], r['annual_gas_balance_error_MWh']) <= 2e-4))
            np.savez_compressed(HERE / 'results' / (job['name'] + '.npz'), boost=a['boost'],
                                unserved_hourly=a['unserved'].sum(axis=1), gas_inventory=a['gas_inventory'])
    rec.update(completed_utc=now(), elapsed_seconds=time.time() - started)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    name = ap.parse_args().job
    job = get_job(name)
    out = HERE / 'results' / (name + '.json')
    if out.exists():
        return
    try:
        rec = run(job)
    except Exception:
        rec = dict(job=job, success=False, physical_verified=False, completed_utc=now(),
                   error=traceback.format_exc())
    dump(out, rec)
    print('DONE', name, rec.get('success'), rec.get('EENS_MWh'), rec.get('net_reduction_MWh'), flush=True)


if __name__ == '__main__':
    main()
