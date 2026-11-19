"""Summarise V35 receipts: scenario ratios, calibration of new boundaries, UK layouts, no-foresight retention."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common35 import HERE, V32B, TRAIN, arrays, dump, now, sha

REG = json.loads((HERE / 'protocol/jobs_v35.json').read_text())
HOLD = REG['holdout_weather']
OFFICIAL = pd.read_csv(V32B.parent / 'report/official_36_weather_reference.csv')


def rec(name):
    p = HERE / 'results' / (name + '.json')
    if not p.exists():
        return None
    return json.loads(p.read_text())


def calibration(scenario):
    rows = []
    for w in range(1, 37):
        r = rec(f'{scenario}_baseline_WS{w:02d}')
        if r is None or not r.get('success'):
            return None
        rows.append(dict(weather=w, EENS_MWh=r['EENS_MWh'], hours=r.get('system_shortage_hours', 0)))
    df = pd.DataFrame(rows)
    m = df.merge(OFFICIAL[['weather', 'official_EENS_GWh', 'official_system_shortage_hours']], on='weather')
    rho = spearmanr(m.EENS_MWh, m.official_EENS_GWh).correlation
    hr = df.hours.mean() / m.official_system_shortage_hours.mean()
    return dict(mean_EENS_GWh=float(df.EENS_MWh.mean() / 1e3), mean_shortage_hours=float(df.hours.mean()),
                zero_weathers=int((df.EENS_MWh <= 1e-3).sum()),
                hours_ratio_to_official=float(hr),
                EENS_ratio_to_official=float(df.EENS_MWh.mean() / 1e3 / m.official_EENS_GWh.mean()),
                spearman=float(rho), criterion_met=bool(0.5 <= hr <= 2 and rho >= 0.6))


def geography(scenario):
    tot = None
    zones = None
    for w in range(1, 37):
        r = rec(f'{scenario}_baseline_WS{w:02d}') if scenario != 'main' else json.loads(
            (V32B / 'results' / f'baseline_WS{w:02d}.json').read_text())
        if r is None:
            return None
        if scenario == 'main':
            if r['EENS_MWh'] <= 1e-3:
                continue
            ze = np.maximum(arrays(V32B / 'results' / f'baseline_WS{w:02d}.npz')['unserved'], 0).sum(axis=0)
        else:
            ze = np.asarray(r['zone_EENS_MWh'])
        zones = r.get('zones', zones)
        tot = ze if tot is None else tot + ze
    if zones is None:
        zones = json.loads((V32B / 'results/design_rule_eens.json').read_text())['zones']
    z = np.array(zones)
    s = tot / tot.sum()
    return dict(DE=float(s[z == 'DE00'].sum()), UK=float(s[np.isin(z, ['UK00', 'UKNI'])].sum()),
                other=float(1 - s[z == 'DE00'].sum() - s[np.isin(z, ['UK00', 'UKNI'])].sum()),
                total_EENS_GWh=float(tot.sum() / 1e3))


def ratio(scenario, kind='eval'):
    rows = []
    for w in HOLD:
        e = rec(f'{scenario}_{kind}_rule_eens_WS{w:02d}')
        r = rec(f'{scenario}_{kind}_rule_resource_WS{w:02d}')
        if e is None or r is None or not e.get('success') or not r.get('success'):
            return None, None
        rows.append(dict(weather=w, eens_net_MWh=e['net_reduction_MWh'], resource_net_MWh=r['net_reduction_MWh'],
                         baseline_EENS_MWh=e['baseline_EENS_MWh'],
                         eens_boost_MWh=e.get('boost_MWh', 0.), resource_boost_MWh=r.get('boost_MWh', 0.)))
    df = pd.DataFrame(rows)
    pos = df[df.baseline_EENS_MWh > 1e-3]
    E, R = pos.eens_net_MWh.to_numpy(), pos.resource_net_MWh.to_numpy()
    rng = np.random.default_rng(20261102)
    idx = rng.integers(0, len(pos), size=(20000, len(pos)))
    boot = E[idx].sum(axis=1) / np.maximum(R[idx].sum(axis=1), 1e-12)
    out = dict(n_positive_holdout=int(len(pos)), eens_mean_MWh=float(E.mean()), resource_mean_MWh=float(R.mean()),
               ratio=float(E.sum() / R.sum()), bootstrap95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               weathers_eens_higher=int((E > R + 1e-6).sum()),
               share_of_baseline_33holdout_eens=float(df.eens_net_MWh.sum() / df.baseline_EENS_MWh.sum()))
    return out, df


def greedy():
    rows = []
    for rule in ['eens', 'resource']:
        for w in REG['main_positive_holdout']:
            g = rec(f'main_greedy_rule_{rule}_WS{w:02d}')
            if g is None or not g.get('success'):
                return None, None
            lp = json.loads((V32B / 'results' / f'B3_rule_{rule}_WS{w:02d}.json').read_text())
            rows.append(dict(rule=rule, weather=w, lp_net_MWh=lp['net_reduction_MWh'], greedy_net_MWh=g['net_reduction_MWh'],
                             lp_boost_MWh=lp['boost_MWh'], greedy_boost_MWh=g['boost_MWh']))
    df = pd.DataFrame(rows)
    out = {}
    for rule, g in df.groupby('rule'):
        out[rule] = dict(lp_sum=float(g.lp_net_MWh.sum()), greedy_sum=float(g.greedy_net_MWh.sum()),
                         retained=float(g.greedy_net_MWh.sum() / g.lp_net_MWh.sum()),
                         min_weather_retained=float((g.greedy_net_MWh / g.lp_net_MWh.where(g.lp_net_MWh > 1e-6)).min()),
                         weathers_full=int(((g.greedy_net_MWh >= g.lp_net_MWh - 1e-3) | (g.lp_net_MWh < 1e-6)).sum()),
                         greedy_mean=float(g.greedy_net_MWh.mean()), lp_mean=float(g.lp_net_MWh.mean()))
    e, r = df[df.rule == 'eens'].greedy_net_MWh.sum(), df[df.rule == 'resource'].greedy_net_MWh.sum()
    out['greedy_ratio'] = float(e / r)
    return out, df


def uk_layouts():
    out = {}
    for rule in ['eens', 'resource']:
        p = HERE / 'results' / f'design_uk_rule_{rule}.json'
        if not p.exists():
            return None
        v = json.loads(p.read_text())
        z = np.array(v['zones']); P = np.array(v['power'])
        out[rule] = dict(DE_MW=float(P[z == 'DE00'].sum()), UK_MW=float(P[np.isin(z, ['UK00', 'UKNI'])].sum()),
                         total_MW=float(P.sum()), zones=v['zones'], power=v['power'])
    return out


def boundary_geography():
    """Shortfall geography for every stress combination: V32 A1/A2 baselines plus V35 baselines."""
    A = V32B.parent / 'results'
    zones = np.array(json.loads((V32B / 'results/design_rule_eens.json').read_text())['zones'])
    rows = []
    combos = [('main', 0.3, 0.5, 'stress'), ('cut0.3_ext0.75', 0.3, 0.75, 'cut0.3_ext0.75'),
              ('cut0.3_ext1', 0.3, 1.0, 'cut0.3_ext1'), ('cut0_ext0.5', 0.0, 0.5, 'cut0_ext0.5'),
              ('cut0_ext0.75', 0.0, 0.75, 'cut0_ext0.75')]
    for label, cut, ext, key in combos:
        tot = np.zeros(len(zones)); hours = []
        for w in range(1, 37):
            r = json.loads((A / f'WS{w:02d}_full_mapped_{key}_baseline.json').read_text())
            hours.append(r.get('system_shortage_hours', 0))
            if r['EENS_MWh'] > 1e-3:
                tot += np.maximum(arrays(A / f'WS{w:02d}_full_mapped_{key}_baseline.npz')['unserved'], 0).sum(axis=0)
        rows.append((label, cut, ext, tot, float(np.mean(hours)), 'V32 A-stage'))
    for label, sc, cut, ext in [('cut0.5_ext0.75', 'cut05ext075', 0.5, 0.75), ('main + UK resource', 'uk', 0.3, 0.5)]:
        tot = np.zeros(len(zones)); hours = []
        for w in range(1, 37):
            r = rec(f'{sc}_baseline_WS{w:02d}')
            if r is None or not r.get('success'):
                tot = None
                break
            tot += np.asarray(r['zone_EENS_MWh']); hours.append(r.get('system_shortage_hours', 0))
        if tot is not None:
            rows.append((label, cut, ext, tot, float(np.mean(hours)), 'V35'))
    out = []
    for label, cut, ext, tot, h, origin in rows:
        s = tot / tot.sum() if tot.sum() > 0 else tot
        de = float(s[zones == 'DE00'].sum()); uk = float(s[np.isin(zones, ['UK00', 'UKNI'])].sum())
        out.append(dict(boundary=label, gas_cut=cut, external_availability=ext, mean_EENS_GWh=float(tot.sum() / 36 / 1e3),
                        mean_shortage_hours=h, DE_share=de, UK_share=uk, other_share=1 - de - uk, source=origin))
    df = pd.DataFrame(out)
    df.to_csv(HERE / 'report/v35_boundary_geography.csv', index=False)
    return df


def main():
    summary = dict(created_utc=now(), scenarios={})
    summary['boundary_geography'] = boundary_geography().to_dict('records')
    frames = []
    for sc in ['ext075', 'cut05ext075', 'uk']:
        r, df = ratio(sc)
        summary['scenarios'][sc] = dict(ratio=r, geography=geography(sc),
                                        calibration=calibration(sc) if sc != 'uk' else None)
        if df is not None:
            frames.append(df.assign(scenario=sc))
    summary['scenarios']['main'] = dict(geography=geography('main'))
    g, gdf = greedy()
    summary['no_foresight'] = g
    summary['uk_layouts'] = uk_layouts()
    if frames:
        pd.concat(frames).to_csv(HERE / 'report/v35_scenario_by_weather.csv', index=False)
    if gdf is not None:
        gdf.to_csv(HERE / 'report/v35_no_foresight_by_weather.csv', index=False)
    dump(HERE / 'report/v35_summary.json', summary)
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
