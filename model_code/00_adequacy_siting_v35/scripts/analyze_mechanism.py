"""Deliverability mechanism behind the stable layout ratio, from frozen V32 arrays.

1. Import-interface saturation of shortfall zones in their baseline shortfall hours.
2. Net reduction per MWh of boost by layout and by weather.
3. Where boost is produced, split into shortfall zones, their direct neighbours and other zones.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common35 import HERE, V32B, arrays, dump, now, input_path

POS = [2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 17, 18, 19, 21, 25, 26, 27, 32, 33, 36]
TOL = 1e-3


def main():
    rows, sat_rows, corr = [], [], {}
    for w in POS:
        d = arrays(input_path(w))
        z = d['zones']
        ba = arrays(V32B / 'results' / f'baseline_WS{w:02d}.npz')
        un = np.maximum(ba['unserved'], 0)
        flow, ef, et = ba['flow'], d['edge_from'], d['edge_to']
        cap = d['edge_cap']
        ext = ba['external_import']
        extcap = d['external_cap'] * 0.5
        for name in ['DE00', 'UK00']:
            k = int(np.flatnonzero(z == name)[0])
            hrs = un[:, k] > TOL
            if hrs.sum() == 0:
                continue
            inb = np.flatnonzero(et == k)
            outb = np.flatnonzero(ef == k)
            inflow = flow[:, inb].sum(axis=1)
            outflow = flow[:, outb].sum(axis=1)
            netimp = inflow + ext[:, k] - outflow
            # Each inbound line at its limit, or the zonal net-import limit binding.
            line_sat = np.all(flow[:, inb] >= cap[:, inb] - 1e-3, axis=1)
            net_bind = netimp >= d['net_import_cap'][:, k] - 1e-3
            gross_bind = inflow + ext[:, k] >= d['gross_import_cap'][:, k] - 1e-3
            ext_sat = ext[:, k] >= extcap[:, k] - 1e-3
            any_limit = line_sat | net_bind | gross_bind
            for e in inb:
                key = (name, str(z[ef[e]]))
                c = corr.setdefault(key, dict(full_hours=0, hours=0, cap_sum=0., flow_sum=0.))
                c['full_hours'] += int((flow[hrs, e] >= cap[hrs, e] - 1e-3).sum())
                c['hours'] += int(hrs.sum())
                c['cap_sum'] += float(cap[hrs, e].sum())
                c['flow_sum'] += float(flow[hrs, e].sum())
            sat_rows.append(dict(weather=w, zone=name, shortfall_hours=int(hrs.sum()),
                                 share_hours_import_limited=float(any_limit[hrs].mean()),
                                 share_hours_all_lines_full=float(line_sat[hrs].mean()),
                                 share_hours_net_import_limit=float(net_bind[hrs].mean()),
                                 share_hours_gross_import_limit=float(gross_bind[hrs].mean()),
                                 share_hours_external_full=float(ext_sat[hrs].mean()),
                                 shortfall_MWh=float(un[hrs, k].sum()),
                                 shortfall_MWh_in_limited_hours=float(un[hrs & any_limit, k].sum())))
        short = un.sum(axis=0) > TOL
        neigh = np.zeros(len(z), bool)
        for a, b in zip(ef, et):
            if short[b]:
                neigh[a] = True
            if short[a]:
                neigh[b] = True
        neigh &= ~short
        for rule in ['eens', 'resource']:
            r = json.loads((V32B / 'results' / f'B3_rule_{rule}_WS{w:02d}.json').read_text())
            a = arrays(V32B / 'results' / f'B3_rule_{rule}_WS{w:02d}.npz')
            zb = a['boost'].sum(axis=0)
            rows.append(dict(weather=w, rule=rule, boost_MWh=float(zb.sum()), net_MWh=float(r['net_reduction_MWh']),
                             net_per_boost=float(r['net_reduction_MWh'] / zb.sum()) if zb.sum() > 1e-6 else np.nan,
                             boost_in_shortfall_zones=float(zb[short].sum()), boost_in_neighbours=float(zb[neigh].sum()),
                             boost_elsewhere=float(zb[~short & ~neigh].sum())))
    df, sat = pd.DataFrame(rows), pd.DataFrame(sat_rows)
    cdf = pd.DataFrame([dict(shortfall_zone=k[0], from_zone=k[1], shortfall_hours=v['hours'],
                             share_hours_full=v['full_hours'] / v['hours'], mean_capacity_MW=v['cap_sum'] / v['hours'],
                             mean_flow_MW=v['flow_sum'] / v['hours']) for k, v in sorted(corr.items())])
    cdf.to_csv(HERE / 'report/mechanism_corridors.csv', index=False)
    df.to_csv(HERE / 'report/mechanism_boost_by_weather.csv', index=False)
    sat.to_csv(HERE / 'report/mechanism_import_limits.csv', index=False)
    summ = {}
    for rule, g in df.groupby('rule'):
        tb = g.boost_MWh.sum()
        summ[rule] = dict(total_boost_MWh=float(tb), total_net_MWh=float(g.net_MWh.sum()),
                          pooled_net_per_boost=float(g.net_MWh.sum() / tb),
                          per_weather_net_per_boost_median=float(g.net_per_boost.median()),
                          boost_share_shortfall_zones=float(g.boost_in_shortfall_zones.sum() / tb),
                          boost_share_neighbours=float(g.boost_in_neighbours.sum() / tb),
                          boost_share_elsewhere=float(g.boost_elsewhere.sum() / tb))
    for name, g in sat.groupby('zone'):
        summ['import_limits_' + name] = dict(
            weathers=int(len(g)), shortfall_hours=int(g.shortfall_hours.sum()),
            share_shortfall_hours_import_limited=float((g.share_hours_import_limited * g.shortfall_hours).sum() / g.shortfall_hours.sum()),
            share_shortfall_energy_in_limited_hours=float(g.shortfall_MWh_in_limited_hours.sum() / g.shortfall_MWh.sum()),
            share_hours_all_lines_full=float((g.share_hours_all_lines_full * g.shortfall_hours).sum() / g.shortfall_hours.sum()),
            share_hours_net_import_limit=float((g.share_hours_net_import_limit * g.shortfall_hours).sum() / g.shortfall_hours.sum()),
            share_hours_gross_import_limit=float((g.share_hours_gross_import_limit * g.shortfall_hours).sum() / g.shortfall_hours.sum()))
    summ['created_utc'] = now()
    dump(HERE / 'report/mechanism_summary.json', summ)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
