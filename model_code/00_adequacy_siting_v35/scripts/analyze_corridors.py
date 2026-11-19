"""Import-corridor saturation into DE and UK during their baseline shortfall hours, for every stress boundary.

main: V32 B0 baselines; ext075: V32 A-stage arrays (same electrical LP); cut05ext075 and uk: V35 baselines.
Only held-out weathers with positive baseline shortfall are used, as in the main analysis.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common35 import HERE, V32B, TRAIN, arrays, input_path, dump, now

TOL = 1e-3


def baseline_arrays(scenario, w):
    if scenario == 'main':
        p = V32B / 'results' / f'baseline_WS{w:02d}'
    elif scenario == 'ext075':
        p = V32B.parent / 'results' / f'WS{w:02d}_full_mapped_cut0.3_ext0.75_baseline'
    else:
        p = HERE / 'results' / f'{scenario}_baseline_WS{w:02d}'
    r = json.loads(open(str(p) + '.json').read())
    if r['EENS_MWh'] <= TOL:
        return None
    a = arrays(str(p) + '.npz')
    return a if 'flow' in a else None


def corridors(scenario):
    acc = {}
    weathers = 0
    for w in range(1, 37):
        if w in TRAIN:
            continue
        a = baseline_arrays(scenario, w)
        if a is None:
            continue
        weathers += 1
        d = arrays(input_path(w))
        z = d['zones']; un = np.maximum(a['unserved'], 0)
        for name in ['DE00', 'UK00']:
            k = int(np.flatnonzero(z == name)[0]); hrs = un[:, k] > TOL
            if not hrs.any():
                continue
            for e in np.flatnonzero(d['edge_to'] == k):
                key = (scenario, name, str(z[d['edge_from'][e]]))
                c = acc.setdefault(key, [0, 0])
                c[0] += int((a['flow'][hrs, e] >= d['edge_cap'][hrs, e] - TOL).sum()); c[1] += int(hrs.sum())
    rows = [dict(scenario=k[0], shortfall_zone=k[1], from_zone=k[2], full_hours=v[0], shortfall_hours=v[1],
                 share_hours_full=v[0] / v[1]) for k, v in acc.items()]
    return pd.DataFrame(rows), weathers


def main():
    frames = []
    for sc in sys.argv[1:] or ['main', 'ext075', 'cut05ext075', 'uk']:
        try:
            df, n = corridors(sc)
        except FileNotFoundError:
            print(sc, 'baselines incomplete'); continue
        if len(df):
            frames.append(df.assign(weathers=n))
            key = df[df.from_zone.isin(['FR00', 'BE00', 'DKE1'])]
            print(sc, 'weathers', n, key[['shortfall_zone', 'from_zone', 'shortfall_hours', 'share_hours_full']].round(3).to_string(index=False))
    if frames:
        out = pd.concat(frames)
        out.to_csv(HERE / 'report/v35_corridors_by_boundary.csv', index=False)


if __name__ == '__main__':
    main()
