"""Weather-sampling uncertainty of the main layout ratio (R1-M2) and its relation
to shortfall concentration (R1-M1), from frozen V32 B3 receipts only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common35 import HERE, V32B, arrays, dump, now, sha

POS = [2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 17, 18, 19, 21, 25, 26, 27, 32, 33, 36]
SEED = 20261102
B = 20000


def net(rule, w):
    p = V32B / 'results' / f'B3_rule_{rule}_WS{w:02d}.json'
    r = json.loads(p.read_text())
    assert r['success'] and r['physical_verified']
    return float(r['net_reduction_MWh']), sha(p)


def main():
    rows, sources = [], {}
    for w in POS:
        e, he = net('eens', w)
        r, hr = net('resource', w)
        sources[f'B3_rule_eens_WS{w:02d}'] = he
        sources[f'B3_rule_resource_WS{w:02d}'] = hr
        b = json.loads((V32B / 'results' / f'baseline_WS{w:02d}.json').read_text())
        un = np.maximum(arrays(V32B / 'results' / f'baseline_WS{w:02d}.npz')['unserved'], 0).sum(axis=0)
        zones = json.loads((V32B / 'results/design_rule_eens.json').read_text())['zones']
        z = np.array(zones)
        de = un[z == 'DE00'].sum() / un.sum()
        uk = un[np.isin(z, ['UK00', 'UKNI'])].sum() / un.sum()
        rows.append(dict(weather=w, eens_net_MWh=e, resource_net_MWh=r, ratio=e / r if r > 1e-6 else np.nan,
                         baseline_EENS_MWh=float(b['EENS_MWh']), DE_share=de, UK_share=uk, DE_UK_share=de + uk))
    df = pd.DataFrame(rows)
    E, R = df.eens_net_MWh.to_numpy(), df.resource_net_MWh.to_numpy()
    point = E.sum() / R.sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(df), size=(B, len(df)))
    boot = E[idx].sum(axis=1) / R[idx].sum(axis=1)
    loo = np.array([(E.sum() - E[i]) / (R.sum() - R[i]) for i in range(len(df))])
    i13 = int(np.flatnonzero(df.weather == 13)[0])
    # Paired weather-level sign test: share of weathers where shortage-aligned siting serves more.
    wins = int((E > R + 1e-6).sum())
    rho, pval = spearmanr(df.DE_UK_share, df.ratio, nan_policy='omit')
    rho_de, p_de = spearmanr(df.DE_share, df.ratio, nan_policy='omit')
    # Ratio within the top/bottom halves of DE+UK concentration.
    med = df.DE_UK_share.median()
    hi, lo = df[df.DE_UK_share >= med], df[df.DE_UK_share < med]
    out = dict(
        created_utc=now(), n_weather=len(df), seed=SEED, bootstrap_resamples=B,
        point_ratio=point,
        bootstrap_percentile_95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        bootstrap_percentile_90=[float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
        bootstrap_share_at_least_2=float((boot >= 2).mean()),
        bootstrap_share_above_1=float((boot > 1).mean()),
        leave_one_out_range=[float(loo.min()), float(loo.max())],
        leave_one_out_min_weather=int(df.weather[int(loo.argmin())]),
        leave_one_out_max_weather=int(df.weather[int(loo.argmax())]),
        ratio_without_WS13=float(loo[i13]),
        WS13_share_of_resource_net=float(R[i13] / R.sum()), WS13_share_of_eens_net=float(E[i13] / E.sum()),
        per_weather_ratio_median=float(df.ratio.median()),
        per_weather_ratio_quartiles=[float(df.ratio.quantile(.25)), float(df.ratio.quantile(.75))],
        per_weather_ratio_min=float(df.ratio.min()), per_weather_ratio_max=float(df.ratio.max()),
        weathers_eens_higher=wins,
        sign_test_p_two_sided=float(2 * sum(__import__('math').comb(len(df), k) for k in range(wins, len(df) + 1)) / 2 ** len(df)),
        DE_UK_share_range=[float(df.DE_UK_share.min()), float(df.DE_UK_share.max())],
        DE_UK_share_median=float(med),
        spearman_ratio_vs_DE_UK_share=dict(rho=float(rho), p=float(pval)),
        spearman_ratio_vs_DE_share=dict(rho=float(rho_de), p=float(p_de)),
        ratio_high_concentration_half=float(hi.eens_net_MWh.sum() / hi.resource_net_MWh.sum()),
        ratio_low_concentration_half=float(lo.eens_net_MWh.sum() / lo.resource_net_MWh.sum()),
        means=dict(eens_mean_MWh=float(E.mean()), resource_mean_MWh=float(R.mean())),
        sources=sources)
    df.to_csv(HERE / 'report/weather_uncertainty_by_weather.csv', index=False)
    np.save(HERE / 'report/weather_bootstrap_ratios.npy', boot)
    dump(HERE / 'report/weather_uncertainty.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'sources'}, indent=1))


if __name__ == '__main__':
    main()
