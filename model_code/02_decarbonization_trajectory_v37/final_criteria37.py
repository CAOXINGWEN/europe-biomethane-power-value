"""Final T1-T5 and E1 verdicts from the completed pair tables (main specification F1/F2; registered B1 for T1-T3, T5, E1).

Criteria as registered in protocol/registration_v37.json (leakage 2.5 %, GWP100, h = 0, 15-country system, 36-weather means).
Output: report/final_criteria_v37.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]; R = HERE / 'report'
Y = [2028, 2030, 2033, 2035]; LEAK = .025 * .072 * 27; HEAT = .24302 * .488 / .9


def u1_checks(df, spec):
    out = []; g = df.groupby('year'); D = g.D.mean(); av = g.avoided.mean(); sf = g.s_f.mean(); m = av / (.4 * sf)
    piv = df.pivot(index='weather', columns='year', values='D')
    mono = int(((piv[2028] > piv[2030]) & (piv[2030] > piv[2033]) & (piv[2033] > piv[2035])).sum()); pos = int((piv[2028] - piv[2035] > 0).sum())
    out.append(dict(spec=spec, item='T1', value=f'D {D[2028]:.3f}/{D[2030]:.3f}/{D[2033]:.3f}/{D[2035]:.3f}; monotone weathers {mono}/36; D2028-D2035>0 in {pos}/36',
                    verdict=bool(D[2028] > D[2030] > D[2033] > D[2035] and pos >= 32)))
    net = av - LEAK; be = av / (.072 * 27)
    out.append(dict(spec=spec, item='T2', value=f'net {net[2028]:+.4f} -> {net[2035]:+.4f}; break-even {be[2028]:.2%} -> {be[2035]:.2%}', verdict=bool(net[2028] > 0 > net[2035])))
    sa, sb, ma, mb = sf[2028], sf[2035], m[2028], m[2035]; sh = .4 * (sb - sa) * (ma + mb) / 2; it = .4 * (mb - ma) * (sa + sb) / 2
    out.append(dict(spec=spec, item='T3', value=f'fossil-share term {sh / (sh + it):.1%} of the 2028-2035 change', verdict=bool(sh / (sh + it) > .5)))
    mef = av / .4; inj = .24302 - .03 * mef; h = (inj - av) / HEAT
    out.append(dict(spec=spec, item='T5', value=f'h* {h[2028]:.2f} (2028), {h[2035]:.2f} (2035)', verdict=bool(h[2028] <= 1 < h[2035])))
    injnet = inj - LEAK
    out.append(dict(spec=spec, item='E1', value='injection minus U1 net: ' + ', '.join(f'{y} {injnet[y] - net[y]:+.3f}' for y in Y), verdict=bool(all(injnet[y] > net[y] for y in Y))))
    return out


def main():
    f1 = pd.read_csv(R / 'preliminary_F1_by_weather.csv'); b1 = pd.read_csv(R / 'preliminary_B1_by_weather.csv'); f2 = pd.read_csv(R / 'preliminary_F2_by_weather.csv')
    assert all(len(x[x.year == y]) == 36 for x in (f1, b1, f2) for y in Y)
    rows = u1_checks(f1, 'main') + u1_checks(b1, 'registered')
    g = f2.groupby('year'); ratio = g.dEENS.mean() * 12240 / g.fuel_saving.mean()
    rows.append(dict(spec='main', item='T4', value='adequacy/fuel-saving value ' + ' -> '.join(f'{ratio[y]:.2f}' for y in Y), verdict=bool(ratio[2035] > ratio[2028])))
    T = pd.DataFrame(rows); T.to_csv(R / 'final_criteria_v37.csv', index=False); print(T.to_string(index=False))


if __name__ == '__main__': main()
