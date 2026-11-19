"""Compare F4 checks and F5 sensitivities with the main-specification baseline on the same weather years.

Baselines: F1 (15-country site power, Central) from report/preliminary_F1_by_weather.csv; F3 single-country 2028 for L3.
Output: report/final_F4F5_summary.csv (per variant: n, D and baseline D, paired difference, net abatement, balance shares).
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]
LEAK = .025 * .072 * 27
SH = ['gas', 'Nuclear', 'Lignite', 'Hard coal', 'external_import', 'hydrogen']; INC = ['curtailment', 'electrolyser_consumption', 'power_to_heat_consumption']


def shares(df, pre):
    o = {f'gas': df[f'{pre}reduced_gas'].mean(), 'coal_lignite': (df[f'{pre}reduced_Lignite'] + df[f'{pre}reduced_Hard coal']).mean(),
         'nuclear': df[f'{pre}reduced_Nuclear'].mean(), 'imports': df[f'{pre}reduced_external_import'].mean(), 'curtailment': df[f'{pre}increased_curtailment'].mean(),
         'electrolyser_p2h': (df[f'{pre}increased_electrolyser_consumption'] + df[f'{pre}increased_power_to_heat_consumption']).mean()}
    return o


def main():
    S = pd.read_csv(HERE / 'report/final_F4F5_by_case.csv'); f1 = pd.read_csv(HERE / 'report/preliminary_F1_by_weather.csv')
    f3 = pd.read_csv(HERE / 'report/preliminary_F3_by_case.csv')
    V = [('L1 V36 resource vector', (S.batch == 'F4') & (S.resource == 'V36Central'), 2028, None),
         ('L2 High resource', (S.batch == 'F4') & (S.resource == 'High'), 2028, None),
         ('L3 DE x5', (S.batch == 'F4') & (S.country == 'DE'), 2028, 'DE'), ('L3 SI x5', (S.batch == 'F4') & (S.country == 'SI'), 2028, 'SI'),
         ('S-nuc nuclear fixed', S.sens == 'nuclear_fixed', 2028, None),
         ('S-gas no gas cut', (S.sens == 'no_gascut') & (S.year == 2028), 2028, None), ('S-gas no gas cut', (S.sens == 'no_gascut') & (S.year == 2035), 2035, None),
         ('S-fleet existing biogas', (S.resource == 'fleet') & (S.year == 2028), 2028, None), ('S-fleet existing biogas', (S.resource == 'fleet') & (S.year == 2035), 2035, None)]
    rows = []
    for lab, m, y, c in V:
        v = S[m].set_index('weather')
        if c is None: b = f1[f1.year == y].set_index('weather').loc[v.index]; bsh = shares(b, '')
        else:
            b = f3[(f3.year == y) & (f3.country == c)].set_index('weather').loc[v.index]; bsh = shares(b, 'share_')
            b = b.assign(scale=b.scale)
        dd = v.D - b.D
        rows.append(dict(variant=lab, year=y, n=len(v), weathers=' '.join(str(int(w)) for w in v.index), B_TWh=v.B.mean() / 1e6, scale=v.scale.mean(),
                         base_scale=b.scale.mean() if 'scale' in b else 1.0, D=v.D.mean(), D_base=b.D.mean(), dD=dd.mean(), dD_min=dd.min(), dD_max=dd.max(),
                         rel=v.D.mean() / b.D.mean() - 1, net=(v.avoided - LEAK).mean(), net_base=(b.avoided - LEAK).mean(),
                         s_f=v.s_f.mean(), s_f_base=b.s_f.mean(), dEENS_MWh=v.dEENS.mean(),
                         **{f'sh_{k}': x for k, x in shares(v, 'share_').items()}, **{f'base_sh_{k}': x for k, x in bsh.items()}))
    T = pd.DataFrame(rows); T.to_csv(HERE / 'report/final_F4F5_summary.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(T[['variant', 'year', 'n', 'B_TWh', 'scale', 'base_scale', 'D', 'D_base', 'dD', 'dD_min', 'dD_max', 'rel', 'net', 'net_base']].round(4).to_string(index=False))
    print(T[['variant', 'year'] + [c for c in T.columns if c.startswith('sh_') or c.startswith('base_sh_')]].round(3).to_string(index=False))


if __name__ == '__main__': main()
