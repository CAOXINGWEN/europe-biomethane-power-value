"""Extended Data Figures 1-10 (figED1 to figED10) and Supplementary Fig. S1 for manuscript revision v17."""
import sys
import shutil
import hashlib
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import PchipInterpolator
from matplotlib import colors
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from style import (
    plt, fig, ax, label, grid, yearaxis, band, textheat, repel, laea,
    countrymap, export, read, C, YC, YC_WARM, SEQ, BLUE, WARM, DIV,
    INK, MUTED, GREY, GRID, FIG, QA, SD, V, SOURCES, ribbon
)

YEARS = [2028, 2030, 2033, 2035]


def read52(name: str) -> pd.DataFrame:
    p = SD / 'site52' / f'{name}.csv'
    SOURCES[f'site52/{p.name}'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p)


def finish(f, axes, name, letters='abcdefghij'):
    for a_obj, ltr in zip(axes, letters):
        label(f, a_obj, ltr)
    export(f, name)


def ed1_system_resources():
    cap = read('eraa_capacities_24zones').set_index('technology')
    cf = read('counterfactual')
    pr = read('eraa_dispatch_costs')
    lmdi = read('resource_lmdi')
    res = read('resource')
    bio = read('biogas_end_use')

    f = fig(188)
    a = ax(f, 14, 8, 53, 42)
    b = ax(f, 82, 8, 44, 42)
    c = ax(f, 141, 8, 37, 42)

    d = ax(f, 14, 66, 44, 42)
    e = ax(f, 74, 66, 48, 42)
    f_ax = ax(f, 136, 66, 42, 42)

    g = ax(f, 3, 122, 78, 46)
    h = ax(f, 99, 128, 76, 42)

    # --- a: 24-zone installed capacity ---
    order_cap = [
        ('Coal and lignite', C['brown']), ('Nuclear', C['purple']), ('Gas', C['power']),
        ('Hydrogen turbines', C['sand']), ('Batteries', C['blue']), ('Wind', '#7BA7CA'), ('Solar', '#E5C365')
    ]
    xf = np.linspace(2028, 2035, 120)
    vals_c = {k: PchipInterpolator(YEARS, cap.loc[k, [str(y) for y in YEARS]].to_numpy())(xf) for k, _ in order_cap}
    base = np.zeros_like(xf)
    mids = []
    for k, col in order_cap:
        a.fill_between(xf, base, base + vals_c[k], color=col, lw=0, alpha=0.95)
        a.plot(xf, base + vals_c[k], color='white', lw=0.3)
        mids.append(base[-1] + vals_c[k][-1] / 2.0)
        base = base + vals_c[k]
    ly = repel(mids, 145.0)
    ly = ly + max(0.0, 55.0 - ly.min())
    short_n = {'Coal and lignite': 'Coal', 'Hydrogen turbines': 'H$_2$', 'Batteries': 'Batt.'}
    for (k, _), m_, l_ in zip(order_cap, mids, ly):
        a.plot([2035, 2035.4], [m_, l_], color=GREY, lw=0.45)
        a.text(2035.5, l_, f'{short_n.get(k, k)} {cap.loc[k, "2035"]:.0f}', fontsize=7.0, va='center', ha='left')
    a.set_xlim(2028, 2039.8)
    a.set_ylim(0, 1780)
    a.set_xticks(YEARS)
    a.set_xticklabels(["'28", "'30", "'33", "'35"])
    a.set_xlabel('Target year')
    a.set_ylabel('Installed capacity, 24 zones (GW)')

    # --- b: Counterfactual curtailment rate vs fossil share (36 weathers x 4 yrs) ---
    for y in YEARS:
        t = cf[cf.year == y]
        b.scatter(100.0 * t.curtail_rate, 100.0 * t.fossil_share, s=7, color=YC_WARM[y], alpha=0.55, lw=0, zorder=2)
    mn = cf.groupby('year')[['curtail_rate', 'fossil_share']].mean() * 100.0
    b.plot(mn.curtail_rate, mn.fossil_share, color=INK, lw=1.0, zorder=3)
    for y in YEARS:
        b.scatter(mn.loc[y, 'curtail_rate'], mn.loc[y, 'fossil_share'], s=24, color=YC_WARM[y], edgecolor=INK, lw=0.55, zorder=4)
        dx, dy, ha = {2028: (0.45, 0.2, 'left'), 2030: (-0.45, -0.3, 'right'), 2033: (0.4, 0.4, 'left'), 2035: (0.45, 0.0, 'left')}[y]
        b.text(mn.loc[y, 'curtail_rate'] + dx, mn.loc[y, 'fossil_share'] + dy, str(y), fontsize=7.0, ha=ha, va='center')
    b.set_xlim(2.5, 13.0)
    b.set_ylim(0, 16.5)
    b.set_xlabel('Curtailment (% of VRE)')
    b.set_ylabel('Fossil share of demand (%)')
    grid(b)

    # --- c: Dispatch cost crossing ---
    for col_k, lab_c, col_c, ls_c in [
        ('hard_coal_EUR_per_MWhe', 'Hard coal', '#2D2A28', '-'),
        ('lignite_G2_EUR_per_MWhe', 'Lignite', C['brown'], '--'),
        ('gas_CCGT_EUR_per_MWhe', 'Gas CCGT', C['power'], '-'),
        ('hydrogen_CCGT_EUR_per_MWhe', 'H$_2$ CCGT', C['sand'], '-'),
    ]:
        c.plot(pr.year, pr[col_k], color=col_c, ls=ls_c, lw=1.2, marker='o', ms=2.6, label=lab_c)
    c.set_xticks(YEARS)
    c.set_xticklabels(["'28", "'30", "'33", "'35"])
    c.set_ylim(60, 225)
    c.set_xlabel('Target year')
    c.set_ylabel('Dispatch cost (EUR/MWh$_e$)')
    c.legend(loc='upper left', fontsize=7.0, handlelength=1.2, labelspacing=0.2)
    grid(c)

    # --- d: LMDI 2022->2035 Central resource waterfall ---
    L = lmdi[(lmdi.scenario == 'Central') & (lmdi.year == 2035)][['R_base_2022_TWh', 'R_year_TWh', 'population_TWh', 'prevention_TWh', 'capture_TWh']].sum()
    steps = [('2022', L.R_base_2022_TWh, 'tot'), ('Pop.', L.population_TWh, 'd'), ('Prev.', L.prevention_TWh, 'd'),
             ('Coll.', L.capture_TWh, 'd'), ('2035', L.R_year_TWh, 'tot')]
    lvl = 0.0
    for i, (lab_s, v_s, kind_s) in enumerate(steps):
        if kind_s == 'tot':
            d.add_patch(Rectangle((i - 0.34, 0), 0.68, v_s, facecolor=C['navy'], edgecolor='none'))
            lvl = v_s
            top = v_s
        else:
            d.add_patch(Rectangle((i - 0.34, lvl), 0.68, v_s, facecolor=C['teal'] if v_s > 0 else C['power'], edgecolor='none'))
            top = max(lvl, lvl + v_s)
            lvl += v_s
        d.text(i, top + 0.18, f'{v_s:+.2f}' if kind_s == 'd' else f'{v_s:.2f}', fontsize=7.0, ha='center', va='bottom')
        if i < 4:
            d.plot([i + 0.34, i + 0.66], [lvl] * 2, color=GREY, lw=0.5)
    d.set_xticks(range(5), [s_[0] for s_ in steps])
    d.set_xlim(-0.6, 4.6)
    d.set_ylim(0, 9.6)
    d.set_ylabel('15-country resource (TWh)')
    grid(d)

    # --- e: Deliverable biomethane under Low, Central, High ---
    tot_r = res[res.scenario.isin(['Low', 'Central', 'High'])].groupby(['scenario', 'year']).deliverable_TWh.sum().unstack()
    xr = [2022] + YEARS
    b22 = L.R_base_2022_TWh
    lo_r = [b22] + list(tot_r.loc['Low', YEARS])
    mid_r = [b22] + list(tot_r.loc['Central', YEARS])
    hi_r = [b22] + list(tot_r.loc['High', YEARS])
    e.fill_between(xr, lo_r, hi_r, color=C['teal'], alpha=0.14, lw=0)
    for vv, lab_r, ls_r in [(hi_r, 'High 12.6', '--'), (mid_r, 'Central 7.3', '-'), (lo_r, 'Low 3.0', ':')]:
        e.plot(xr, vv, color=C['teal'], ls=ls_r, lw=1.2, marker='o', ms=2.6)
        e.text(2035.3, vv[-1], lab_r, fontsize=7.0, va='center', ha='left', color=C['teal'])
    e.set_xlim(2021.5, 2040.2)
    e.set_ylim(0, 14.5)
    e.set_xticks(xr)
    e.set_xticklabels(["'22", "'28", "'30", "'33", "'35"])
    e.set_xlabel('Year')
    e.set_ylabel('Deliverable biomethane (TWh)')
    grid(e)

    # --- f: Country capture rate & per-capita resource arrows (2022 -> 2035) ---
    cc_r = res[res.scenario == 'Central']
    rows_c = []
    for cc_code in cc_r.country_code.unique():
        a_r = cc_r[(cc_r.country_code == cc_code) & (cc_r.year == 2035)].iloc[0]
        kwh35 = a_r.deliverable_TWh * 1e9 / a_r.population
        kwh22 = kwh35 / a_r.r * (a_r.k0 / a_r.k) if a_r.k > 0 else 0.0
        rows_c.append(dict(country=cc_code, k0=a_r.k0, k35=a_r.k, kwh22=kwh22, kwh35=kwh35))
    df_c = pd.DataFrame(rows_c)
    for r_ in df_c.itertuples():
        f_ax.add_patch(FancyArrowPatch((100.0 * r_.k0, r_.kwh22), (100.0 * r_.k35, r_.kwh35),
                                       arrowstyle='-|>', mutation_scale=6, lw=0.75, color=C['teal'], zorder=3))
        f_ax.scatter(100.0 * r_.k0, r_.kwh22, s=9, color=GREY, zorder=4)
    for cc_k, (xp, yp) in [('IT', (75, 38)), ('DK', (63, 25)), ('DE', (53, 22)), ('GB', (46, 16)), ('FR', (46, 11)), ('ES', (44, 4.5))]:
        f_ax.text(xp, yp, 'UK' if cc_k == 'GB' else cc_k, fontsize=7.0, weight='bold', va='center')
    f_ax.set_xlim(-3, 84)
    f_ax.set_ylim(0, 52)
    f_ax.set_xlabel('Separate collection rate (%)')
    f_ax.set_ylabel('Per-capita resource (kWh/yr)')
    grid(f_ax)

    # --- g: 2035 Central food-waste biomethane European map ---
    g_r = res[(res.year == 2035) & (res.scenario == 'Central')].set_index('country_code')
    vals_m = {('UK' if k == 'GB' else k): v for k, v in g_r.deliverable_TWh.to_dict().items()}
    sm = countrymap(g, vals_m, cmap=BLUE, vmin=0.0, vmax=max(vals_m.values()))
    cb_g = f.colorbar(sm, cax=ax(f, 12, 171, 60, 2.5), orientation='horizontal')
    cb_g.set_label('2035 food-waste biomethane (TWh)', fontsize=7.0)
    cb_g.ax.tick_params(labelsize=7.0)

    # --- h: Eurostat 2024 all-biogas energy balance ---
    geos = ['EU27_2020', 'DE', 'IT', 'FR', 'DK']
    bb = bio[bio.geo.isin(geos) & (bio.year == 2024)].set_index('geo').loc[geos]
    parts = [('power_only_GWh', 'Power only', C['blue']), ('chp_GWh', 'CHP', C['purple']),
             ('blended_into_gas_grid_GWh', 'Gas grid', C['teal']), ('final_consumption_GWh', 'Final use', C['sand'])]
    bot = np.zeros(5)
    for key_p, lab_p, col_p in parts:
        vv = bb[key_p] / bb.gross_inland_GWh * 100.0
        h.bar(np.arange(5), vv, bottom=bot, color=col_p, width=0.65, label=lab_p)
        bot += vv.to_numpy()
    h.bar(np.arange(5), 100.0 - bot, bottom=bot, color=C['light'], width=0.65, label='Other')
    h.set_xticks(range(5), ['EU27', 'DE', 'IT', 'FR', 'DK'])
    h.set_ylabel('All-biogas energy balance (%)')
    h.set_ylim(0, 100)
    h.legend(loc='lower left', bbox_to_anchor=(-0.05, 1.02), ncol=3, fontsize=7.0, handlelength=0.8, columnspacing=0.7)

    finish(f, [a, b, c, d, e, f_ax, g, h], 'figED1_system_resources')


def ed2_weather_methane():
    d = read('site_sensitivities')
    cf = read('counterfactual')
    cs = read('crossing_sensitivity')

    f = fig(185)
    a = ax(f, 16, 10, 62, 152)
    b = ax(f, 106, 10, 72, 34)
    c = ax(f, 106, 58, 72, 34)
    d_ax = ax(f, 106, 106, 72, 32)
    e = ax(f, 106, 150, 72, 26)

    mat = d.pivot(index='weather', columns='year', values='D')
    im = a.imshow(mat, cmap=BLUE, aspect='auto', vmin=0, vmax=0.6)
    a.set_xticks(range(4), mat.columns.astype(str))
    a.set_yticks(range(0, 36, 2), [f'{w:02}' for w in mat.index[::2]])
    a.set_ylabel('Weather identifier (01–36)')
    a.set_xlabel('Target year')
    a.tick_params(length=0)
    cb = f.colorbar(im, cax=ax(f, 16, 173, 62, 2.5), orientation='horizontal')
    cb.set_label('Fossil fuel displaced (MWh/MWh)', fontsize=7.0)
    cb.ax.tick_params(labelsize=7.0)

    for leak, ls, col in [('0.01', '-', C['teal']), ('0.025', '--', C['blue']), ('0.042', ':', C['red'])]:
        band(b, d, 'net_GWP100_leak' + leak, col, f'{float(leak)*100:g}%', ls=ls)
    b.axhline(0, color=GREY, lw=0.6)
    b.set_ylabel('Net abatement\n(t CO$_2$e/MWh)')
    yearaxis(b, short=True)
    b.legend(ncol=3, fontsize=7.0, columnspacing=0.7, loc='upper right')
    grid(b)

    for key, col, lab in [('net_GWP100_leak0.025', C['blue'], 'GWP$_{100}$'), ('net_GWP20_leak0.025', C['red'], 'GWP$_{20}$')]:
        band(c, d, key, col, lab)
    c.axhline(0, color=GREY, lw=0.6)
    yearaxis(c, short=True)
    c.set_ylabel('Net abatement\n(t CO$_2$e/MWh)')
    c.legend(fontsize=7.0, ncol=2, loc='upper right')
    grid(c)

    z = d.merge(cf[['year', 'weather', 'curtail_rate']], on=['year', 'weather'])
    for y in YEARS:
        gy = z[z.year == y]
        d_ax.scatter(gy.curtail_rate * 100.0, gy.D, s=9, color=YC_WARM[y], label=str(y), lw=0)
    k_fit = np.polyfit(z.curtail_rate * 100.0, z.D, 1)
    xx_f = np.linspace(3.5, 12.0, 50)
    d_ax.plot(xx_f, np.polyval(k_fit, xx_f), color=INK, lw=0.8, ls='--')
    d_ax.text(0.97, 0.94, 'pooled r = −0.93\nwithin-yr r = −0.11', transform=d_ax.transAxes, ha='right', va='top', fontsize=7.0, color=MUTED)
    d_ax.set_xlabel('Counterfactual curtailment (%)')
    d_ax.set_ylabel('Displacement\n(MWh/MWh)')
    d_ax.legend(ncol=2, fontsize=7.0, loc='lower left', columnspacing=0.4, handletextpad=0.1)
    grid(d_ax)

    base_yr = float(cs.crossing_year.iloc[0])
    sub_cs = cs.iloc[1:].copy()
    sub_cs['xp'] = sub_cs.crossing_year.replace({np.inf: 2036.8, -np.inf: 2027.2})
    sub_cs = sub_cs.sort_values('xp', ascending=False).reset_index(drop=True)
    short_cs = {
        'Leakage 1.0%': '1.0% methane', 'Leakage 4.2%': '4.2% methane', 'GWP20': 'GWP20',
        'Heat use h = 0.25': 'Heat h = 25%', 'Heat use h = 0.50': 'Heat h = 50%',
        'Registered specification (no electrolysers)': 'No flex. load',
    }
    e.axvline(base_yr, color=INK, lw=0.8)
    for i, r_cs in sub_cs.iterrows():
        lo, hi = sorted([base_yr, r_cs.xp])
        col_cs = C['teal'] if r_cs.xp > base_yr + 0.05 else (C['power'] if r_cs.xp < base_yr - 0.05 else C['grey'])
        e.barh(i, max(hi - lo, 0.12), left=lo if hi - lo > 0.12 else base_yr - 0.06, height=0.56, color=col_cs)
    e.set_yticks(range(len(sub_cs)), [short_cs[v] for v in sub_cs.variant])
    e.set_xlim(2027.5, 2037.2)
    e.set_xticks([2028, 2030, 2033, 2035, 2037], ['2028', '2030', '2033', '2035', '>2035'])
    e.set_xlabel('Year site-power net abatement turns negative')
    grid(e, 'x')

    finish(f, [a, b, c, d_ax, e], 'figED2_weather_methane')


def ed3_temporal_prices():
    d = read('month_hour')
    p = read('price_bands')
    f = fig(188)
    axes = []
    vmax = d.curtail_GW.max()
    for i, y in enumerate(YEARS):
        a = ax(f, 16 + (i % 2) * 86, 9 + (i // 2) * 53, 74, 37)
        axes.append(a)
        m = d[d.year == y].pivot(index='month', columns='hour', values='curtail_GW')
        im = a.imshow(m, origin='lower', aspect='auto', cmap=SEQ, vmin=0, vmax=vmax)
        a.set_xticks([0, 6, 12, 18, 23])
        a.set_yticks([0, 3, 6, 9, 11], ['Jan', 'Apr', 'Jul', 'Oct', 'Dec'])
        a.set_xlabel('Hour of day')
        a.set_ylabel(str(y))
    cb = f.colorbar(im, cax=ax(f, 16, 110, 160, 2.5), orientation='horizontal')
    cb.set_label('Mean curtailed wind and solar (GW)', fontsize=7.0)
    cb.ax.tick_params(labelsize=7.0)

    e = ax(f, 16, 134, 96, 38)
    axes.append(e)
    bot = np.zeros(4)
    vals = [p.below_1, p.below_30 - p.below_1, p.below_gas - p.below_30, p.at_or_above_gas]
    for v_b, lab_b, col_b in zip(vals, ['<1', '1–30', '30–gas cost', '≥ gas cost'], [C['light'], C['green'], C['blue'], C['red']]):
        e.barh(range(4), v_b * 100.0, left=bot, height=0.62, color=col_b, label=lab_b)
        bot += v_b.to_numpy() * 100.0
    e.set_yticks(range(4), p.year.astype(str))
    e.set_xlim(0, 100)
    e.set_xlabel('Site electricity by price band (%)')
    e.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.03), fontsize=7.0, handlelength=0.8, columnspacing=0.8)

    f_ax = ax(f, 130, 134, 48, 38)
    axes.append(f_ax)
    f_ax.plot(p.year, p.gas_cost, color=C['power'], marker='s', ms=3.2, label='Gas dispatch cost')
    f_ax.plot(p.year, p['median'], color=C['navy'], marker='o', ms=3.2, label='Site median price')
    yearaxis(f_ax, short=True)
    f_ax.set_ylim(30, 175)
    f_ax.set_ylabel('EUR per MWh')
    f_ax.legend(loc='upper left', fontsize=7.0, handlelength=1.1)
    grid(f_ax)

    finish(f, axes, 'figED4_temporal_prices')


def ed4_factorial():
    m = read('mechanism_by_weather')
    s = read('factor_shapley_by_weather')
    it = read('factor_interactions_by_weather')

    f = fig(196)
    a = ax(f, 18, 9, 160, 44)
    b = ax(f, 24, 73, 154, 44)
    c = ax(f, 22, 138, 66, 44)
    d_ax = ax(f, 114, 138, 64, 44)

    x = m.pivot_table(index='weather', columns=['background_year', 'vre_year', 'price_year'], values='D')
    a.imshow(x, cmap=BLUE, vmin=0, vmax=0.65, aspect='auto')
    textheat(a, x, '.3f', 0.34, fs=7.0)
    a.set_xticks(range(8), ['/'.join(str(int(v))[-2:] for v in col) for col in x.columns])
    a.set_xlabel('Background / wind-solar / fossil-cost year (B/V/C)')
    a.set_yticks(range(8), [f'WS{w:02}' for w in x.index])
    a.tick_params(length=0)

    # b: Electricity destination shares across all 8 factorial cells
    cats = [
        ('Gas', 'balance_reduced_gas_MWh', C['power']),
        ('Coal/oil', ['balance_reduced_Lignite_MWh', 'balance_reduced_Hard coal_MWh', 'balance_reduced_Light oil_MWh', 'balance_reduced_Heavy oil_MWh'], C['brown']),
        ('Imports', 'balance_reduced_external_import_MWh', C['grey']),
        ('Nuclear', 'balance_reduced_Nuclear_MWh', C['purple']),
        ('Curtailment', 'balance_increased_curtailment_MWh', C['green']),
        ('Flex. load', ['balance_increased_electrolyser_consumption_MWh', 'balance_increased_power_to_heat_consumption_MWh'], C['blue']),
        ('H$_2$ power', 'balance_reduced_hydrogen_MWh', C['sand']),
    ]
    g8 = m.groupby(['background_year', 'vre_year', 'price_year']).mean(numeric_only=True)
    xlabs = ['/'.join(str(int(v))[-2:] for v in idx) for idx in g8.index]
    pos = np.zeros(len(g8))
    for lab_k, col_spec, col_c in cats:
        vals_b = (g8[col_spec].sum(axis=1) if isinstance(col_spec, list) else g8[col_spec]).to_numpy() / 0.40 * 100.0
        b.bar(range(len(g8)), vals_b, bottom=pos, width=0.65, color=col_c, label=lab_k, edgecolor='white', lw=0.3)
        pos += np.maximum(vals_b, 0)
    b.bar(range(len(g8)), 100.0 - pos, bottom=pos, width=0.65, color=C['light'], label='Other')
    b.set_xticks(range(len(g8)), xlabs)
    b.set_ylim(0, 102)
    b.set_ylabel('Response share (%)')
    b.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=8, fontsize=7.0, handlelength=0.8, columnspacing=0.6)

    t_sh = s[(s.kind == 'mechanism') & (s.metric == 'D')]
    z_sh = t_sh.pivot(index='weather', columns='factor', values='effect')[['vre_year', 'price_year', 'background_year']]
    c.imshow(z_sh, cmap=DIV, vmin=-0.30, vmax=0.30, aspect='auto')
    textheat(c, z_sh, '+.3f', 0.16, fs=7.0)
    c.set_xticks(range(3), ['Wind/solar', 'Fossil costs', 'Other'])
    c.set_yticks(range(8), [f'WS{w:02}' for w in z_sh.index])
    c.set_xlabel('Shapley effect (MWh/MWh)')
    c.tick_params(length=0)

    it_a = it[(it.kind == 'mechanism') & (it.metric == 'avoided')]
    facs = ['background_year+vre_year', 'background_year+price_year', 'vre_year+price_year', 'background_year+vre_year+price_year']
    for i, fk in enumerate(facs):
        v = it_a[it_a.factors == fk].effect.to_numpy() * 1000.0
        d_ax.scatter(i + np.linspace(-0.14, 0.14, len(v)), v, s=15, color=C['blue'], alpha=0.75)
        d_ax.plot([i - 0.24, i + 0.24], [v.mean()] * 2, color=INK, lw=1.2)
    d_ax.axhline(0, color=GREY, lw=0.7)
    d_ax.set_xticks(range(4), ['B×V', 'B×C', 'V×C', 'B×V×C'])
    d_ax.set_ylabel('Avoided-emission\ninteraction (kg CO$_2$e/MWh)')
    grid(d_ax)

    finish(f, [a, b, c, d_ax], 'figED5_factorial')


def ed5_country_spillover():
    d = read('country')
    cc = read('country_cases')
    cr = read('country_crossing')

    f = fig(196)
    a = ax(f, 15, 9, 162, 46)
    b = ax(f, 18, 73, 68, 52)
    c = ax(f, 108, 73, 70, 52)
    d_ax = ax(f, 18, 141, 68, 44)
    e = ax(f, 108, 141, 70, 44)

    order = d[d.year == 2035].sort_values('D', ascending=False).country.tolist()
    g = d[d.year == 2035].set_index('country').loc[order]
    cats = [('gas', 'Gas', C['power']), ('coal_lignite', 'Coal', C['brown']), ('nuclear', 'Nuclear', C['purple']),
            ('imports', 'Imports', C['grey']), ('curtailment', 'Curtailment', C['green']),
            ('electrolyser_p2h', 'Flex. load', C['blue']), ('other', 'Other', C['light'])]
    pos = np.zeros(15)
    neg = np.zeros(15)
    for key, lab_k, col_k in cats:
        vals = g[key].to_numpy() * 100.0
        if key == 'other':
            vals = vals + g['oil'].to_numpy() * 100.0
        bot = np.where(vals >= 0, pos, neg)
        a.bar(range(15), vals, bottom=bot, color=col_k, label=lab_k, width=0.70, edgecolor='white', lw=0.25)
        pos += np.maximum(vals, 0)
        neg += np.minimum(vals, 0)
    a.set_xticks(range(15), ['UK' if k == 'GB' else k for k in order])
    a.set_ylabel('Response / added elec. (%)')
    a.axhline(0, color=INK, lw=0.5)
    a.set_ylim(min(neg.min() - 3, -3), max(pos.max() + 3, 103))
    a.legend(ncol=7, loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=7.0, handlelength=0.8, columnspacing=0.6)

    # b: 15x15 cross-border fossil displacement share matrix in 2035
    cols_sp = [f'fossil_gen_reduction_in_{k}_MWh' for k in order]
    m_sp = cc[cc.year == 2035].groupby('country')[cols_sp].mean().loc[order]
    row_tot = m_sp.sum(axis=1)
    sp_pct = m_sp.div(row_tot.where(row_tot > 1.0, np.nan), axis=0) * 100.0
    cmap_sp = BLUE.copy()
    cmap_sp.set_bad('#EEF2F6')
    b.imshow(np.ma.masked_invalid(sp_pct.to_numpy()), cmap=cmap_sp, vmin=0, vmax=100, aspect='auto')
    lbls_o = ['UK' if k == 'GB' else k for k in order]
    b.set_xticks(range(15), lbls_o, rotation=90)
    b.set_yticks(range(15), lbls_o)
    b.set_xlabel('Country where fossil gen. drops (2035)')
    b.set_ylabel('Injecting country')
    b.tick_params(length=0)

    # c: 15-country 4-year D heatmap
    mat_d = d.pivot(index='country', columns='year', values='D').loc[order]
    c.imshow(mat_d, cmap=BLUE, vmin=0, vmax=0.8, aspect='auto')
    textheat(c, mat_d, '.2f', 0.42, fs=7.0)
    c.set_xticks(range(4), mat_d.columns.astype(str))
    c.set_yticks(range(15), lbls_o)
    c.set_xlabel('Target year (D, MWh/MWh)')
    c.tick_params(length=0)

    # d: 15-country 4-year Net abatement (kg CO2e/MWh)
    mat_net = d.pivot(index='country', columns='year', values='net25').loc[order] * 1000.0
    d_ax.imshow(mat_net, cmap=DIV, norm=colors.TwoSlopeNorm(vmin=-55, vcenter=0, vmax=135), aspect='auto')
    for i in range(15):
        for j in range(4):
            vv = mat_net.iloc[i, j]
            txt = f'{vv:+.1f}' if abs(vv) < 1.0 else f'{vv:+.0f}'
            d_ax.text(j, i, txt, ha='center', va='center', fontsize=7.0, color='white' if abs(vv) > 70 else INK)
    d_ax.set_xticks(range(4), mat_net.columns.astype(str))
    d_ax.set_yticks(range(15), lbls_o)
    d_ax.set_xlabel('Net abatement (kg CO$_2$e/MWh)')
    d_ax.tick_params(length=0)

    # e: 15-country 4-year h* heatmap
    mat_h = d.pivot(index='country', columns='year', values='h_star').loc[order]
    e.imshow(mat_h, cmap=WARM, vmin=0.3, vmax=1.9, aspect='auto')
    textheat(e, mat_h, '.2f', 1.15, fs=7.0)
    e.set_xticks(range(4), mat_h.columns.astype(str))
    e.set_yticks(range(15), lbls_o)
    e.set_xlabel('Heat threshold $h_*$ to match grid')
    e.tick_params(length=0)

    finish(f, [a, b, c, d_ax, e], 'figED7_country_spillover')


def ed6_robustness():
    sp = read('spec_checks')
    sup = read('superposition')
    old = read('legacy_sensitivities')
    s_w = read('scale_by_weather')
    imp_w = read52('import_windows')
    fac_e = read52('factorial_effects')
    dose = read52('dose_checks')
    nat_w = read52('national_weights')

    sp_fl = sp[sp.variant == 'No electrolysers or power-to-heat']
    f = fig(192)
    a = ax(f, 15, 9, 44, 42)
    b = ax(f, 77, 9, 42, 42)
    c = ax(f, 135, 9, 43, 42)

    d_ax = ax(f, 15, 68, 44, 42)
    e = ax(f, 75, 68, 44, 42)
    f_ax = ax(f, 135, 68, 43, 42)

    g = ax(f, 15, 130, 44, 42)
    h = ax(f, 82, 134, 95, 38)

    # a: With vs without flexible load (144 pairs)
    for y, grp_y in sp_fl.groupby('year'):
        a.scatter(grp_y.D_main, grp_y.D_variant, s=11, color=YC_WARM[y], label=str(y), alpha=0.75, lw=0)
    a.plot([0, 0.65], [0, 0.65], color=GREY, ls='--', lw=0.75)
    a.set_xlim(0, 0.65)
    a.set_ylim(0, 0.65)
    a.set_xlabel('With flex. load (MWh/MWh)')
    a.set_ylabel('Without flex. load\n(MWh/MWh)')
    a.legend(ncol=2, fontsize=7.0, loc='upper left', handletextpad=0.2, columnspacing=0.5)

    # b: Fixed nuclear & No gas derating (mean + min/max range)
    z = old[old.variant.isin(['S-nuc nuclear fixed', 'S-gas no gas cut'])]
    for i, r in enumerate(z.itertuples()):
        b.barh(i, r.dD, color=C['blue'], alpha=0.68, height=0.50)
        b.errorbar(r.dD, i, xerr=[[max(0, r.dD - r.dD_min)], [max(0, r.dD_max - r.dD)]],
                   fmt='none', ecolor=INK, capsize=1.8, lw=0.75)
    b.set_yticks(range(len(z)), [('Fixed nuc.' if 'nuc' in r.variant else 'No gas cut') + f"\n('{str(r.year)[2:]})" for r in z.itertuples()])
    b.axvline(0, color=GREY, lw=0.65)
    b.set_xlabel('ΔD (MWh/MWh)')
    grid(b, 'x')

    # c: 4 External-import dispatch boundary regimes (A1)
    imp_specs = [
        ('all_hours', 'fixed', '50% cap, 100 EUR', C['power'], '-', 'o'),
        ('event_only', 'fixed', 'Event cap, 100 EUR', C['sand'], '--', 's'),
        ('all_hours', 'linked', '50% cap, gas-linked', C['teal'], '-', '^'),
        ('event_only', 'linked', 'Event cap, gas-linked', C['navy'], ':', 'D'),
    ]
    for av, pr_m, lab_i, col_i, ls_i, mk_i in imp_specs:
        sub_i = imp_w[(imp_w.availability == av) & (imp_w.price == pr_m)].sort_values('year')
        c.plot(sub_i.year, sub_i.D, color=col_i, ls=ls_i, marker=mk_i, ms=2.8, label=lab_i)
    yearaxis(c, short=True)
    c.set_ylim(0.0, 0.62)
    c.set_ylabel('Window mean D\n(MWh/MWh)')
    c.legend(loc='upper right', fontsize=7.0, handlelength=1.1, labelspacing=0.18)
    grid(c)

    # d: Shapley attribution under flat_all vs linked_event import pricing (A1)
    fa_m = fac_e.groupby('trade')[['vre', 'cost', 'background']].mean()
    x_fa = np.arange(3)
    d_ax.axhline(0, color=INK, lw=0.55)
    d_ax.bar(x_fa - 0.18, fa_m.loc['flat_all', ['vre', 'cost', 'background']], width=0.34,
             color=C['power'], alpha=0.85, label='Fixed 100 EUR')
    d_ax.bar(x_fa + 0.18, fa_m.loc['linked_event', ['vre', 'cost', 'background']], width=0.34,
             color=C['teal'], alpha=0.85, label='Gas-linked')
    d_ax.set_xticks(x_fa, ['VRE (V)', 'Cost (C)', 'Other (B)'])
    d_ax.set_ylim(-0.36, 0.12)
    d_ax.set_ylabel('Shapley effect\n(MWh/MWh)')
    d_ax.legend(loc='lower right', fontsize=7.0, handlelength=0.9)
    grid(d_ax)

    # e: Actual dose vs 20 MW minimum probing dose (C3)
    e.plot([0, 0.85], [0, 0.85], color=GREY, ls='--', lw=0.75, zorder=1)
    for cmp_k, lab_d, col_d, mk_d in [
        ('SI_actual', 'SI actual vs 20 MW', C['teal'], 'o'),
        ('HR_actual', 'HR actual vs 20 MW', C['purple'], 's'),
        ('region_raw', 'EU15 equal-raw', C['blue'], '^'),
    ]:
        sd_d = dose[dose.comparison == cmp_k]
        e.scatter(sd_d.reference_D, sd_d.actual_D, s=13, color=col_d, marker=mk_d, alpha=0.80, label=lab_d, zorder=3)
    e.text(0.96, 0.06, '2035 |ΔD| < 10$^{-11}$', transform=e.transAxes, ha='right', va='bottom', fontsize=7.0, color=MUTED)
    e.set_xlim(-0.03, 0.85)
    e.set_ylim(-0.03, 0.85)
    e.set_xlabel('Archive / 20 MW floor D')
    e.set_ylabel('Actual-dose D\n(MWh/MWh)')
    e.legend(loc='upper left', fontsize=7.0, handletextpad=0.2)
    grid(e)

    # f: 2035 Alternative national food-waste weightings vs Fleet (C1)
    w_ord = [('Low', 'Low\nFW', C['teal']), ('Central', 'Cent.\nFW', C['blue']),
             ('High', 'High\nFW', C['sand']), ('fleet', 'Biogas\nfleet', C['purple'])]
    for idx_w, (wk, wlab, wcol) in enumerate(w_ord):
        vals_w = nat_w[nat_w.composition == wk].linear_D.to_numpy()
        f_ax.boxplot([vals_w], positions=[idx_w], widths=0.44, patch_artist=True, showfliers=False,
                     boxprops={'facecolor': wcol, 'alpha': 0.30}, medianprops={'color': INK, 'lw': 1.0})
        f_ax.scatter(idx_w + np.linspace(-0.10, 0.10, len(vals_w)), vals_w, s=10, color=wcol, zorder=4)
        f_ax.text(idx_w, vals_w.max() + 0.008, f'{vals_w.mean():.2f}', ha='center', va='bottom', fontsize=7.0, weight='bold')
    f_ax.set_xticks(range(4), [x[1] for x in w_ord])
    f_ax.set_ylim(0.08, 0.30)
    f_ax.set_ylabel('2035 linear D\n(MWh/MWh)')
    grid(f_ax)

    # g: Superposition ratio across 4 target years
    for y, grp_s in sup.groupby('year'):
        g.scatter(np.full(len(grp_s), y) + np.linspace(-0.14, 0.14, len(grp_s)), grp_s.ratio, s=13, color=YC_WARM[y])
        g.plot([y - 0.22, y + 0.22], [grp_s.ratio.mean()] * 2, color=INK, lw=1.1)
    g.axhline(1.0, color=GREY, ls='--', lw=0.75)
    yearaxis(g, short=True)
    g.set_ylabel('Sum of single-country /\nsimultaneous D')
    grid(g)

    # h: Electricity destination across 2x2 scale x mix cells
    g4 = s_w.groupby(['mix', 'total']).mean(numeric_only=True)
    order4 = [('food', 'small', 'Food 7.3'), ('food', 'large', 'Food 151.4'),
              ('fleet', 'small', 'Fleet 7.3'), ('fleet', 'large', 'Fleet 151.4')]
    cats = [('Gas', 'balance_reduced_gas_MWh', C['power']),
            ('Nuclear', 'balance_reduced_Nuclear_MWh', C['purple']),
            ('Imports', 'balance_reduced_external_import_MWh', C['grey']),
            ('Curt.', 'balance_increased_curtailment_MWh', C['green']),
            ('Electrol.', 'balance_increased_electrolyser_consumption_MWh', C['blue']),
            ('P2H', 'balance_increased_power_to_heat_consumption_MWh', '#8EC0D6'),
            ('H$_2$ pwr', 'balance_reduced_hydrogen_MWh', C['sand'])]
    left4 = np.zeros(4)
    for lab_c, col_k, col_c in cats:
        vv = np.array([g4.loc[(mx, tot_k), col_k] / 0.40 * 100.0 for mx, tot_k, _ in order4])
        h.barh(range(4), vv, left=left4, height=0.58, color=col_c, label=lab_c, edgecolor='white', lw=0.3)
        left4 += np.maximum(vv, 0)
    h.barh(range(4), 100.0 - left4, left=left4, height=0.58, color=C['light'], label='Other')
    h.set_yticks(range(4), [x[2] for x in order4])
    h.invert_yaxis()
    h.set_xlim(0, 100)
    h.set_xlabel('2035 added electricity response allocation (%)')
    h.legend(loc='lower left', bbox_to_anchor=(-0.06, 1.02), ncol=4, fontsize=7.0, handlelength=0.75, columnspacing=0.55)

    finish(f, [a, b, c, d_ax, e, f_ax, g, h], 'figED6_robustness')


def ed7_product_boundaries():
    eta = read('CHP_thermal_efficiency_sensitivity')
    eq_raw = read52('annual_equal_raw')
    th_sum = read52('threshold_summary')
    cm_cr = read52('common_credit')
    acc_tm = read52('access_and_timing')
    bb_w = read('boundary_base_by_weather')

    f = fig(192)
    a = ax(f, 15, 9, 75, 44)
    b = ax(f, 104, 9, 74, 44)
    c = ax(f, 18, 70, 78, 45)
    d_ax = ax(f, 110, 70, 68, 45)
    e = ax(f, 18, 132, 62, 45)
    f_ax = ax(f, 94, 132, 84, 45)

    # a: Dual-basis 36-weather box-and-jitter comparison (Equal raw biogas vs Deliverable biomethane h*)
    a.axhspan(0, 100, color=C['teal'], alpha=0.08, lw=0, zorder=0)
    a.axhline(100, color=GREY, ls='--', lw=0.8, zorder=1)
    for idx_y, yr in enumerate(YEARS):
        v_raw = eq_raw[eq_raw.year == yr].h_match.to_numpy() * 100.0
        v_del = bb_w[bb_w.year == yr].h_match_injection_primary.to_numpy() * 100.0
        for dx_b, arr_b, col_b in [(-0.18, v_raw, C['teal']), (0.18, v_del, C['navy'])]:
            px = idx_y + dx_b
            a.boxplot([arr_b], positions=[px], widths=0.28, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': col_b, 'alpha': 0.30, 'edgecolor': col_b, 'lw': 0.7},
                      medianprops={'color': INK, 'lw': 1.0}, whiskerprops={'color': col_b, 'lw': 0.65},
                      capprops={'color': col_b, 'lw': 0.65}, zorder=3)
            a.scatter(px + np.linspace(-0.07, 0.07, len(arr_b)), arr_b, s=6, color=col_b, alpha=0.65, lw=0, zorder=4)
        if yr == 2035:
            a.text(idx_y - 0.18, v_raw.min() - 4.0, f'{v_raw.mean():.0f}%', ha='center', va='top', fontsize=7.0, color=C['teal'], weight='bold')
            a.text(idx_y + 0.18, v_del.max() + 3.0, f'{v_del.mean():.0f}%', ha='center', va='bottom', fontsize=7.0, color=C['navy'], weight='bold')
    a.set_xticks(range(4), [str(y) for y in YEARS])
    a.set_xlim(-0.52, 3.55)
    a.set_ylim(45, 174)
    a.set_xlabel('Target year')
    a.set_ylabel('Heat use to match grid (%)')
    a.legend(handles=[
        Patch(facecolor=C['teal'], alpha=0.55, edgecolor=C['teal'], label='Equal raw biogas'),
        Patch(facecolor=C['navy'], alpha=0.55, edgecolor=C['navy'], label='Deliverable basis'),
    ], loc='lower right', fontsize=7.0, handlelength=0.9, labelspacing=0.18)
    grid(a)

    # b: Floating engine thermal-efficiency design-envelope bars (eta_th = 0.40 to 0.55, central 0.488)
    b.axhspan(0, 100, color=C['teal'], alpha=0.07, lw=0, zorder=0)
    b.axhline(100, color=GREY, ls='--', lw=0.8, zorder=1)
    for idx_y, yr in enumerate(YEARS):
        gy = eta[eta.year == yr].set_index('thermal_efficiency').heat_fraction_to_match_injection * 100.0
        lo_h = float(gy.loc[0.55])
        mid_h = float(gy.loc[0.488])
        hi_h = float(gy.loc[0.40])
        col_y = YC_WARM[yr]
        b.bar(idx_y, hi_h - lo_h, bottom=lo_h, width=0.44, color=col_y, alpha=0.35, edgecolor=col_y, lw=0.85, zorder=3)
        b.plot([idx_y - 0.22, idx_y + 0.22], [mid_h, mid_h], color=INK, lw=1.2, zorder=4)
        b.scatter([idx_y], [mid_h], marker='D', s=16, color='white', edgecolor=INK, lw=0.8, zorder=5)
        b.text(idx_y, hi_h + 2.5, f'{hi_h:.0f}%', ha='center', va='bottom', fontsize=7.0, color=col_y, weight='bold')
        b.text(idx_y, lo_h - 2.5, f'{lo_h:.0f}%', ha='center', va='top', fontsize=7.0, color=MUTED)
    b.text(0.03, 0.95, 'Span: $\\eta_{th}$ = 0.40 (top) to 0.55 (bottom)\nDiamond: central $\\eta_{th}$ = 0.488',
           transform=b.transAxes, fontsize=7.0, va='top', ha='left', color=INK, linespacing=1.12)
    b.set_xticks(range(4), [str(y) for y in YEARS])
    b.set_xlim(-0.55, 3.55)
    b.set_ylim(48, 202)
    b.set_xlabel('Target year')
    b.set_ylabel('Heat use to match grid (%)')
    grid(b)

    # c: All 270 joint equal-raw summary cells in 2035 (horizontal box + beeswarm across differential leakage)
    t35_all = th_sum[th_sum.year == 2035].copy()
    t35_all['leak_grp'] = np.where(t35_all.site_leak > t35_all.grid_leak + 1e-6, 0,
                          np.where(np.isclose(t35_all.site_leak, t35_all.grid_leak), 1, 2))
    grp_labels = ['Site > grid\nleak. (n=90)', 'Equal leak.\n(n=90)', 'Grid > site\nleak. (n=90)']
    c.axvline(0, color=INK, lw=0.75, zorder=2)
    c.axvspan(-0.075, 0.0, color=C['power'], alpha=0.07, lw=0, zorder=0)
    rng_c = np.random.default_rng(42)
    for g_idx in range(3):
        sub_g = t35_all[t35_all.leak_grp == g_idx]
        vals_g = sub_g.grid_minus_site_q1.to_numpy()
        c.boxplot([vals_g], positions=[g_idx], vert=False, widths=0.50, patch_artist=True, showfliers=False,
                  boxprops={'facecolor': '#E2E8F0', 'alpha': 0.55, 'edgecolor': MUTED, 'lw': 0.65},
                  medianprops={'color': INK, 'lw': 1.1}, whiskerprops={'color': MUTED, 'lw': 0.6},
                  capprops={'color': MUTED, 'lw': 0.6}, zorder=2)
        jit_y = g_idx + rng_c.uniform(-0.19, 0.19, size=len(sub_g))
        hi_heat = sub_g.h >= 0.74
        c.scatter(vals_g[~hi_heat], jit_y[~hi_heat], s=9, color=C['teal'], alpha=0.78, lw=0, zorder=3,
                  label='h ≤ 50%' if g_idx == 0 else None)
        c.scatter(vals_g[hi_heat], jit_y[hi_heat], s=10, color=C['power'], alpha=0.82, edgecolor='white', lw=0.2, zorder=4,
                  label='h = 75–100%' if g_idx == 0 else None)
    c.text(0.98, 0.06, '252 / 270 (93.3%) > 0\n18 / 270 (6.7%) < 0', transform=c.transAxes,
           fontsize=7.0, ha='right', va='bottom', weight='bold', color=INK, linespacing=1.1)
    c.set_yticks(range(3), grp_labels)
    c.invert_yaxis()
    c.set_xlim(-0.072, 0.285)
    c.set_xlabel('2035 grid minus cogeneration at q=1 (t CO$_2$e/MWh raw)')
    c.legend(loc='lower right', bbox_to_anchor=(1.0, 1.01), ncol=2, fontsize=7.0, handlelength=0.8, columnspacing=0.7)
    grid(c, 'x')

    # d: Grouped bar + 36-weather min-max whiskers for minimum fossil-gas replacement q* across heat tiers
    t_q = th_sum[(th_sum.site_leak == 0.025) & (th_sum.grid_leak == 0.025) &
                 (th_sum.credits == 'domestic') & (th_sum.compression == 'domestic')]
    hs_d = [0.0, 0.25, 0.50, 1.0]
    xd = np.arange(len(hs_d))
    d_ax.axhline(100, color=GREY, ls='--', lw=0.75, zorder=1)
    for k_y, (yr_q, col_q) in enumerate([(2028, C['navy']), (2030, C['blue']), (2033, '#58A6A6'), (2035, C['teal'])]):
        sub_y = t_q[t_q.year == yr_q].set_index('h')
        means_q = [sub_y.loc[hv, 'qstar'] * 100.0 for hv in hs_d]
        lo_q = [(sub_y.loc[hv, 'qstar'] - sub_y.loc[hv, 'qstar_min']) * 100.0 for hv in hs_d]
        hi_q = [(sub_y.loc[hv, 'qstar_max'] - sub_y.loc[hv, 'qstar']) * 100.0 for hv in hs_d]
        px_d = xd + (k_y - 1.5) * 0.19
        d_ax.bar(px_d, means_q, width=0.17, color=col_q, alpha=0.88, label=str(yr_q), zorder=3)
        d_ax.errorbar(px_d, means_q, yerr=[lo_q, hi_q], fmt='none', ecolor=INK, elinewidth=0.55, capsize=1.3, zorder=4)
        if yr_q == 2035:
            d_ax.annotate('20.9%', xy=(px_d[0], means_q[0] + hi_q[0] + 1.0), xytext=(0.24, 52.0),
                          ha='center', va='bottom', fontsize=7.0, color=C['teal'], weight='bold',
                          arrowprops=dict(arrowstyle='->', lw=0.55, color=C['teal']))
    d_ax.set_xticks(xd, ['0%', '25%', '50%', '100%'])
    d_ax.set_xlabel('Effective heat use (%)')
    d_ax.set_ylim(0, 146)
    d_ax.set_ylabel('Gas replacement $q^*$ needed (%)')
    d_ax.legend(ncol=2, loc='upper left', fontsize=7.0, handlelength=0.85, columnspacing=0.6, labelspacing=0.16)
    grid(d_ax)

    # e: Horizontal stacked 36-weather outcome bars (<0 vs >=0 in 2035, with 2033 diamond markers)
    credits_e = [(-0.010, '−0.010'), (0.000, '0 (base)'), (0.004, '+0.004'), (0.010, '+0.010')]
    ye_pos = np.arange(len(credits_e))
    for i_c, (wc_v, _) in enumerate(credits_e):
        n35 = int(cm_cr[(cm_cr.year == 2035) & np.isclose(cm_cr.common_t_per_raw, wc_v)].n_negative.iloc[0])
        n33 = int(cm_cr[(cm_cr.year == 2033) & np.isclose(cm_cr.common_t_per_raw, wc_v)].n_negative.iloc[0])
        e.barh(i_c, n35, height=0.54, color=C['power'], alpha=0.85, label='2035 (< 0)' if i_c == 0 else None, zorder=3)
        e.barh(i_c, 36 - n35, left=n35, height=0.54, color='#DCE6EE', edgecolor='none', label='2035 (≥ 0)' if i_c == 0 else None, zorder=2)
        e.scatter([n33], [i_c], marker='D', s=18, color=C['sand'], edgecolor=INK, lw=0.55, zorder=5, label='2033 (< 0)' if i_c == 0 else None)
        tx_pos = n35 - 1.5 if n35 >= 12 else n35 + 1.2
        ha_tx = 'right' if n35 >= 12 else 'left'
        col_tx = 'white' if n35 >= 12 else C['power']
        e.text(tx_pos, i_c, f'{n35}/36', va='center', ha=ha_tx, fontsize=7.0, color=col_tx, weight='bold', zorder=4)
    e.set_yticks(ye_pos, [x[1] for x in credits_e])
    e.invert_yaxis()
    e.set_xlim(-1.2, 36.8)
    e.set_xticks([0, 12, 24, 36])
    e.set_ylabel('Shared waste credit (t/MWh)')
    e.set_xlabel('Weather years < 0 (of 36; 2028–30: all 0)')
    e.legend(loc='lower right', bbox_to_anchor=(1.02, 1.01), ncol=3, fontsize=7.0, handlelength=0.8, columnspacing=0.5)

    # f: Grouped bars + 2030/2032 start-year markers across ALL 36 constrained rollout scenarios
    shares_f = [0.25, 0.50, 0.75, 1.00]
    qs_f = [(0.50, 'q = 0.50', '#94A8BA'), (0.75, 'q = 0.75', C['blue']), (1.00, 'q = 1.00', C['teal'])]
    xf_s = np.arange(len(shares_f))
    for k_q, (qv, qlab, qcol) in enumerate(qs_f):
        v28_s = [float(acc_tm[(acc_tm.start_year == 2028) & np.isclose(acc_tm.eligible_share, es) & np.isclose(acc_tm.fossil_share, qv)].increment_over_keep_Mt.iloc[0]) for es in shares_f]
        v30_s = [float(acc_tm[(acc_tm.start_year == 2030) & np.isclose(acc_tm.eligible_share, es) & np.isclose(acc_tm.fossil_share, qv)].increment_over_keep_Mt.iloc[0]) for es in shares_f]
        v32_s = [float(acc_tm[(acc_tm.start_year == 2032) & np.isclose(acc_tm.eligible_share, es) & np.isclose(acc_tm.fossil_share, qv)].increment_over_keep_Mt.iloc[0]) for es in shares_f]
        px_s = xf_s + (k_q - 1) * 0.25
        f_ax.bar(px_s, v28_s, width=0.22, color=qcol, alpha=0.85, label=f'{qlab} (2028)', zorder=3)
        f_ax.scatter(px_s, v30_s, marker='o', s=12, facecolor='white', edgecolor=INK, lw=0.65, zorder=4, label='Start 2030' if k_q == 0 else None)
        f_ax.scatter(px_s, v32_s, marker='^', s=13, facecolor=C['sand'], edgecolor=INK, lw=0.45, zorder=5, label='Start 2032' if k_q == 0 else None)
        if k_q == 2:
            f_ax.text(px_s[-1], v28_s[-1] + 0.22, f'+{v28_s[-1]:.2f}', ha='center', va='bottom', fontsize=7.0, color=C['teal'], weight='bold')
        if k_q == 0:
            f_ax.text(px_s[0], max(v28_s[0], v30_s[0]) + 0.22, '+0.41', ha='center', va='bottom', fontsize=7.0, color=MUTED)
    f_ax.axhline(0, color=INK, lw=0.55)
    f_ax.set_xticks(xf_s, ['25%', '50%', '75%', '100%'])
    f_ax.set_xlim(-0.52, 3.52)
    f_ax.set_ylim(0, 9.6)
    f_ax.set_xlabel('Grid-accessible plant share')
    f_ax.set_ylabel('2028–35 gain vs power (Mt CO$_2$e)')
    f_ax.legend(loc='upper left', ncol=2, fontsize=7.0, handlelength=0.85, columnspacing=0.6, labelspacing=0.16)
    grid(f_ax)

    finish(f, [a, b, c, d_ax, e, f_ax], 'figED9_product_boundaries')


def ed8_costs_pathways():
    co = read('cost_components')
    u = read('uses')
    cum = read('cumulative')
    mat_sc = read('scenario_use_matrix')

    f = fig(185)
    a = ax(f, 16, 10, 96, 52)
    b = ax(f, 134, 10, 44, 52)
    c = ax(f, 44, 82, 54, 52)
    d_ax = ax(f, 116, 82, 62, 52)
    # Wait: let's add a clean bottom wide cumulative fossil panel OR 2x2 crisp layout!
    # Wait: 2x2 is 4 panels (a, b, c, d), let's make sure bottom row fills cleanly:
    c.remove(); d_ax.remove()
    c = ax(f, 48, 84, 54, 46)
    d_ax = ax(f, 118, 84, 60, 46)
    # Plus e? Let's keep 4 rich, perfectly balanced panels or add e (2035 cost breakdown comparison)?
    # Wait: c + d on row 2 at y=84..132, or resize f(150) cleanly! Yes, 148 mm height fits 4 panels effortlessly with zero wasted whitespace!
    f.set_size_inches(180 / 25.4, 150 / 25.4)
    a.remove(); b.remove(); c.remove(); d_ax.remove()
    a = ax(f, 16, 10, 96, 54)
    b = ax(f, 134, 10, 44, 54)
    c = ax(f, 48, 84, 54, 52)
    d_ax = ax(f, 118, 84, 60, 52)

    order = co.use.unique().tolist()
    labels_x = []
    pos = 0
    for y in YEARS:
        for key in order:
            r = co[(co.year == y) & (co.use == key)].iloc[0]
            a.bar(pos, r.production, color=GREY, width=0.7, label='Production' if pos == 0 else None)
            a.bar(pos, r.use_specific, bottom=r.production, color=C['blue'], width=0.7, label='Use-specific' if pos == 0 else None)
            a.bar(pos, r.fuel_saving, color=C['teal'], width=0.7, label='Fuel saving' if pos == 0 else None)
            a.scatter(pos, r.net, marker='_', s=45, color=INK, zorder=5)
            labels_x.append(f"{str(y)[2:]}\n" + ('P' if 'Site' in key else 'G' if 'Grid' in key or 'grid' in key else 'T'))
            pos += 1
    a.set_xticks(range(pos), labels_x)
    a.set_xlabel('Year / P power, G gas grid, T transport')
    a.set_ylabel('Cost (EUR per MWh biomethane)')
    a.axhline(0, color=INK, lw=0.6)
    a.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.03), fontsize=7.0, columnspacing=0.8, handlelength=1.0)
    grid(a)

    co2_p = u.groupby('year').CO2_price.first()
    for key, col, mark, lab_u in [('U1', C['power'], 'o', 'Power'), ('U4', C['inject'], 's', 'Gas grid'), ('U5_CNG_diesel', C['truck'], '^', 'Transport')]:
        z = u[u.use == key]
        v = z.cost_per_t.to_numpy()
        valid = np.isfinite(v) & (v > 0)
        b.plot(z.year[valid], v[valid], marker=mark, ms=3.2, color=col, label=lab_u)
    b.plot(co2_p.index, co2_p.values, color=INK, ls='--', lw=1.0, drawstyle='steps-mid', label='ERAA CO$_2$')
    b.set_yscale('log')
    b.set_ylim(45, 16000)
    b.set_yticks([50, 100, 300, 1000, 3000, 10000], ['50', '100', '300', '1,000', '3,000', '10,000'])
    b.minorticks_off()
    b.set_ylabel('Abatement cost (EUR/t)')
    yearaxis(b, short=True)
    grid(b)
    b.legend(fontsize=7.0, loc='center right', bbox_to_anchor=(1.0, 0.57))

    # c: Low / Central / High x 7 uses cumulative net abatement bubble matrix
    uses_m = ['U1', 'U2', 'U2+', 'U4', 'U5_CNG_fossil_CNG', 'U5_LNG_diesel', 'U5_CNG_diesel']
    ulab = {'U1': 'U1 site power', 'U2': 'U2 flex. reserve', 'U2+': 'U2+ spare to grid', 'U4': 'U4 gas grid',
            'U5_CNG_fossil_CNG': 'U5 vs fossil CNG', 'U5_LNG_diesel': 'U5 LNG truck', 'U5_CNG_diesel': 'U5 CNG truck'}
    scs = ['Low', 'Central', 'High']
    for i, u_k in enumerate(uses_m):
        for j_s, sc_k in enumerate(scs):
            vv = float(mat_sc[(mat_sc.use == u_k) & (mat_sc.scenario == sc_k)].cum_net_Mt.iloc[0])
            col_m = C['power'] if u_k in ('U1', 'U2') else C['teal']
            c.scatter(j_s, i, s=11 * abs(vv) + 12, color=col_m, alpha=0.85, edgecolor=INK, lw=0.35, zorder=3)
            c.text(j_s + 0.24, i, f'{vv:.1f}', fontsize=7.0, va='center', ha='left')
    c.set_xticks(range(3), [f'{s_k}\n{mat_sc[mat_sc.scenario==s_k].cum_resource_TWh.iloc[0]:.0f} TWh' for s_k in scs])
    c.set_yticks(range(len(uses_m)), [ulab[u_k] for u_k in uses_m])
    c.set_xlim(-0.45, 2.85)
    c.set_ylim(len(uses_m) - 0.4, -0.6)
    c.tick_params(length=0)
    c.set_xlabel('Resource scenario (2028–2035)')

    for key, lab_c, col_c in [('keep', 'Continuous power', C['power']), ('switch', 'Switch at net zero', C['switch']), ('inject', 'Gas grid from 2028', C['inject'])]:
        d_ax.plot(cum.year, cum['cum_' + key + '_fossil_TWh'], color=col_c, marker='o', ms=3, label=lab_c)
    yearaxis(d_ax)
    d_ax.set_ylabel('Cumulative fossil fuel\ndisplaced (TWh)')
    grid(d_ax)
    d_ax.legend(loc='upper left', fontsize=7.0)

    finish(f, [a, b, c, d_ax], 'figED10_costs_pathways')


def ed9_flexible_reserve():
    d = read('reserve')
    v = read('reserve_diverted')
    u = read('uses')
    f = fig(188)
    a = ax(f, 15, 9, 86, 46)
    b = ax(f, 116, 9, 62, 46)
    c = ax(f, 15, 72, 82, 45)
    d_ax = ax(f, 107, 72, 71, 45)
    e = ax(f, 15, 134, 163, 42)

    # a: 36-weather Raincloud + Interquartile Bar of Annual Unserved Energy Avoided (GWh)
    d['dEENS_GWh'] = d.dEENS / 1000.0
    for i, y in enumerate(YEARS):
        sub_a = d[d.year == y].sort_values('dEENS_GWh')
        vals_a = sub_a.dEENS_GWh.to_numpy()
        pos_mask = vals_a > 1e-4
        col_y = YC_WARM[y]
        jit_x = i - 0.18 + np.linspace(-0.10, 0.10, len(vals_a))
        a.scatter(jit_x[~pos_mask], vals_a[~pos_mask], s=9, facecolors='white', edgecolors=GREY, lw=0.45, zorder=2)
        a.scatter(jit_x[pos_mask], vals_a[pos_mask], s=10, color=col_y, edgecolor='white', lw=0.25, alpha=0.86, zorder=3)
        kde_a = gaussian_kde(vals_a[pos_mask], bw_method=0.42)
        gy_a = np.linspace(0.0, vals_a.max() + 1.2, 120)
        dens_a = kde_a(gy_a)
        dens_a = dens_a / dens_a.max() * 0.34
        a.fill_betweenx(gy_a, i + 0.04, i + 0.04 + dens_a, color=col_y, alpha=0.46, lw=0, zorder=2)
        a.plot([i + 0.04, i + 0.04], [0.0, gy_a.max()], color=col_y, lw=0.65, zorder=3)
        q25, q75 = np.percentile(vals_a, [25, 75])
        a.plot([i + 0.04, i + 0.04], [q25, q75], color=INK, lw=2.0, zorder=4)
        mu_a = vals_a.mean()
        a.plot([i + 0.04, i + 0.36], [mu_a, mu_a], color=INK, lw=1.1, zorder=5)
        a.text(i + 0.22, mu_a + 1.1, f'{mu_a:.2f}', fontsize=7.0, ha='center', va='bottom', weight='bold', color=INK)
    a.text(0.02, 0.95, 'Positive-shortfall weather years: 28, 28, 28, 23 (of 36)', transform=a.transAxes,
           fontsize=7.0, ha='left', va='top', color=MUTED)
    a.set_xticks(range(4), [str(y) for y in YEARS])
    a.set_xlim(-0.48, 3.55)
    a.set_ylim(-1.2, 31.5)
    a.set_xlabel('Target year')
    a.set_ylabel('Unserved energy avoided\n(GWh yr$^{-1}$)')
    grid(a)

    # b: Log-scaled Box-and-Beeswarm of System Shortfall Share Avoided (%) + Aggregate Weighted Share
    for i, y in enumerate(YEARS):
        t = d[(d.year == y) & (d.EENS_R0 > 1.0)]
        vals_b = 100.0 * t.dEENS.to_numpy() / t.EENS_R0.to_numpy()
        col_y = YC_WARM[y]
        b.boxplot([vals_b], positions=[i], widths=0.42, patch_artist=True, showfliers=False,
                  boxprops={'facecolor': col_y, 'alpha': 0.26, 'edgecolor': col_y, 'lw': 0.65},
                  medianprops={'color': col_y, 'lw': 0.9, 'ls': ':'}, whiskerprops={'color': col_y, 'lw': 0.6},
                  capprops={'color': col_y, 'lw': 0.6}, zorder=2)
        xs = np.full(len(vals_b), float(i))
        srt = np.argsort(np.log10(vals_b))
        bins = np.round(np.log10(vals_b[srt]) / 0.055)
        for bk in np.unique(bins):
            idx = srt[bins == bk]
            nb = len(idx)
            xs[idx] = i + (np.arange(nb) - (nb - 1) / 2.0) * 0.075
        b.scatter(xs, vals_b, s=10, color=col_y, edgecolor='white', lw=0.25, zorder=3)
        tot_r = 100.0 * t.dEENS.sum() / t.EENS_R0.sum()
        b.plot([i - 0.26, i + 0.26], [tot_r, tot_r], color=INK, lw=1.2, zorder=4)
        b.text(i, vals_b.max() * 1.12, f'{tot_r:.1f}%', fontsize=7.0, ha='center', va='bottom', weight='bold')
    b.set_yscale('log')
    b.set_ylim(0.72, 31.0)
    b.set_yticks([1, 2, 5, 10, 20], ['1%', '2%', '5%', '10%', '20%'])
    b.minorticks_off()
    b.set_xticks(range(4), [str(y) for y in YEARS])
    b.set_xlim(-0.55, 3.55)
    b.set_xlabel('Target year')
    b.set_ylabel('System shortfall avoided (%)')
    grid(b)

    # c: Paired Economic Value Crossover Bars (Operational fuel-saving vs Adequacy scarcity value + Ratio badges)
    xc = np.arange(4)
    f_means, f_ci = [], []
    a_means, a_ci = [], []
    for y in YEARS:
        gy_c = d[d.year == y]
        fv = gy_c.fuel_saving.to_numpy() / 1e6
        av = gy_c.dEENS.to_numpy() * 12240.0 / 1e6
        f_means.append(fv.mean()); f_ci.append(1.96 * fv.std(ddof=1) / np.sqrt(len(fv)))
        a_means.append(av.mean()); a_ci.append(1.96 * av.std(ddof=1) / np.sqrt(len(av)))
    c.bar(xc - 0.19, f_means, width=0.34, color=C['sand'], alpha=0.86, label='Fuel-saving value', zorder=3)
    c.bar(xc + 0.19, a_means, width=0.34, color=C['reserve'], alpha=0.86, label='Adequacy scarcity value', zorder=3)
    c.errorbar(xc - 0.19, f_means, yerr=f_ci, fmt='none', ecolor=INK, elinewidth=0.65, capsize=1.6, zorder=4)
    c.errorbar(xc + 0.19, a_means, yerr=a_ci, fmt='none', ecolor=INK, elinewidth=0.65, capsize=1.6, zorder=4)
    for i_c in range(4):
        rat = a_means[i_c] / f_means[i_c]
        top_c = max(f_means[i_c] + f_ci[i_c], a_means[i_c] + a_ci[i_c])
        c.text(i_c, top_c + 2.5, f'{rat:.2f}×', ha='center', va='bottom', fontsize=7.0,
               weight='bold', color=C['reserve'] if rat >= 1.0 else MUTED)
    c.set_xticks(xc, [str(y) for y in YEARS])
    c.set_xlim(-0.55, 3.55)
    c.set_ylim(0, 102)
    c.set_xlabel('Target year')
    c.set_ylabel('Annual value (M EUR yr$^{-1}$)')
    c.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=7.0, handlelength=0.9, columnspacing=0.8)
    grid(c)

    # d: Physical Gas-Budget Allocation Sankey Ribbon + Dispatched Peaking Gas Mini-Inset
    d_ax.set_xlim(0, 10.4)
    d_ax.set_ylim(-0.48, 3.22)
    d_ax.axis('off')
    ribbon(d_ax, 2.32, [0.2, 2.8, 5.2, 7.4], [0.75, 0.75, 0.75, 0.75], C['power'], H=1.02, alpha=0.86, z=2)
    ribbon(d_ax, 1.15, [0.2, 2.6, 4.4, 7.4], [0.25, 0.25, 0.2453, 0.2453], C['teal'], H=1.02, alpha=0.82, z=2)
    ribbon(d_ax, 1.52, [2.6, 4.6, 7.4], [0.025, 0.035, 0.035], C['sand'], H=1.02, alpha=0.95, z=3)
    d_ax.text(0.2, 2.96, '100% biomethane budget', fontsize=7.0, weight='bold', va='bottom', ha='left')
    d_ax.text(3.8, 2.32, '75.0% continuous baseload\n(0.30 MWh$_e$)', fontsize=7.0, color='white', weight='bold', va='center', ha='center', linespacing=1.05, zorder=4)
    d_ax.text(7.55, 2.32, '75.0%\nbaseload', fontsize=7.0, color=C['power'], weight='bold', va='center', ha='left', linespacing=1.04)
    d_ax.text(7.55, 1.56, '0.47% called\npeaking', fontsize=7.0, color=C['sand'], weight='bold', va='center', ha='left', linespacing=1.04)
    d_ax.text(7.55, 0.98, '24.53%\nuncalled', fontsize=7.0, color=C['teal'], weight='bold', va='center', ha='left', linespacing=1.04)

    ins_d = d_ax.inset_axes([0.06, -0.05, 0.58, 0.31])
    ins_d.set_facecolor('#FAFBFD')
    for sp_s in ('top', 'right'):
        ins_d.spines[sp_s].set_visible(False)
    ins_d.spines['left'].set_linewidth(0.45)
    ins_d.spines['bottom'].set_linewidth(0.45)
    called_pct = [((d[d.year == y].B - d[d.year == y].diverted) / d[d.year == y].B * 100.0 - 75.0).to_numpy() for y in YEARS]
    bp_d = ins_d.boxplot(called_pct, positions=range(4), widths=0.45, patch_artist=True, showfliers=False,
                         boxprops={'facecolor': C['sand'], 'alpha': 0.45, 'edgecolor': C['sand'], 'lw': 0.6},
                         medianprops={'color': INK, 'lw': 0.9})
    ins_d.set_xticks(range(4), ["'28", "'30", "'33", "'35"])
    ins_d.set_ylim(0.22, 0.82)
    ins_d.set_yticks([0.3, 0.5, 0.7])
    ins_d.tick_params(labelsize=7.0, length=1.6, pad=1.2)
    ins_d.text(0.98, 0.92, 'Called peaking (% of gas)', transform=ins_d.transAxes, fontsize=7.0, ha='right', va='top', color=MUTED)

    # e: Full-width Paired 36-Weather Box-and-Jitter + Upward Grid-Routing Gain Arrows
    u_inj = u[u.use == 'U4'].set_index('year').net
    e.axhline(0, color=INK, lw=0.6, zorder=1)
    e.plot(range(4), [u_inj.loc[y] for y in YEARS], color=C['inject'], ls='--', lw=0.95, marker='s', ms=3.0,
           label='100% direct gas-grid injection benchmark', zorder=3)
    for i_e, y in enumerate(YEARS):
        sub_v = v[v.year == y]
        vp = sub_v['net_GWP100_leak0.025'].to_numpy()
        vg = sub_v['net_with_diverted'].to_numpy()
        for dx_e, arr_e, col_e in [(-0.17, vp, C['power']), (0.17, vg, C['teal'])]:
            px_e = i_e + dx_e
            e.boxplot([arr_e], positions=[px_e], widths=0.24, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': col_e, 'alpha': 0.32, 'edgecolor': col_e, 'lw': 0.7},
                      medianprops={'color': INK, 'lw': 1.0}, whiskerprops={'color': col_e, 'lw': 0.65},
                      capprops={'color': col_e, 'lw': 0.65}, zorder=3)
            e.scatter(px_e + np.linspace(-0.06, 0.06, len(arr_e)), arr_e, s=7, color=col_e, alpha=0.68, lw=0, zorder=4)
        gain_e = vg.mean() - vp.mean()
        e.annotate('', xy=(i_e + 0.02, vg.mean()), xytext=(i_e + 0.02, vp.mean()),
                   arrowprops=dict(arrowstyle='->', lw=0.8, color=C['teal']), zorder=5)
        e.text(i_e + 0.04, (vg.mean() + vp.mean()) / 2.0, f'+{gain_e:.3f}', fontsize=7.0, va='center', ha='left', color='#0E5A4D', weight='bold')
    e.set_xticks(range(4), [str(y) for y in YEARS])
    e.set_xlim(-0.52, 3.52)
    e.set_ylim(-0.032, 0.228)
    e.set_xlabel('Target year')
    e.set_ylabel('Net abatement\n(t CO$_2$e per MWh)')
    e.legend(handles=[
        Patch(facecolor=C['power'], alpha=0.55, edgecolor=C['power'], label='Power-only accounting (uncalled gas idle)'),
        Patch(facecolor=C['teal'], alpha=0.55, edgecolor=C['teal'], label='Power + uncalled reserve gas routed to grid'),
        plt.Line2D([0], [0], color=C['inject'], ls='--', marker='s', ms=3.0, label='100% gas-grid injection (0.184–0.191)'),
    ], loc='center right', bbox_to_anchor=(0.99, 0.60), fontsize=7.0, handlelength=1.1, labelspacing=0.18)
    grid(e)

    finish(f, [a, b, c, d_ax, e], 'figED3_flexible_reserve')


def ed10_legacy_adequacy():
    v35 = SD / 'v35'

    def vsrc(fn):
        p = v35 / fn
        SOURCES[f'v35/{fn}'] = hashlib.sha256(p.read_bytes()).hexdigest()
        if fn.endswith('.csv'):
            return pd.read_csv(p)
        if fn.endswith('.json'):
            return json.loads(p.read_text())
        return np.load(p)

    PLAIN = FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else f'{v:g}')

    def plain_log(a_obj, x=True, y=True):
        for on, axis in [(x, a_obj.xaxis), (y, a_obj.yaxis)]:
            if on:
                axis.set_major_locator(LogLocator(base=10, numticks=6))
                axis.set_major_formatter(PLAIN)
                axis.set_minor_formatter(NullFormatter())

    s = vsrc('v35_summary.json')
    f = fig(198)
    a = ax(f, 3, 6, 80, 48)
    b = ax(f, 99, 8, 52, 42)
    c = ax(f, 15, 64, 62, 36)
    d = ax(f, 98, 64, 78, 36)
    e = ax(f, 15, 112, 62, 36)
    f_ax = ax(f, 124, 110, 52, 40)
    g = ax(f, 22, 165, 66, 25)
    h = ax(f, 108, 165, 68, 25)

    # --- a: Import corridors into shortfall zones ---
    cor = vsrc('mechanism_corridors.csv')
    zs = vsrc('zone_shares.csv')
    short = (zs.assign(cc=zs.zone.str[:2]).groupby('cc').share.sum() * 100.0).to_dict()
    short = {('UK' if k == 'GB' else k): v for k, v in short.items()}
    soft = colors.LinearSegmentedColormap.from_list('soft16', ['#FBF4E6', '#F1CFA0', '#E3A06A'])
    countrymap(a, short, cmap=soft, norm=colors.LogNorm(0.1, 70.0), zero_color='#F3F5F7')
    x0, y0 = laea(-8.5, 45.2)
    x1, _ = laea(17.5, 45.2)
    _, y1 = laea(8.0, 57.6)
    a.set_xlim(x0, x1)
    a.set_ylim(y0, y1)
    from matplotlib.transforms import Bbox
    from matplotlib.patches import Polygon as _Poly
    win = Bbox([[x0, y0], [x1, y1]])
    for patch in list(a.patches):
        if not isinstance(patch, _Poly):
            continue
        xy_p = patch.get_xy()
        if xy_p[:, 0].max() < x0 or xy_p[:, 0].min() > x1 or xy_p[:, 1].max() < y0 or xy_p[:, 1].min() > y1:
            patch.remove()
            continue
        clipped = patch.get_path().clip_to_bbox(win, inside=True)
        if clipped is None or len(clipped.vertices) < 3:
            patch.remove()
            continue
        patch.set_xy(clipped.vertices)
    LONLAT = {'FR00': (2.6, 47.0), 'BE00': (4.7, 50.6), 'DE00': (10.2, 51.0), 'DKW1': (9.1, 56.1), 'DKE1': (12.0, 55.4),
              'CZ00': (15.3, 49.8), 'AT00': (14.4, 47.5), 'UK00': (-1.6, 52.6), 'UKNI': (-6.6, 54.6)}
    SHORT_Z = {'DE00': 'DE', 'UK00': 'UK', 'FR00': 'FR', 'BE00': 'BE', 'DKW1': 'DK-W', 'DKE1': 'DK-E',
               'CZ00': 'CZ', 'AT00': 'AT', 'UKNI': 'NI'}
    FULL = colors.LinearSegmentedColormap.from_list('full16', ['#C9D6E2', '#9DB4C8', '#E7A35E', '#C4473A'])
    xy = {k: laea(*v) for k, v in LONLAT.items()}
    rad = {('UK00', 'DE00'): 0.25, ('DE00', 'UK00'): 0.25}
    for r_c in cor.itertuples():
        p0, p1 = xy[r_c.from_zone], xy[r_c.shortfall_zone]
        lw = 0.6 + 2.4 * np.sqrt(r_c.mean_capacity_MW / 8000.0)
        a.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad.get((r_c.from_zone, r_c.shortfall_zone), 0.12)}",
                                    arrowstyle='-|>', mutation_scale=6 + 1.8 * lw, lw=lw,
                                    color=FULL(r_c.share_hours_full), shrinkA=7, shrinkB=9, zorder=5, alpha=0.95))
    offs_a = {'FR00': (0, -95, 'center'), 'BE00': (-60, 42, 'right'), 'DKW1': (-70, 0, 'right'),
              'DKE1': (70, 0, 'left'), 'CZ00': (75, 0, 'left'), 'AT00': (75, -10, 'left'),
              'UKNI': (0, 72, 'center'), 'UK00': (-80, -60, 'right'), 'DE00': (80, 50, 'left')}
    for k, (px, py) in xy.items():
        big = k in ('DE00', 'UK00')
        a.scatter([px], [py], s=30 if big else 10, color=INK if big else 'white', edgecolor=INK, lw=0.6, zorder=6)
        dx, dy, ha_k = offs_a[k]
        a.text(px + dx, py + dy, SHORT_Z[k], fontsize=7.0, fontweight='bold' if big else 'normal',
               ha=ha_k, va='center', zorder=7, bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.78))
    lim = vsrc('mechanism_import_limits.csv')
    hrs = lim.groupby('zone').shortfall_hours.sum()
    a.text(0.48, 0.97, f"DE {int(hrs['DE00'])} h, UK {int(hrs['UK00'])} h", transform=a.transAxes,
           ha='center', va='top', fontsize=7.0, color=MUTED,
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=0.4))
    cax_a = a.inset_axes([0.03, 0.06, 0.24, 0.04])
    cb_a = plt.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 100), cmap=FULL), cax=cax_a, orientation='horizontal')
    cb_a.set_ticks([0, 100])
    cb_a.ax.tick_params(labelsize=7.0, length=1.5, pad=1)
    cax_a.set_title('Line full (%)', fontsize=7.0, pad=2)

    # --- b: Marimekko boost allocation ---
    mb = vsrc('mechanism_boost_by_weather.csv')
    order_b = [('resource', 'Resource'), ('eens', 'Shortage')]
    cols_b = [('boost_in_shortfall_zones', 'Shortfall zones', C['teal']),
              ('boost_in_neighbours', 'Adjacent', '#9CCFC4'), ('boost_elsewhere', 'Elsewhere', '#C9D0D6')]
    xb = 0.0
    for rule, lab_b in order_b:
        gb = mb[mb.rule == rule]
        tot = gb.boost_MWh.sum() / 1e3
        yb = 0.0
        for col_n, _, col_c in cols_b:
            sh = gb[col_n].sum() / gb.boost_MWh.sum() * 100.0
            b.add_patch(Rectangle((xb, yb), tot, sh, fc=col_c, ec='white', lw=0.7))
            if sh > 10:
                b.text(xb + tot / 2.0, yb + sh / 2.0, f'{sh:.0f}%', ha='center', va='center', fontsize=7.0,
                       color='white' if col_c == C['teal'] else INK)
            yb += sh
        net_b = gb.net_MWh.sum() / gb.boost_MWh.sum() * 100.0
        b.plot([xb, xb + tot], [net_b, net_b], color=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=5)
        b.text(xb + tot / 2.0, net_b - 3.0, f'{net_b:.0f}%', ha='center', va='top', fontsize=7.0, weight='bold',
               bbox=dict(boxstyle='square,pad=0.06', fc='white', ec='none', alpha=0.85), zorder=6)
        b.text(xb + tot / 2.0, 102.0, lab_b, ha='center', va='bottom', fontsize=7.0)
        xb += tot + 1.8
    b.set(xlim=(-0.3, xb - 1.5), ylim=(0, 100), xlabel='Boost energy, 20 yr (GWh)', ylabel='Boost share (%)')
    b.set_yticks([0, 50, 100])
    b.set_xticks([0, 10, 14.8, 24.8], ['0', '10', '0', '10'])
    b.legend(handles=[Patch(fc=col_c, ec='none', label=nm) for _, nm, col_c in cols_b],
             loc='center left', bbox_to_anchor=(1.01, 0.55), fontsize=7.0, handlelength=0.9)

    # --- c: Paired log-log scatter across 20 held-out shortfall years ---
    dfc = vsrc('weather_uncertainty_by_weather.csv')
    dc = dfc[dfc.resource_net_MWh > 1e-3]
    lo_c, hi_c = 8.0, 6000.0
    for k_m, lab_m in [(1, '1×'), (2, '2×'), (3, '3×')]:
        c.plot([lo_c, hi_c / k_m], [lo_c * k_m, hi_c], color='#B9C2CB' if k_m != 2 else C['red'], lw=0.65,
               ls='-' if k_m == 1 else (0, (3, 2)), zorder=1)
        c.text(hi_c / k_m, hi_c * 1.06, lab_m, ha='center', va='bottom', fontsize=7.0,
               color=C['red'] if k_m == 2 else MUTED)
    sc_c = c.scatter(dc.resource_net_MWh, dc.eens_net_MWh,
                     s=10 + 26 * np.sqrt(dc.baseline_EENS_MWh / dc.baseline_EENS_MWh.max()),
                     c=dc.DE_UK_share * 100.0, cmap=BLUE, vmin=80, vmax=100, edgecolor='white', lw=0.4, zorder=3)
    c.set(xscale='log', yscale='log', xlim=(lo_c, hi_c), ylim=(lo_c, hi_c),
          xlabel='Resource layout (MWh)', ylabel='Shortage layout (MWh)')
    plain_log(c)
    cax_c = c.inset_axes([0.58, 0.12, 0.36, 0.05])
    cb_c = plt.colorbar(sc_c, cax=cax_c, orientation='horizontal')
    cb_c.set_ticks([80, 100])
    cb_c.ax.tick_params(labelsize=7.0, length=1.5, pad=1)
    cax_c.set_title('DE+UK (%)', fontsize=7.0, pad=2)
    wu = vsrc('weather_uncertainty.json')
    c.text(0.04, 0.95, f"Median {wu['per_weather_ratio_median']:.2f}×", transform=c.transAxes, fontsize=7.0, va='top')

    # --- d: Bootstrap KDE + leave-one-out + single years ---
    boot = vsrc('weather_bootstrap_ratios.npy')
    xs = np.linspace(1.9, 3.1, 400)
    kde = gaussian_kde(boot)(xs)
    d.set(xlim=(1.9, 3.1), ylim=(-2.8, kde.max() * 1.22))
    lo_b, hi_b = wu['bootstrap_percentile_95']
    msk = (xs >= lo_b) & (xs <= hi_b)
    d.fill_between(xs[msk], 0, kde[msk], color=C['teal'], alpha=0.32, lw=0, zorder=2)
    d.plot(xs, kde, color=C['teal'], lw=1.1, zorder=3)
    d.plot([wu['point_ratio']] * 2, [0, kde.max() * 1.02], color=INK, lw=0.85, zorder=4)
    d.text(wu['point_ratio'], kde.max() * 1.06, f"{wu['point_ratio']:.2f} ({lo_b:.2f}–{hi_b:.2f})",
           ha='center', va='bottom', fontsize=7.0)
    d.axvline(2.0, color=C['red'], lw=0.75, ls=(0, (3, 2)))
    d.text(2.03, kde.max() * 0.55, '2.0× ratio', fontsize=7.0, color=C['red'], va='center')
    loo = [(dfc.eens_net_MWh.sum() - r_.eens_net_MWh) / (dfc.resource_net_MWh.sum() - r_.resource_net_MWh) for r_ in dfc.itertuples()]
    d.scatter(loo, np.full(len(loo), -0.85), marker='|', s=36, color=INK, lw=0.7, zorder=4)
    pw = dfc.ratio.dropna()
    jit = (np.arange(len(pw)) % 3 - 1) * 0.30
    d.scatter(pw, -2.0 + jit, s=9, color='white', edgecolor=C['navy'], lw=0.6, zorder=4)
    d.text(3.08, -0.85, 'LOO', ha='right', va='center', fontsize=7.0, color=MUTED)
    d.text(2.26, -2.0, 'Single yr', ha='right', va='center', fontsize=7.0, color=MUTED)
    d.set_yticks([])
    d.spines['left'].set_visible(False)
    d.set_xlabel('Placement ratio (shortage / resource)')

    # --- e: Stress boundary geography ---
    ge = vsrc('v35_boundary_geography.csv')
    ge = ge[~ge.boundary.str.contains('UK')]
    e.axvspan(23.0, 92.0, color='#EEF4F2', zorder=0, lw=0)
    e.text(46.0, 100.6, 'Calibration window', ha='center', va='bottom', fontsize=7.0, color=C['teal'])
    lab_e = {'main': 'Gas −30%, imp. 0.5', 'cut0.3_ext0.75': '−30%, 0.75',
             'cut0.3_ext1': '−30%, 1.0', 'cut0_ext0.5': '0%, 0.5',
             'cut0_ext0.75': '0%, 0.75', 'cut0.5_ext0.75': '−50%, 0.75'}
    off_e = {'main': (7, -9), 'cut0.3_ext0.75': (-7, 8), 'cut0.3_ext1': (7, 3),
             'cut0_ext0.5': (7, -6), 'cut0_ext0.75': (7, -6), 'cut0.5_ext0.75': (5, 10)}
    div_e = colors.LinearSegmentedColormap.from_list('deuk16', [C['blue'], '#E9EDF1', C['sand']])
    for r_e in ge.itertuples():
        xv = max(r_e.mean_shortage_hours, 0.3)
        yv = (r_e.DE_share + r_e.UK_share) * 100.0
        e.scatter([xv], [yv], s=8 + 14 * np.sqrt(r_e.mean_EENS_GWh),
                  c=[div_e(r_e.DE_share / (r_e.DE_share + r_e.UK_share))], edgecolor=INK, lw=0.5, zorder=3)
        dx, dy = off_e.get(r_e.boundary, (6, 0))
        e.annotate(lab_e.get(r_e.boundary, r_e.boundary), (xv, yv), xytext=(dx, dy), textcoords='offset points',
                   fontsize=7.0, ha='left' if dx > 0 else 'right', va='center', color=INK)
    e.set(xscale='log', xlim=(0.3, 180.0), ylim=(84, 100.5), xlabel='Mean system shortfall (h/yr)',
          ylabel='DE + UK shortfall (%)')
    plain_log(e, y=False)

    # --- f: Robustness forest across 10 sensitivity specs ---
    rob = vsrc('robustness.csv')
    loss = vsrc('b5r_losses.csv')
    coin = []
    for (_, _), gq in loss.groupby(['length_h', 'alpha']):
        gq = gq.set_index('strategy')
        if {'rule_eens', 'rule_resource'} <= set(gq.index):
            e_r, r_r = gq.loc['rule_eens'], gq.loc['rule_resource']
            coin.append((e_r.control_sum_MWh + e_r.aligned_minus_control_sum_MWh) / (r_r.control_sum_MWh + r_r.aligned_minus_control_sum_MWh))
    seeds = rob[rob.group == 'gas_bootstrap'].ratio.dropna()
    sc_s = s.get('scenarios', {})
    gr = (s.get('no_foresight') or {}).get('greedy_ratio', np.nan)

    def sr(k):
        v_s = (sc_s.get(k) or {}).get('ratio') or {}
        return v_s.get('ratio', np.nan), v_s.get('bootstrap95', [np.nan, np.nan])

    rows_f = [
        ('Bootstrap (20 yr)', wu['point_ratio'], *wu['bootstrap_percentile_95'], C['navy']),
        ('Leave-one-out', np.nan, *wu['leave_one_out_range'], C['navy']),
        ('Gas −30%, imp. 0.75', sr('ext075')[0], *sr('ext075')[1], C['sand']),
        ('Gas −50%, imp. 0.75', sr('cut05ext075')[0], *sr('cut05ext075')[1], C['sand']),
        ('UK local resource', sr('uk')[0], *sr('uk')[1], C['blue']),
        ('H$_2$ 25% budget', float(rob[rob.test == 'Hydrogen 25% budget'].ratio.iloc[0]), np.nan, np.nan, C['blue']),
        ('Detailed DSR', float(rob[rob.test == 'DSR detailed'].ratio.iloc[0]), np.nan, np.nan, C['blue']),
        ('Resampled gas (7)', np.nan, seeds.min(), seeds.max(), C['purple']),
        ('Low gas in stress', np.nan, min(coin), max(coin), C['purple']),
        ('No foresight', gr, np.nan, np.nan, C['teal']),
    ]
    for yi, (lab_f, pt, l_v, h_v, col_f) in enumerate(rows_f):
        if np.isfinite(l_v) and np.isfinite(h_v):
            f_ax.plot([l_v, h_v], [yi, yi], color=col_f, lw=1.8, solid_capstyle='round', alpha=0.65, zorder=2)
        if np.isfinite(pt):
            f_ax.scatter([pt], [yi], s=18, color=col_f, edgecolor='white', lw=0.45, zorder=3, marker='D')
        txt_v = f'{pt:.2f}' if np.isfinite(pt) else f'{l_v:.2f}–{h_v:.2f}'
        f_ax.text(3.38, yi, txt_v, fontsize=7.0, va='center', ha='right', color=INK)
    f_ax.axvline(2.0, color=C['red'], lw=0.75, ls=(0, (3, 2)), zorder=1)
    f_ax.set_yticks(range(len(rows_f)), [x[0] for x in rows_f])
    f_ax.tick_params(axis='y', length=0, labelsize=7.0)
    f_ax.set(xlim=(1.45, 3.42), ylim=(len(rows_f) - 0.4, -0.6), xlabel='Placement ratio')
    f_ax.set_xticks([1.5, 2.0, 2.5, 3.0])
    grid(f_ax, 'x')
    f_ax.spines['left'].set_visible(False)

    # --- g: Boost power allocation by zone ---
    lp = vsrc('layout_power.csv')
    lay = s.get('uk_layouts') or {}
    zones_g = ['DE00', 'UK00', 'CZ00', 'AT00', 'DK', 'FR00', 'IT', 'ES00', 'RO00', 'other']
    names_g = ['DE', 'UK', 'CZ', 'AT', 'DK', 'FR', 'IT', 'ES', 'RO', 'Oth.']

    def agg_g(zl, pw):
        zl = np.asarray(zl)
        pw = np.asarray(pw, float)
        out_g, used = [], np.zeros(len(zl), bool)
        for z in zones_g[:-1]:
            m_z = np.isin(zl, ['UK00', 'UKNI']) if z == 'UK00' else (np.char.startswith(zl.astype(str), z) if len(z) == 2 else zl == z)
            out_g.append(pw[m_z].sum())
            used |= m_z
        out_g.append(pw[~used].sum())
        return out_g

    rows_g = []
    for rule_g, lab_g in [('rule_resource', 'Resource'), ('rule_eens', 'Shortage')]:
        gg = lp[lp.strategy == rule_g]
        rows_g.append((lab_g, agg_g(gg.zone.values, gg.power_MW.values)))
    for rule_g, lab_g in [('resource', 'Res. (UK)'), ('eens', 'Short. (UK)')]:
        if rule_g in lay:
            rows_g.append((lab_g, agg_g(lay[rule_g]['zones'], lay[rule_g]['power'])))
    for i_g, (lab_g, vals_g) in enumerate(rows_g):
        for j_g, vg in enumerate(vals_g):
            if vg > 0.02:
                g.scatter([j_g], [i_g], s=vg * 7.2, color=C['teal'] if 'Short' in lab_g else '#8D99A6',
                          edgecolor='none', zorder=3, alpha=0.92)
            if vg >= 3.5:
                g.text(j_g, i_g + (0.0 if vg > 18 else 0.44), f'{vg:.1f}', ha='center', va='center',
                       fontsize=7.0, color='white' if vg > 18 else INK, zorder=4)
    g.set_xticks(range(len(names_g)), names_g)
    g.set_yticks(range(len(rows_g)), [r_[0] for r_ in rows_g])
    g.tick_params(length=0, labelsize=7.0)
    g.set(xlim=(-0.6, len(names_g) - 0.4), ylim=(len(rows_g) - 0.1, -0.65), xlabel='Boost power by zone (MW)')
    for sp in g.spines.values():
        sp.set_visible(False)
    g.axhline(1.5, color=GRID, lw=0.6)

    # --- h: Service retained without foresight ---
    df_nf = vsrc('v35_no_foresight_by_weather.csv')
    h.plot([16, 4800], [100, 100], color='#B9C2CB', lw=0.7, zorder=1)
    for rule_h, col_h, lab_h, mk_h in [('resource', '#8D99A6', 'Resource', 's'), ('eens', C['teal'], 'Shortage', 'o')]:
        gh = df_nf[(df_nf.rule == rule_h) & (df_nf.lp_net_MWh > 1e-3)]
        ret = 100.0 * gh.greedy_net_MWh / gh.lp_net_MWh
        h.scatter(gh.lp_net_MWh, ret, s=14, color=col_h, edgecolor='white', lw=0.4, zorder=3, label=lab_h, marker=mk_h)
    nf = s.get('no_foresight') or {}
    if nf:
        h.text(0.03, 0.08, f"Retained: {nf['eens']['retained']*100:.1f}% / {nf['resource']['retained']*100:.1f}% (ratio {nf['greedy_ratio']:.2f}×)",
               transform=h.transAxes, fontsize=7.0, va='bottom')
    h.set(xscale='log', xlim=(15, 5000), ylim=(96.5, 100.6), xlabel='Optimised reduction (MWh)', ylabel='No-foresight (%)')
    plain_log(h, y=False)
    h.set_yticks([97, 98, 99, 100])
    h.legend(loc='center left', bbox_to_anchor=(0.02, 0.56), fontsize=7.0, handletextpad=0.2)

    finish(f, [a, b, c, d, e, f_ax, g, h], 'figS1_legacy_adequacy')


def ed10_de_it_sites():
    sy = read52('site_year')
    ms = read52('matched_sites')
    sp_c = read52('site_partition_check')
    it_t = read52('pinerolo_thresholds')
    it_e = read52('pinerolo_yield_envelope')
    it_b = read52('pinerolo_energy_basis')
    de_t = read52('ruhleben_thresholds')

    f = fig(192)
    a = ax(f, 15, 9, 44, 42)
    b = ax(f, 75, 9, 44, 42)
    c = ax(f, 135, 9, 43, 42)

    d_ax = ax(f, 15, 68, 44, 42)
    e = ax(f, 75, 68, 44, 42)
    f_ax = ax(f, 135, 68, 43, 42)

    g = ax(f, 15, 130, 76, 44)
    h = ax(f, 108, 130, 70, 44)

    # a: Annual 8,760-h paired fossil displacement D (Pinerolo vs Berlin Ruhleben)
    specs_s = [('Pinerolo', 'Pinerolo (ITN1)', C['teal'], 'o', -0.10),
               ('Ruhleben', 'Ruhleben (DE00)', C['power'], 's', 0.10)]
    for st_k, st_lab, st_col, st_mk, dx in specs_s:
        sub = sy[sy.site == st_k]
        for w_id, gw in sub.groupby('weather'):
            gw = gw.sort_values('year')
            a.plot([0 + dx, 1 + dx], gw.D, color=st_col, alpha=0.25, lw=0.65)
        m28 = sub[sub.year == 2028].D.mean()
        m35 = sub[sub.year == 2035].D.mean()
        a.plot([0 + dx, 1 + dx], [m28, m35], color=st_col, marker=st_mk, ms=3.5, lw=1.3, label=st_lab, zorder=4)
        a.text(0 + dx, m28 + (0.025 if st_k == 'Pinerolo' else -0.050), f'{m28:.3f}', ha='center',
               va='bottom' if st_k == 'Pinerolo' else 'top', fontsize=7.0, color=st_col, weight='bold')
        a.text(1 + dx, m35 + (0.025 if st_k == 'Pinerolo' else -0.055), f'{m35:.3f}', ha='center',
               va='bottom' if st_k == 'Pinerolo' else 'top', fontsize=7.0, color=st_col, weight='bold')
    a.set_xticks([0, 1], ['2028', '2035'])
    a.set_xlim(-0.38, 1.38)
    a.set_ylim(-0.02, 0.78)
    a.set_xlabel('Target year')
    a.set_ylabel('Fossil displaced D\n(MWh/MWh raw gas)')
    a.legend(loc='lower left', fontsize=7.0, handlelength=1.0)
    grid(a)

    # b: Annual 8,760-h avoided emissions factor mu (t CO2e / MWhe)
    for st_k, st_lab, st_col, st_mk, dx in specs_s:
        sub = sy[sy.site == st_k]
        for yc_i, yr in enumerate([2028, 2035]):
            vals = sub[sub.year == yr].mu.to_numpy()
            pos = yc_i * 1.1 + (dx * 1.8)
            b.boxplot([vals], positions=[pos], widths=0.28, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': st_col, 'alpha': 0.30}, medianprops={'color': INK, 'lw': 1.0})
            b.scatter(pos + np.linspace(-0.06, 0.06, len(vals)), vals, s=10, color=st_col, zorder=4)
    b.set_xticks([0, 1.1], ['2028', '2035'])
    b.set_xlim(-0.45, 1.55)
    b.set_ylim(0.0, 0.52)
    b.set_xlabel('Target year')
    b.set_ylabel('Avoided emissions μ\n(t CO$_2$e per MWh$_e$)')
    grid(b)

    # c: Matched 1.635 MW nodal swap across 2028-2035 (ITN1 vs DE00)
    for st_k, st_lab, st_col, st_mk in [('Pinerolo', 'ITN1 (1.635 MW)', C['teal'], 'o'),
                                        ('Ruhleben', 'DE00 (1.635 MW)', C['power'], 's')]:
        gm = ms[ms.site == st_k].groupby('year').D.agg(['mean', 'min', 'max'])
        c.fill_between(gm.index, gm['min'], gm['max'], color=st_col, alpha=0.12, lw=0)
        c.plot(gm.index, gm['mean'], color=st_col, marker=st_mk, ms=3.0, label=st_lab)
    yearaxis(c, short=True)
    c.set_ylim(-0.03, 0.82)
    c.set_ylabel('Matched-power D\n(MWh/MWh)')
    c.legend(loc='upper right', fontsize=7.0, handlelength=1.0)
    grid(c)

    # d: Weekly vs monthly anchor partition verification (WS17)
    d_ax.plot([0.1, 0.72], [0.1, 0.72], color=GREY, ls='--', lw=0.75)
    for r_p in sp_c.itertuples():
        col_p = C['teal'] if r_p.site == 'Pinerolo' else C['power']
        mk_p = 'o' if r_p.year == 2028 else 's'
        d_ax.scatter(r_p.D, r_p.D_month, s=22, color=col_p, edgecolor=INK, lw=0.4, marker=mk_p, zorder=4)
        ox = -28 if (r_p.site == 'Pinerolo' and r_p.year == 2028) else 4
        oy = -3 if (r_p.site == 'Pinerolo' and r_p.year == 2028) else (-7 if r_p.year == 2028 else 4)
        d_ax.annotate(f"{r_p.site[:3]} '{str(r_p.year)[2:]}", (r_p.D, r_p.D_month),
                      xytext=(ox, oy), textcoords='offset points', fontsize=7.0)
    d_ax.text(0.05, 0.92, 'Max |ΔD| = 0.0079', transform=d_ax.transAxes, fontsize=7.0, color=MUTED, va='top')
    d_ax.set_xlim(0.08, 0.74)
    d_ax.set_ylim(0.08, 0.74)
    d_ax.set_xlabel('Weekly-anchored D')
    d_ax.set_ylabel('Monthly-anchored D')
    grid(d_ax)

    # e: Pinerolo (Italy) 2028 vs 2035 grid_minus_chp_at_q1 under measured heat (0.080 MWh/MWh_raw)
    it_med = it_t[(it_t.gas_basis == 'adjusted_median') & np.isclose(it_t.heat_delivery_fraction, 0.85)]
    slips = [(-0.032, 'Grid −3.2%', C['teal']), (0.0, 'Equal leak.', C['blue']), (0.032, 'Grid +3.2%', C['power'])]
    x_it = np.arange(2)
    e.axhline(0, color=INK, lw=0.6)
    for idx_s, (sl_v, sl_lab, sl_col) in enumerate(slips):
        vals_28 = it_med[(it_med.year == 2028) & np.isclose(it_med.leak_grid_minus_site, sl_v)].grid_minus_chp_at_q1
        vals_35 = it_med[(it_med.year == 2035) & np.isclose(it_med.leak_grid_minus_site, sl_v)].grid_minus_chp_at_q1
        pos_x = x_it + (idx_s - 1) * 0.24
        means_s = [vals_28.mean(), vals_35.mean()]
        e.bar(pos_x, means_s, width=0.21, color=sl_col, alpha=0.82, label=sl_lab)
        e.errorbar(pos_x, means_s,
                   yerr=[[means_s[0] - vals_28.min(), means_s[1] - vals_35.min()],
                         [vals_28.max() - means_s[0], vals_35.max() - means_s[1]]],
                   fmt='none', ecolor=INK, lw=0.65, capsize=1.6)
    e.set_xticks(x_it, ['2028', '2035'])
    e.set_ylim(-0.10, 0.13)
    e.set_xlabel('Pinerolo target year')
    e.set_ylabel('Grid minus CHP at q=1\n(t CO$_2$e/MWh raw)')
    e.legend(loc='upper left', fontsize=7.0, handlelength=0.8, labelspacing=0.18)
    grid(e)

    # f: Pinerolo 2035 parity gas-replacement threshold q* across yield quartiles & LHV basis
    q_med = it_e[(it_e.year == 2035) & (it_e.electric_basis == 'it_chp_electric') & (it_e.gas_basis == 'it_grid_adjusted')].qstar * 100.0
    q_all = it_e[it_e.year == 2035].qstar * 100.0
    q_lhv = it_b[(it_b.year == 2035) & np.isclose(it_b.input_reported_to_lhv, 1.0) & np.isclose(it_b.gas_reported_to_lhv, 0.90)].qstar * 100.0
    for idx_q, (arr_q, lab_q, col_q) in enumerate([
        (q_med, 'Median\nyield', C['teal']),
        (q_all, 'Q1–Q3\nyields', C['blue']),
        (q_lhv, '0.90 LHV\nproduct', C['sand']),
    ]):
        f_ax.boxplot([arr_q], positions=[idx_q], widths=0.42, patch_artist=True, showfliers=False,
                     boxprops={'facecolor': col_q, 'alpha': 0.32}, medianprops={'color': INK, 'lw': 1.0})
        f_ax.text(idx_q, arr_q.max() + 1.5, f'{arr_q.mean():.1f}%', ha='center', va='bottom', fontsize=7.0, weight='bold')
    f_ax.axhline(100, color=GREY, ls='--', lw=0.75)
    f_ax.set_xticks(range(3), ['Median\nyield', 'Q1–Q3\nspan', '0.90 LHV\nbasis'])
    f_ax.set_ylim(58, 106)
    f_ax.set_ylabel('Pinerolo 2035 parity $q^*$ (%)')
    grid(f_ax)

    # g: Berlin Ruhleben (Germany) 2035 grid_minus_chp_at_q1 across all 768 parameter combinations
    d35 = de_t[de_t.year == 2035]
    grp_de = [
        (d35[(d35.h == 0.0) & np.isclose(d35.leak_grid_minus_site, -0.032)].grid_minus_chp_at_q1, 'h=0\nΔλ−3.2%', C['teal']),
        (d35[(d35.h == 0.0) & np.isclose(d35.leak_grid_minus_site, 0.0)].grid_minus_chp_at_q1, 'h=0\nEqual', C['blue']),
        (d35[(d35.h == 0.0) & np.isclose(d35.leak_grid_minus_site, 0.032)].grid_minus_chp_at_q1, 'h=0\nΔλ+3.2%', C['sand']),
        (d35[(d35.h == 0.5) & np.isclose(d35.leak_grid_minus_site, -0.032)].grid_minus_chp_at_q1, 'h=50%\nΔλ−3.2%', C['teal']),
        (d35[(d35.h == 0.5) & np.isclose(d35.leak_grid_minus_site, 0.0)].grid_minus_chp_at_q1, 'h=50%\nEqual', C['blue']),
        (d35[(d35.h == 0.5) & np.isclose(d35.leak_grid_minus_site, 0.032)].grid_minus_chp_at_q1, 'h=50%\nΔλ+3.2%', C['power']),
    ]
    g.axhline(0, color=INK, lw=0.65)
    for idx_d, (arr_d, lab_d, col_d) in enumerate(grp_de):
        g.boxplot([arr_d], positions=[idx_d], widths=0.45, patch_artist=True, showfliers=False,
                  boxprops={'facecolor': col_d, 'alpha': 0.32}, medianprops={'color': INK, 'lw': 1.0})
    g.text(0.98, 0.95, 'Ruhleben 2035: 738 / 768 (96.1%) > 0', transform=g.transAxes,
           fontsize=7.0, weight='bold', ha='right', va='top')
    g.set_xticks(range(6), [x[1] for x in grp_de])
    g.set_ylim(-0.06, 0.34)
    g.set_ylabel('Grid minus CHP at q=1\n(t CO$_2$e/MWh raw)')
    grid(g)

    # h: Berlin Ruhleben 2028 vs 2035 parity gas-replacement threshold q*
    for idx_yr, (yr_h, col_h) in enumerate([(2028, C['navy']), (2035, C['teal'])]):
        for idx_hv, hv_d in enumerate([0.0, 0.5]):
            pos_h = idx_yr * 2.2 + idx_hv * 0.85
            arr_h = de_t[(de_t.year == yr_h) & np.isclose(de_t.h, hv_d) & np.isclose(de_t.leak_grid_minus_site, 0.0)].qstar * 100.0
            h.boxplot([arr_h], positions=[pos_h], widths=0.52, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': col_h, 'alpha': 0.32}, medianprops={'color': INK, 'lw': 1.0})
            h.text(pos_h, arr_h.max() + 3.5, f'{np.median(arr_h):.0f}%', ha='center', va='bottom', fontsize=7.0, weight='bold')
    h.axhline(100, color=GREY, ls='--', lw=0.75)
    h.set_xticks([0.0, 0.85, 2.2, 3.05], ["'28\nh=0", "'28\nh=50%", "'35\nh=0", "'35\nh=50%"])
    h.set_ylim(0, 196)
    h.set_ylabel('Ruhleben parity $q^*$ (%, equal leak.)')
    grid(h)

    finish(f, [a, b, c, d_ax, e, f_ax, g, h], 'figED8_de_it_sites')


if __name__ == '__main__':
    jobs = {
        '1': ed1_system_resources, '2': ed2_weather_methane, '3': ed9_flexible_reserve,
        '4': ed3_temporal_prices, '5': ed4_factorial, '6': ed6_robustness,
        '7': ed5_country_spillover, '8': ed10_de_it_sites, '9': ed7_product_boundaries,
        '10': ed8_costs_pathways, '11': ed10_legacy_adequacy,
    }
    for k in sys.argv[1:] or jobs:
        jobs[k]()
