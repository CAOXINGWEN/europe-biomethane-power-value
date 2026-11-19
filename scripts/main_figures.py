"""Redesigned 8-panel Nature-grade main figures for manuscript revision v17 (asymmetric showcase layouts, zero numeric heatmaps, zero lollipop charts)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.interpolate import PchipInterpolator
from matplotlib import colors
from matplotlib.patches import Rectangle, Patch
from style import (
    plt, fig, ax, label, grid, yearaxis, band, laea, countrymap, export, read,
    C, YC, YC_WARM, SEQ, BLUE, WARM, DIV, INK, MUTED, GREY, GRID, MM, ribbon
)

YEARS = [2028, 2030, 2033, 2035]


def fig1_trajectory():
    d = read('boundary_base_by_weather')
    u = read('uses')
    fl = read('fuel')
    f = fig(192)

    # Row 1: a (conversion ribbons), b (hero raincloud 36 weathers), c (paired interval step changes)
    a = ax(f, 2, 9, 49, 49)
    b = ax(f, 65, 9, 64, 49)
    c = ax(f, 144, 9, 34, 49)

    # Row 2: d (displaced fossil stream + coal-exit inset), e (pathway net abatement), f (site-power 36-weather box+dots)
    d_ax = ax(f, 14, 74, 45, 46)
    e = ax(f, 74, 74, 59, 46)
    f_ax = ax(f, 148, 74, 30, 46)

    # Row 3: g (break-even methane intensity GWP100 vs GWP20), h (resource expansion vs shrinking total displacement)
    g = ax(f, 14, 136, 73, 44)
    h = ax(f, 105, 136, 73, 44)

    # --- a: Conversion & displacement flow ribbons per MWh biomethane ---
    a.set_xlim(0, 10)
    a.set_ylim(-0.2, 3.5)
    a.axis('off')
    d28 = d[d.year == 2028].D.mean()
    d35 = d[d.year == 2035].D.mean()
    dsl = float(u[(u.year == 2028) & (u.use == 'U5_CNG_diesel')].D.iloc[0])
    ys = {'site': 2.55, 'inj': 1.42, 'cng': 0.38}
    ribbon(a, ys['site'], [0.2, 2.3, 3.4, 4.7, 6.2, 7.1], [1.0, 1.0, 0.40, 0.40, d28, d28], C['power'], H=0.68, alpha=0.40, z=2)
    ribbon(a, ys['site'], [0.2, 2.3, 3.4, 4.7, 6.2, 7.1], [1.0, 1.0, 0.40, 0.40, d35, d35], C['power'], H=0.68, alpha=0.95, z=3)
    ribbon(a, ys['inj'], [0.2, 7.1], [1.0, 1.0], C['inject'], H=0.68, alpha=0.88, z=2)
    ribbon(a, ys['cng'], [0.2, 2.3, 4.7, 7.1], [1.0, 1.0, dsl, dsl], C['truck'], H=0.68, alpha=0.88, z=2)
    a.text(0.2, ys['site'] + 0.39, 'Site power (40% elec.)', fontsize=7.0, va='bottom', ha='left')
    a.text(0.2, ys['inj'] + 0.39, 'Gas grid injection', fontsize=7.0, va='bottom', ha='left')
    a.text(0.2, ys['cng'] + 0.39, 'CNG truck (vs diesel)', fontsize=7.0, va='bottom', ha='left')
    a.text(4.05, ys['site'] - 0.25, '0.40 MWh', fontsize=7.0, ha='center', va='top', color=MUTED)
    a.text(7.25, ys['site'] + 0.14, f"'28: {d28:.2f}", fontsize=7.0, va='center', ha='left', color=C['power'])
    a.text(7.25, ys['site'] - 0.12, f"'35: {d35:.2f}", fontsize=7.0, va='center', ha='left', weight='bold')
    a.text(7.25, ys['inj'], '1.00 gas', fontsize=7.0, va='center', ha='left')
    a.text(7.25, ys['cng'], f'{dsl:.2f} diesel', fontsize=7.0, va='center', ha='left')
    a.text(0.2, 3.46, '1 MWh input', fontsize=7.0, va='top', ha='left', color=MUTED)
    a.text(9.85, 3.46, 'Displaced\nfossil (MWh)', fontsize=7.0, va='top', ha='right', color=MUTED, linespacing=1.08)

    # --- b: Hero raincloud across 36 paired weather years ---
    wide = d.pivot(index='weather', columns='year', values='D')
    order = wide[2028].rank(method='first').astype(int)
    off = {w: -0.60 + 0.42 * (rk - 1) / 35.0 for w, rk in order.items()}
    for w in wide.index:
        b.plot([y + off[w] for y in YEARS], [wide.loc[w, y] for y in YEARS], color=GREY, lw=0.4, alpha=0.55, zorder=1)
    for y in YEARS:
        v = wide[y].to_numpy()
        b.scatter([y + off[w] for w in wide.index], v, s=9, color=YC_WARM[y], edgecolor='white', lw=0.3, zorder=3)
        kde = gaussian_kde(v)
        gy = np.linspace(v.min() - 0.018, v.max() + 0.018, 120)
        dens = kde(gy)
        dens = dens / dens.max() * 0.62
        b.fill_betweenx(gy, y + 0.06, y + 0.06 + dens, color=YC_WARM[y], alpha=0.52, lw=0, zorder=2)
        b.plot([y + 0.06, y + 0.06], [gy.min(), gy.max()], color=YC_WARM[y], lw=0.6, zorder=2)
        mu = v.mean()
        b.plot([y + 0.06, y + 0.06 + 0.62], [mu, mu], color=INK, lw=1.1, zorder=4)
        b.text(y + 0.36, v.max() + 0.025, f'{mu:.3f}', fontsize=7.0, va='bottom', ha='center', weight='bold')
    b.axhline(1.0, color=C['inject'], lw=1.0, ls='--', zorder=1)
    b.text(2027.15, 1.02, 'Gas grid displaces 1.00 MWh fossil gas', fontsize=7.0, color=C['inject'], va='bottom', ha='left')
    b.text(2027.15, 0.04, '36/36 years decline at every step (−65.5%)', fontsize=7.0, color=MUTED, va='bottom', ha='left')
    b.set_xlim(2026.9, 2036.4)
    b.set_ylim(0.0, 1.13)
    b.set_xticks(YEARS)
    b.set_xlabel('Target year')
    b.set_ylabel('Fossil fuel displaced\n(MWh per MWh biomethane)')
    grid(b)

    # --- c: Paired step changes per weather year ---
    steps = [(2028, 2030), (2030, 2033), (2033, 2035), (2028, 2035)]
    for i, (y0, y1) in enumerate(steps):
        v = (wide[y1] - wide[y0]).to_numpy()
        xs = np.full(len(v), float(i))
        srt = np.argsort(v)
        bins = np.round(v[srt] / 0.008)
        for bk in np.unique(bins):
            idx = srt[bins == bk]
            nb = len(idx)
            xs[idx] = i + (np.arange(nb) - (nb - 1) / 2.0) * 0.06
        c.scatter(xs, v, s=9, color=YC_WARM[y1], edgecolor='white', lw=0.25, zorder=3)
        c.plot([i - 0.32, i + 0.32], [v.mean()] * 2, color=INK, lw=1.1, zorder=4)
        c.text(i, v.min() - 0.012, f'{v.mean():.2f}', fontsize=7.0, ha='center', va='top', weight='bold')
    c.axhline(0, color=INK, lw=0.6)
    c.set_xticks(range(4), ['28–\n30', '30–\n33', '33–\n35', '28–\n35'])
    c.set_xlabel('Interval (20xx)')
    c.set_ylabel('Paired change ΔD (MWh/MWh)')
    c.set_xlim(-0.6, 3.6)
    c.set_ylim(-0.43, 0.03)
    grid(c)

    # --- d: Smooth stacked fossil-carrier stream (MWh/MWh) + embedded Coal & Oil phase-out zoom inset ---
    fm = fl.groupby('year').mean(numeric_only=True)
    fl_tot_w = fl.assign(tot=fl['gas'] + fl['Lignite'] + fl['Hard coal'] + fl['Light oil'] + fl['Heavy oil'])
    tot_env = fl_tot_w.groupby('year').tot.agg(['min', 'max'])
    carriers = [
        ('Gas', fm['gas'].to_numpy(), '#C95443'),
        ('Lignite', fm['Lignite'].to_numpy(), '#7A4935'),
        ('Hard coal', fm['Hard coal'].to_numpy(), '#2F3640'),
        ('Oil', (fm['Light oil'] + fm['Heavy oil']).to_numpy(), '#DF9F45'),
    ]
    xs_yr = np.array(YEARS, float)
    xf_yr = np.linspace(2028, 2035, 160)
    base_d = np.zeros_like(xf_yr)
    for cname, cvals, ccol in carriers:
        yy = np.clip(PchipInterpolator(xs_yr, cvals)(xf_yr), 0.0, None)
        d_ax.fill_between(xf_yr, base_d, base_d + yy, color=ccol, alpha=0.88, lw=0, zorder=2, label=cname)
        d_ax.plot(xf_yr, base_d + yy, color='white', lw=0.35, zorder=3)
        base_d = base_d + yy
    lo_f = PchipInterpolator(xs_yr, tot_env['min'].to_numpy())(xf_yr)
    hi_f = PchipInterpolator(xs_yr, tot_env['max'].to_numpy())(xf_yr)
    d_ax.plot(xf_yr, hi_f, color=INK, lw=0.55, ls=':', alpha=0.7, zorder=4)
    d_ax.plot(xf_yr, lo_f, color=INK, lw=0.55, ls=':', alpha=0.7, zorder=4)
    d_ax.text(2028.25, 0.22, 'Gas\n0.47 → 0.18', fontsize=7.0, color='white', weight='bold', va='center', ha='left', linespacing=1.08, zorder=5)
    d_ax.set_xlim(2028, 2035)
    d_ax.set_ylim(0.0, 0.62)
    yearaxis(d_ax, short=True)
    d_ax.set_ylabel('Displaced fossil fuel\n(MWh/MWh)')
    d_ax.legend(loc='lower left', bbox_to_anchor=(-0.04, 1.01), ncol=2, fontsize=7.0, handlelength=0.85, columnspacing=0.8, labelspacing=0.18)
    grid(d_ax)

    # Embedded zoom inset in upper-right of d_ax showing Lignite + Hard coal + Oil collapse (0.061 -> <0.001)
    ins = d_ax.inset_axes([0.50, 0.50, 0.47, 0.46])
    ins.set_facecolor('#FAFBFD')
    for sp_side in ('top', 'right'):
        ins.spines[sp_side].set_visible(False)
    ins.spines['left'].set_linewidth(0.45)
    ins.spines['bottom'].set_linewidth(0.45)
    xi_c = np.arange(4)
    bot_c = np.zeros(4)
    for cname, cvals, ccol in carriers[1:]:
        ins.bar(xi_c, cvals * 1000.0, bottom=bot_c, width=0.56, color=ccol, alpha=0.92, edgecolor='white', lw=0.25)
        bot_c = bot_c + cvals * 1000.0
    ins.text(0, bot_c[0] + 2.0, '61', fontsize=7.0, ha='center', va='bottom', weight='bold')
    ins.text(3, bot_c[3] + 2.0, '<1', fontsize=7.0, ha='center', va='bottom', color=MUTED)
    ins.set_xticks(xi_c, ["'28", "'30", "'33", "'35"])
    ins.set_yticks([0, 30, 60])
    ins.set_ylim(0, 78)
    ins.tick_params(labelsize=7.0, length=1.6, pad=1.2)
    ins.text(0.98, 0.94, 'Coal & oil\n(kWh/MWh)', transform=ins.transAxes, fontsize=7.0, ha='right', va='top', color=MUTED, linespacing=1.02)

    # --- e: Net abatement across competing biomethane pathways ---
    specs = [
        ('U5_CNG_diesel', 'CNG / diesel', C['truck'], '^', '-', 1.2),
        ('U4', 'Gas grid', C['inject'], 's', '-', 1.2),
        ('U2+', 'Reserve + grid', C['reserve'], 'D', '--', 1.0),
        ('U1', 'Site power', C['power'], 'o', '-', 1.3),
        ('U2', 'Flex. reserve', C['reserve'], 'v', '-', 1.0),
    ]
    end_y = {'U5_CNG_diesel': 0.228, 'U4': 0.191, 'U2+': 0.048, 'U1': 0.004, 'U2': -0.025}
    for key, lab, col, mk, ls, lw in specs:
        g_u = u[u.use == key].sort_values('year')
        if (g_u.net_hi - g_u.net_lo).max() > 1e-4:
            e.fill_between(g_u.year, g_u.net_lo, g_u.net_hi, color=col, alpha=0.14, lw=0, zorder=1)
        e.plot(g_u.year, g_u.net, color=col, ls=ls, lw=lw, marker=mk, ms=3.0, zorder=3)
        e.text(2035.35, end_y[key], lab, fontsize=7.0, color=col, va='center', ha='left')
    e.axhline(0, color=INK, lw=0.6, zorder=2)
    e.set_xlim(2027.5, 2040.2)
    e.set_ylim(-0.045, 0.265)
    e.set_xticks(YEARS)
    e.set_xticklabels(["'28", "'30", "'33", "'35"])
    e.set_xlabel('Target year')
    e.set_ylabel('Net abatement\n(t CO$_2$e per MWh biomethane)')
    grid(e)

    # --- f: Site-power net abatement distribution across 36 weathers ---
    vals = [d[d.year == y].net_primary.to_numpy() for y in YEARS]
    bp = f_ax.boxplot(vals, positions=np.arange(4), widths=0.50, patch_artist=True,
                      showfliers=False, medianprops={'color': INK, 'lw': 1.1})
    for i, (v, box) in enumerate(zip(vals, bp['boxes'])):
        col = YC_WARM[YEARS[i]]
        box.set_facecolor(col)
        box.set_alpha(0.30)
        f_ax.scatter(i + np.linspace(-0.17, 0.17, len(v)), v, s=7, color=col, lw=0, alpha=0.72, zorder=3)
    f_ax.axhline(0, color=C['red'], ls='--', lw=0.85, zorder=2)
    f_ax.text(2.92, -0.025, '30/36\n< 0', fontsize=7.0, color=C['red'], ha='center', va='top', linespacing=1.05)
    f_ax.set_xticks(range(4), ["'28", "'30", "'33", "'35"])
    f_ax.set_xlabel('Target year')
    f_ax.set_ylabel('Site-power net abatement\n(t CO$_2$e per MWh)')
    f_ax.set_ylim(-0.045, 0.108)
    grid(f_ax)

    # --- g: Break-even methane emission intensity (GWP100 vs GWP20) ---
    d_lam = d.copy()
    d_lam['lam100'] = d_lam.lambda_zero_primary * 100.0
    d_lam['lam20'] = d_lam.avoided20 / (0.072 * 79.7) * 100.0
    g.axhspan(1.0, 4.2, color='#EAEFF4', lw=0, zorder=0)
    g.axhline(2.5, color=INK, ls='--', lw=0.8, zorder=2)
    band(g, d_lam, 'lam100', C['power'], 'GWP$_{100}$')
    band(g, d_lam, 'lam20', C['navy'], 'GWP$_{20}$', ls='-.', marker='s')
    g.text(2035.15, 4.38, 'Central 2.5% (range 1.0–4.2%)', fontsize=7.0, ha='right', va='bottom', color=MUTED)
    g.text(2035.15, 7.65, 'Gas grid: ~12%', fontsize=7.0, ha='right', va='top', color=C['inject'])
    g.set_xlim(2027.7, 2035.3)
    g.set_ylim(0, 8.3)
    yearaxis(g)
    g.set_ylabel('Break-even methane\nintensity (%)')
    g.legend(loc='upper left', ncol=2, fontsize=7.0, handlelength=1.4, columnspacing=1.0)
    grid(g)

    # --- h: Food-waste resource expansion vs total displaced fossil energy ---
    d_tot = d.assign(B_TWh=d.B_MWh / 1e6, F_TWh=d.B_MWh * d.D / 1e6)
    qB = d_tot.groupby('year').B_TWh.mean()
    qF = d_tot.groupby('year').F_TWh.agg(['mean', 'min', 'max'])
    x_pos = np.arange(4)
    h.bar(x_pos - 0.18, qB.loc[YEARS], width=0.33, color=C['inject'], alpha=0.85, label='Biomethane resource (= gas grid)')
    h.bar(x_pos + 0.18, qF.loc[YEARS, 'mean'], width=0.33, color=C['power'], alpha=0.88, label='Site-power fossil fuel displaced')
    h.errorbar(x_pos + 0.18, qF.loc[YEARS, 'mean'],
               yerr=[qF.loc[YEARS, 'mean'] - qF.loc[YEARS, 'min'], qF.loc[YEARS, 'max'] - qF.loc[YEARS, 'mean']],
               fmt='none', ecolor=INK, elinewidth=0.75, capsize=2.0, zorder=4)
    for i, y in enumerate(YEARS):
        h.text(i - 0.18, qB.loc[y] + 0.18, f'{qB.loc[y]:.2f}', fontsize=7.0, ha='center', va='bottom', color=C['inject'])
        h.text(i + 0.18, qF.loc[y, 'max'] + 0.18, f'{qF.loc[y, "mean"]:.2f}', fontsize=7.0, ha='center', va='bottom', color=C['power'])
    h.set_xticks(x_pos, [str(y) for y in YEARS])
    h.set_xlabel('Target year')
    h.set_ylabel('Annual energy (TWh yr$^{-1}$)')
    h.set_ylim(0, 9.6)
    h.legend(loc='upper left', fontsize=7.0, handlelength=1.1)
    grid(h)

    for axis_obj, ltr in zip([a, b, c, d_ax, e, f_ax, g, h], 'abcdefgh'):
        label(f, axis_obj, ltr)
    export(f, 'fig1_trajectory')


def fig2_mechanism():
    z = read('response_allocation')
    pq = read('site_price_quantiles')
    pb = read('price_bands')
    fl = read('flows')
    sh = read('factor_shapley_by_weather')
    m = read('mechanism_by_weather')
    it = read('factor_interactions_by_weather')
    old = read('legacy_sensitivities')
    sp = read('spec_checks')

    f = fig(201)
    # Row 1: a (alluvial response shares), b (price duration curves), c (avoided-emission waterfall)
    a = ax(f, 13, 9, 53, 46)
    b = ax(f, 81, 9, 49, 46)
    c = ax(f, 144, 9, 34, 46)

    # Row 2: d (system response factor ridgelines), e (paired 2^3 factorial box + weather-shift cloud), f (3-factor Shapley)
    d_ax = ax(f, 11, 82, 40, 44)
    e = ax(f, 65, 82, 62, 44)
    f_ax = ax(f, 146, 82, 32, 44)

    # Row 3: g (Möbius interactions across 8 weathers), h (dispatch controls: nuclear fixed + no flex load)
    g = ax(f, 14, 148, 58, 43)
    h = ax(f, 92, 148, 65, 43)

    # --- a: Alluvial-style destination shares of added site electricity ---
    cat = [
        ('Gas', C['power'], 'Gas'),
        ('Coal and oil', C['brown'], 'Coal, oil'),
        ('Imports', C['grey'], 'Imports'),
        ('Nuclear', C['purple'], 'Nuclear'),
        ('Curtailment', C['green'], 'Curtailment'),
        ('Electrolysis', C['blue'], 'Electrolysis'),
        ('Power-to-heat', '#8EC0D6', 'P2H'),
        ('Hydrogen power', C['sand'], 'H$_2$ power'),
        ('Other response', C['light'], 'Other'),
    ]
    means = z.groupby('year').mean(numeric_only=True) * 100.0
    xs = np.array(YEARS, float)
    xf = np.linspace(2028, 2035, 180)
    base = np.zeros_like(xf)
    for k, col, _ in cat[::-1]:
        yy = PchipInterpolator(xs, means[k].to_numpy())(xf)
        a.fill_between(xf, base, base + yy, color=col, lw=0, alpha=0.95)
        a.plot(xf, base + yy, color='white', lw=0.35)
        if means.loc[2035, k] >= 9.0:
            a.text(2035.15, base[-1] + yy[-1] / 2.0, f'{means.loc[2035, k]:.0f}%', fontsize=7.0, va='center', ha='left')
        if means.loc[2028, k] >= 10.0:
            tc = 'white' if k in ('Gas', 'Nuclear') else INK
            a.text(2028.12, base[0] + yy[0] / 2.0, f'{means.loc[2028, k]:.0f}%', fontsize=7.0, va='center', ha='left', color=tc)
        base = base + yy
    for y in YEARS:
        a.axvline(y, color='white', lw=0.6, ls=':')
    a.set_xlim(2028, 2036.8)
    a.set_ylim(0, 100)
    a.set_xticks(YEARS)
    a.set_xticklabels(["'28", "'30", "'33", "'35"])
    a.set_xlabel('Target year')
    a.set_ylabel('Response / added electricity (%)')

    lg_ax = ax(f, 13, 64, 165, 9)
    lg_ax.axis('off')
    lg_ax.legend(
        handles=[Patch(facecolor=col, edgecolor='#CBD5E1' if k == 'Other response' else 'none', lw=0.3, label=lab) for k, col, lab in cat],
        loc='center left', ncol=5, fontsize=7.0, handlelength=0.9, columnspacing=0.9, labelspacing=0.25
    )

    # --- b: Counterfactual nodal price duration curves at site hours ---
    for y in YEARS:
        t = pq[pq.year == y].sort_values('quantile', ascending=False)
        b.plot(100.0 * (1.0 - t['quantile']), t.price.clip(upper=240.0), color=YC_WARM[y], lw=1.15, label=str(y), zorder=3)
        gc = float(pb[pb.year == y].gas_cost.iloc[0])
        sh_gas = float(pb[pb.year == y].at_or_above_gas.iloc[0])
        b.plot([100.0 * sh_gas], [gc], marker='o', ms=3.8, mfc='white', mec=YC_WARM[y], mew=1.0, zorder=5)
    b.axhspan(50.7, 79.4, color=C['blue'], alpha=0.10, lw=0, zorder=1)
    b.axhline(15.6, color=C['purple'], lw=0.7, ls='--', zorder=1)
    b.text(2.5, 65, 'Flex. load bids', fontsize=7.0, color=C['blue'], va='center', ha='left')
    b.text(2.5, 19, 'Nuclear', fontsize=7.0, color=C['purple'], va='bottom', ha='left')
    b.text(98, 228, 'Circles: gas cost\n(57% → 20%)', fontsize=7.0, color=MUTED, ha='right', va='top', linespacing=1.08)
    b.legend(loc='upper right', bbox_to_anchor=(1.0, 0.80), fontsize=7.0, handlelength=1.1, labelspacing=0.2)
    b.set_xlim(0, 100)
    b.set_ylim(0, 240)
    b.set_xlabel('Site electricity sorted by price (%)')
    b.set_ylabel('Counterfactual nodal price\n(EUR per MWh)')
    grid(b)

    # --- c: Avoided-emissions decline decomposition waterfall (S vs I) ---
    fg = fl.groupby('year')[['avoided', 's_f']].mean()
    sf = fg['s_f']
    mm = fg['avoided'] / (0.4 * sf)
    xpos = 0
    xt, xl = [], []
    prev = None
    for i, y in enumerate(YEARS):
        tot = fg.loc[y, 'avoided']
        if prev is not None:
            ds = 0.4 * (sf[y] - sf[prev]) * (mm[prev] + mm[y]) / 2.0
            dm = 0.4 * (mm[y] - mm[prev]) * (sf[prev] + sf[y]) / 2.0
            top = fg.loc[prev, 'avoided']
            for dv, col, tag in [(ds, C['power'], 'S'), (dm, C['brown'], 'I')]:
                c.add_patch(Rectangle((xpos - 0.34, top + dv), 0.68, -dv, facecolor=col, alpha=0.85, edgecolor='none'))
                c.plot([xpos + 0.34, xpos + 0.66], [top + dv] * 2, color=GREY, lw=0.45)
                xt.append(xpos)
                xl.append(tag)
                top += dv
                xpos += 1
        c.add_patch(Rectangle((xpos - 0.34, 0), 0.68, tot, facecolor=YC_WARM[y], edgecolor='none'))
        c.text(xpos, tot + 0.004, f'{tot:.3f}', fontsize=7.0, ha='center', va='bottom')
        if i < 3:
            c.plot([xpos + 0.34, xpos + 0.66], [tot] * 2, color=GREY, lw=0.45)
        xt.append(xpos)
        xl.append(f"'{str(y)[2:]}")
        xpos += 1
        prev = y
    c.set_xlim(-0.85, xpos - 0.3)
    c.set_ylim(0, 0.172)
    c.set_xticks(xt, xl)
    c.set_ylabel('Avoided emissions\n(t CO$_2$e per MWh)')
    c.text(0.98, 0.96, 'S: fossil share (92%)\nI: intensity (8%)', transform=c.transAxes, fontsize=7.0, ha='right', va='top', linespacing=1.1)
    grid(c)

    # --- d: System response factor ridgeline KDE across 36 weathers ---
    mef = fl.assign(mu=fl.avoided / 0.4)
    gx = np.linspace(0.05, 0.56, 220)
    for i, y in enumerate(YEARS[::-1]):
        v = mef[mef.year == y].mu.to_numpy()
        kd = gaussian_kde(v)(gx)
        kd = kd / kd.max() * 0.80
        d_ax.fill_between(gx, i, i + kd, color=YC_WARM[y], alpha=0.78, lw=0)
        d_ax.plot(gx, i + kd, color=YC_WARM[y], lw=0.7)
        d_ax.plot([v.mean()] * 2, [i, i + 0.80], color=INK, lw=0.9)
        d_ax.text(0.555, i + 0.16, f"{y} ({v.mean():.2f})", fontsize=7.0, ha='right', va='bottom')
    d_ax.set_xlim(0.05, 0.56)
    d_ax.set_ylim(-0.1, 4.1)
    d_ax.set_yticks([])
    d_ax.spines['left'].set_visible(False)
    d_ax.set_xlabel('System response factor $\\mu$\n(t CO$_2$e per MWh elec.)')

    # --- e: Paired 2^3 factorial box & weather-shift plot across 4 (B, C) regimes (V=2028 vs V=2035) ---
    regimes = [
        (2028, 2028, 'B28 · C28'),
        (2028, 2035, 'B28 · C35'),
        (2035, 2028, 'B35 · C28'),
        (2035, 2035, 'B35 · C35'),
    ]
    e.axvspan(1.5, 3.6, color='#F2F6FA', lw=0, zorder=0)
    e.axvline(1.5, color=GREY, ls=':', lw=0.65, zorder=1)
    col_v28 = C['navy']
    col_v35 = C['teal']
    for idx_r, (bg_y, pr_y, _) in enumerate(regimes):
        sub_r = m[(m.background_year == bg_y) & (m.price_year == pr_y)]
        piv_r = sub_r.pivot(index='weather', columns='vre_year', values='D')
        x28 = idx_r - 0.18
        x35 = idx_r + 0.18
        for w_id in piv_r.index:
            e.plot([x28 + 0.05, x35 - 0.05], [piv_r.loc[w_id, 2028], piv_r.loc[w_id, 2035]],
                   color=GREY, lw=0.5, alpha=0.55, zorder=2)
        for xv, vy, cv in [(x28, 2028, col_v28), (x35, 2035, col_v35)]:
            arr_v = piv_r[vy].to_numpy()
            e.boxplot([arr_v], positions=[xv], widths=0.26, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': cv, 'alpha': 0.30, 'edgecolor': cv, 'lw': 0.7},
                      medianprops={'color': INK, 'lw': 1.0}, whiskerprops={'color': cv, 'lw': 0.65},
                      capprops={'color': cv, 'lw': 0.65}, zorder=3)
            e.scatter(xv + np.linspace(-0.06, 0.06, len(arr_v)), arr_v, s=9, color=cv, edgecolor='white', lw=0.25, zorder=4)
            va_lbl = 'bottom' if vy == 2028 else 'top'
            dy_lbl = (arr_v.max() + 0.015) if vy == 2028 else (arr_v.min() - 0.016)
            e.text(xv, dy_lbl, f'{arr_v.mean():.3f}', ha='center', va=va_lbl, fontsize=7.0, color=cv, weight='bold')
    e.set_xticks(range(4), ['B28\nC28', 'B28\nC35', 'B35\nC28', 'B35\nC35'])
    e.set_xlim(-0.55, 3.55)
    e.set_ylim(0.07, 0.70)
    e.set_xlabel('Background (B) and fossil-cost (C) year')
    e.set_ylabel('Fossil displaced D (MWh/MWh)')
    e.legend(handles=[Patch(facecolor=col_v28, alpha=0.55, edgecolor=col_v28, label='2028 wind/solar (V28)'),
                      Patch(facecolor=col_v35, alpha=0.55, edgecolor=col_v35, label='2035 wind/solar (V35)')],
             loc='lower left', bbox_to_anchor=(-0.02, 1.01), ncol=2, fontsize=7.0, handlelength=0.85, columnspacing=0.7)
    grid(e)

    # --- f: Exact 3-factor Shapley contributions with 8 weather dots ---
    t_sh = sh[(sh.kind == 'mechanism') & (sh.metric == 'D')]
    labs = [('vre_year', 'Wind/\nsolar V', C['green']), ('price_year', 'Fossil\ncosts C', C['sand']), ('background_year', 'Other\ninputs B', C['blue'])]
    for i, (k, lab, col) in enumerate(labs):
        v = t_sh[t_sh.factor == k].effect.to_numpy()
        f_ax.barh(i, v.mean(), color=col, alpha=0.68, height=0.56)
        f_ax.scatter(v, i + np.linspace(-0.16, 0.16, len(v)), s=13, facecolors='white', edgecolors=col, lw=0.8, zorder=4)
        if k == 'vre_year':
            f_ax.text(v.mean() / 2.0, i, f'{v.mean():+.3f}', ha='center', va='center', fontsize=7.0, weight='bold')
        elif k == 'price_year':
            f_ax.text(0.015, i, f'{v.mean():+.3f}', ha='left', va='center', fontsize=7.0)
        else:
            f_ax.text(-0.015, i, f'{v.mean():+.3f}', ha='right', va='center', fontsize=7.0)
    f_ax.axvline(0, color=INK, lw=0.65)
    f_ax.set_yticks(range(3), [x[1] for x in labs])
    f_ax.invert_yaxis()
    f_ax.set_xlim(-0.34, 0.12)
    f_ax.set_xlabel('Shapley effect\n(MWh per MWh)')
    grid(f_ax, 'x')

    # --- g: Möbius interaction effects across 8 weathers ---
    it_d = it[(it.kind == 'mechanism') & (it.metric == 'D')]
    ifacs = [
        ('background_year+vre_year', 'B×V'),
        ('background_year+price_year', 'B×C'),
        ('vre_year+price_year', 'V×C'),
        ('background_year+vre_year+price_year', 'B×V×C'),
    ]
    for i, (fk, flab) in enumerate(ifacs):
        v = it_d[it_d.factors == fk].effect.to_numpy()
        col = C['teal'] if v.mean() > 0 else C['power']
        g.bar(i, v.mean(), width=0.52, color=col, alpha=0.35, edgecolor=col, lw=0.7, zorder=2)
        g.scatter(i + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=col, edgecolor='white', lw=0.3, zorder=3)
        yp = (v.max() + 0.008) if v.mean() > 0 else (v.min() - 0.008)
        g.text(i, yp, f'{v.mean():+.3f}', ha='center', va='bottom' if v.mean() > 0 else 'top', fontsize=7.0)
    g.axhline(0, color=INK, lw=0.6)
    g.set_xticks(range(4), [x[1] for x in ifacs])
    g.set_xlabel('Möbius interaction term')
    g.set_ylabel('Interaction (MWh/MWh)')
    g.set_ylim(-0.118, 0.085)
    grid(g)

    # --- h: Non-fossil channel rerouting under dispatch controls ---
    row = old[old.variant == 'S-nuc nuclear fixed'].iloc[0]
    sp_sub = sp[sp.variant == 'No electrolysers or power-to-heat']
    keys = [('Gas', 'gas', C['power']), ('Nuclear', 'nuclear', C['purple']), ('Curtailment', 'curtailment', C['green']), ('Flex. load', 'electrolyser_p2h', C['blue'])]
    for i, (lab, k, col) in enumerate(keys):
        vals = [row['base_sh_' + k] * 100.0, row['sh_' + k] * 100.0]
        h.bar(np.array([0, 1]) + (i - 1.5) * 0.18, vals, width=0.17, color=col, label=lab, zorder=3)
    h.axvline(1.55, color=GREY, ls='--', lw=0.7)
    h.set_xticks([0, 1, 2.05, 2.55, 3.05, 3.55], ['Disp.\nnuc.', 'Fixed\nnuc.', "'28", "'30", "'33", "'35"])
    h.set_xlim(-0.55, 3.95)
    h.set_ylim(0, 74)
    h.set_ylabel('Response share (%)')
    h.legend(loc='lower left', bbox_to_anchor=(0.12, 1.01), ncol=2, fontsize=7.0, handlelength=0.8, columnspacing=0.6, labelspacing=0.2)
    grid(h)

    h2 = h.twinx()
    h2.spines['top'].set_visible(False)
    h2.spines['left'].set_visible(False)
    h2.spines['right'].set_linewidth(0.55)
    h2.yaxis.tick_right()
    h2.yaxis.set_label_position('right')
    for j_y, y in enumerate(YEARS):
        sub_y = sp_sub[sp_sub.year == y]
        diff_y = (sub_y.D_variant - sub_y.D_main).to_numpy()
        x_sc = 2.05 + j_y * 0.50
        h2.scatter(np.full(len(diff_y), x_sc) + np.linspace(-0.09, 0.09, len(diff_y)), diff_y, s=9, color=YC_WARM[y], alpha=0.82, lw=0, zorder=4)
        h2.plot([x_sc - 0.14, x_sc + 0.14], [diff_y.mean()] * 2, color=INK, lw=0.95, zorder=5)
    h2.axhline(0, xmin=0.47, xmax=1.0, color=INK, lw=0.5, ls=':')
    h2.set_ylim(-0.025, 0.033)
    h2.set_yticks([-0.02, 0.0, 0.02])
    h2.set_ylabel('ΔD without flex. load\n(MWh/MWh)', fontsize=7.0)
    h2.text(2.8, 0.026, 'max |ΔD| = 0.021', fontsize=7.0, ha='center', color=MUTED)

    for axis_obj, ltr in zip([a, b, c, d_ax, e, f_ax, g, h], 'abcdefgh'):
        label(f, axis_obj, ltr)
    export(f, 'fig2_mechanism')


def fig3_spatial_scale():
    co = read('country')
    cr = read('country_crossing')
    res = read('resource')
    sup = read('superposition')
    sc_chk = read('country_scale_checks')
    s = read('scale_by_weather')
    sh = read('factor_shapley_summary')
    sh_w = read('factor_shapley_by_weather')

    f = fig(202)
    # Row 1: a (Europe 2035 D map), b (2028 vs 2035 bubble scatter), c (2035 country destination 100% stacked)
    a = ax(f, 2, 13, 64, 52)
    b = ax(f, 80, 13, 46, 48)
    c = ax(f, 140, 13, 35, 48)

    # Row 2: d (domestic fossil share 28 vs 35), e (superposition + 5x scale)
    d_ax = ax(f, 13, 81, 76, 44)
    e = ax(f, 108, 81, 70, 44)

    # Row 3: f (food vs fleet country shares), g (2x2 scale x mix across 8 weathers), h (Shapley waterfall)
    g_w = ax(f, 15, 142, 48, 46)
    g_2x2 = ax(f, 77, 142, 46, 46)
    h = ax(f, 138, 142, 40, 46)

    # --- a: European choropleth map of 2035 single-country D ---
    v35 = co[co.year == 2035].set_index('country').D.to_dict()
    v35 = {('UK' if k == 'GB' else k): val for k, val in v35.items()}
    sm = countrymap(a, v35, cmap=BLUE, vmin=0.0, vmax=0.50)
    cb = f.colorbar(sm, cax=ax(f, 8, 67, 52, 2.5), orientation='horizontal')
    cb.set_label('2035 fossil fuel displaced (MWh/MWh)', fontsize=7.0)
    cb.ax.tick_params(labelsize=7.0)
    for k, lon, lat in [('IT', 12.5, 42.5), ('DE', 10.4, 51.0), ('FR', 2.4, 46.6), ('UK', -1.8, 53.0), ('ES', -3.7, 40.0), ('GR', 22.0, 39.2)]:
        x, y = laea(lon, lat)
        if k == 'IT':
            a.annotate(f'IT {v35[k]:.2f}', xy=(x, y), xytext=laea(18.5, 43.0), ha='left', va='center', fontsize=7.0,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=0.8),
                       arrowprops=dict(arrowstyle='-', lw=0.6, color=INK))
        elif k == 'GR':
            a.annotate(f'GR {v35[k]:.2f}', xy=(x, y), xytext=laea(25.2, 36.8), ha='center', va='center', fontsize=7.0,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=0.8),
                       arrowprops=dict(arrowstyle='-', lw=0.6, color=INK))
        elif k == 'UK':
            a.annotate(f'UK {v35[k]:.2f}', xy=(x, y), xytext=laea(-9.2, 57.2), ha='center', va='center', fontsize=7.0,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.90, pad=0.6),
                       arrowprops=dict(arrowstyle='-', lw=0.6, color=INK))
        else:
            a.text(x, y, f'{k}\n{v35[k]:.2f}', ha='center', va='center', fontsize=7.0,
                   color='white' if v35[k] > 0.28 else INK,
                   bbox=None if v35[k] > 0.28 else dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4))

    # --- b: 2028 vs 2035 country trajectory bubble scatter ---
    dw = co.pivot(index='country', columns='year', values='D')
    r35 = res[(res.year == 2035) & (res.scenario == 'Central')].set_index('country_code').deliverable_TWh
    BINS = [('before 2028', '#7A2E22'), ('2028–2030', '#C4473A'), ('2030–2033', '#E8A07A'), ('after 2035', '#1F9E89')]
    ccol = dict(BINS)
    cmap_cr = cr.set_index('country').interval.to_dict()
    for fac, lab_f, x1 in [(1.0, '1:1', 0.62), (0.5, '−50%', 0.78), (0.25, '−75%', 0.78)]:
        xx = np.array([0.04, x1])
        b.plot(xx, fac * xx, color=GREY, lw=0.65, ls='--' if fac < 1 else '-', zorder=1)
        b.text(x1 - 0.01, fac * x1 + (0.01 if fac < 1 else -0.02), lab_f, fontsize=7.0, color=MUTED, ha='right', va='bottom' if fac < 1 else 'top')
    for cnt in dw.index:
        b.scatter(dw.loc[cnt, 2028], dw.loc[cnt, 2035], s=10 + 52 * r35[cnt], color=ccol[cmap_cr[cnt]], edgecolor=INK, lw=0.4, alpha=0.92, zorder=3)
    off_b = {'IT': (-0.03, 0.0), 'GR': (-0.03, 0.0), 'ES': (0.025, 0.018), 'FR': (0.03, -0.022), 'DE': (-0.045, 0.052), 'GB': (0.025, 0.01), 'HR': (-0.02, 0.028)}
    for cnt, (dx, dy) in off_b.items():
        b.text(dw.loc[cnt, 2028] + dx, dw.loc[cnt, 2035] + dy, 'UK' if cnt == 'GB' else cnt, fontsize=7.0,
               ha='left' if dx > 0 else 'right', va='center')
    b.text(0.56, -0.022, 'RO, BG', fontsize=7.0, ha='center', va='top')
    b.set_xlim(0.04, 0.79)
    b.set_ylim(-0.045, 0.62)
    b.set_xlabel('2028 fossil displaced (MWh/MWh)')
    b.set_ylabel('2035 fossil displaced (MWh/MWh)')
    grid(b)

    # --- c: 2035 single-country electricity destination shares (100% horizontal stacked) ---
    K = [('gas', 'Gas', C['power']), ('coal_lignite', 'Coal', C['brown']), ('nuclear', 'Nuc.', C['purple']),
         ('imports', 'Imp.', C['grey']), ('curtailment', 'Curt.', C['green']), ('electrolyser_p2h', 'Flex.', C['blue']), ('rest', 'Oth.', C['light'])]
    t35 = co[co.year == 2035].set_index('country').copy()
    t35['rest'] = (t35.oil + t35.other).clip(lower=0)
    P = t35[[k for k, _, _ in K]].clip(lower=0)
    P = P.div(P.sum(axis=1), axis=0).loc[dw[2035].sort_values().index]
    for i, cnt in enumerate(P.index):
        left = 0.0
        for k, lab_k, col_k in K:
            val = 100.0 * P.loc[cnt, k]
            c.barh(i, val, left=left, height=0.76, color=col_k, edgecolor='white', lw=0.25, label=lab_k if i == 0 else None)
            left += val
    c.set_yticks(range(len(P)), ['UK' if k == 'GB' else k for k in P.index])
    c.set_xlim(0, 100)
    c.set_xlabel('2035 response share (%)')
    c.tick_params(axis='y', length=0)
    c.legend(loc='lower right', bbox_to_anchor=(1.02, 1.02), ncol=4, fontsize=7.0, handlelength=0.75, columnspacing=0.55, labelspacing=0.18)

    # --- d: Cross-border decoupling (Signed domestic fossil share 2028 vs 2035 across 15 countries) ---
    dom = co.pivot(index='country', columns='year', values='domestic_fossil_share') * 100.0
    ord_dom = dom[2028].sort_values(ascending=True).index.tolist()
    x_dom = np.arange(len(ord_dom))
    d_ax.axhline(0, color=INK, lw=0.55, zorder=2)
    d_ax.bar(x_dom - 0.18, dom.loc[ord_dom, 2028], width=0.34, color=C['navy'], alpha=0.85, label='2028', zorder=3)
    d_ax.bar(x_dom + 0.18, dom.loc[ord_dom, 2035], width=0.34, color=C['teal'], alpha=0.85, label='2035', zorder=3)
    for k_zero in ('BG', 'RO'):
        if k_zero in ord_dom:
            xz = ord_dom.index(k_zero) + 0.18
            d_ax.text(xz, 2.0, 'n/a', ha='center', va='bottom', fontsize=7.0, color=MUTED, rotation=90)
    d_ax.set_xticks(x_dom, ['UK' if k == 'GB' else k for k in ord_dom])
    d_ax.set_ylim(-9, 106)
    d_ax.set_xlabel('Injecting country (ordered by 2028 domestic share)')
    d_ax.set_ylabel('Domestic share of\nfossil displacement (%)')
    d_ax.legend(loc='upper left', ncol=2, fontsize=7.0, handlelength=1.0)
    grid(d_ax)

    # --- e: Superposition (sum of 15 countries vs simultaneous) + 5x single-country check ---
    lim_e = (sup[['isolated_TWh', 'simultaneous_TWh']].min().min() * 0.90, sup[['isolated_TWh', 'simultaneous_TWh']].max().max() * 1.04)
    e.plot(lim_e, lim_e, color=GREY, lw=0.8, ls='--', zorder=1)
    for y in YEARS:
        sy = sup[sup.year == y]
        e.scatter(sy.simultaneous_TWh, sy.isolated_TWh, s=16, color=YC_WARM[y], edgecolor=INK, lw=0.35, label=str(y), zorder=3)
    e.set_xlim(*lim_e)
    e.set_ylim(*lim_e)
    e.set_xlabel('Simultaneous 15-country displacement (TWh)')
    e.set_ylabel('Sum of single-country runs (TWh)')
    e.legend(loc='upper left', ncol=2, fontsize=7.0, handletextpad=0.2, columnspacing=0.6)
    de5 = sc_chk[sc_chk.variant == 'DE single country x5'].iloc[0]
    si5 = sc_chk[sc_chk.variant == 'SI single country x5'].iloc[0]
    e.text(0.97, 0.06, f'Sum/simul. ratio: {sup.ratio.min():.3f}–{sup.ratio.max():.3f}\n5× scale: DE {de5.rel_mean:+.1f}%, SI {si5.rel_mean:+.1f}%',
           transform=e.transAxes, ha='right', va='bottom', fontsize=7.0, color=MUTED, linespacing=1.15)
    grid(e)

    # --- f (g_w): National share shift (Food waste 7.30 TWh vs Existing biogas 151.41 TWh) ---
    r35_all = res[res.year == 2035]
    fw = r35_all[r35_all.scenario == 'Central'].set_index('country_code').deliverable_TWh
    fl_sh = r35_all[r35_all.scenario == 'fleet'].set_index('country_code').fleet_fuel_TWh
    fw_pct = fw / fw.sum() * 100.0
    fl_pct = fl_sh / fl_sh.sum() * 100.0
    top5 = ['DE', 'IT', 'GB', 'FR', 'CZ']
    rows_w = []
    for k in top5:
        rows_w.append(('UK' if k == 'GB' else k, fw_pct[k], fl_pct[k]))
    rows_w.append(('Other 10', fw_pct.drop(top5).sum(), fl_pct.drop(top5).sum()))
    y_w = np.arange(len(rows_w))
    g_w.barh(y_w - 0.18, [r_[1] for r_ in rows_w], height=0.34, color=C['teal'], label='Food waste (7.3 TWh)')
    g_w.barh(y_w + 0.18, [r_[2] for r_ in rows_w], height=0.34, color=C['purple'], label='Existing biogas (151.4 TWh)')
    g_w.set_yticks(y_w, [r_[0] for r_ in rows_w])
    g_w.invert_yaxis()
    g_w.set_xlim(0, 65)
    g_w.set_xlabel('Country share (%)')
    g_w.legend(loc='center right', bbox_to_anchor=(1.0, 0.48), fontsize=7.0, handlelength=0.9)
    grid(g_w, 'x')

    # --- g (g_2x2): 2035 2x2 Scale x Composition across all 8 weather years ---
    cells = [
        ('food', 'small', 'Food\n7.3', C['blue']),
        ('food', 'large', 'Food\n151.4', C['blue']),
        ('fleet', 'small', 'Fleet\n7.3', C['purple']),
        ('fleet', 'large', 'Fleet\n151.4', C['teal']),
    ]
    for i, (mx, tot_k, clab, col) in enumerate(cells):
        sub = s[(s.mix == mx) & (s.total == tot_k)].D.to_numpy()
        g_2x2.boxplot([sub], positions=[i], widths=0.45, patch_artist=True, showfliers=False,
                      boxprops={'facecolor': col, 'alpha': 0.28}, medianprops={'color': INK, 'lw': 1.0})
        g_2x2.scatter(i + np.linspace(-0.12, 0.12, len(sub)), sub, s=11, color=col, edgecolor='white', lw=0.25, zorder=4)
        g_2x2.text(i, sub.max() + 0.008, f'{sub.mean():.3f}', ha='center', va='bottom', fontsize=7.0, weight='bold')
    g_2x2.set_xticks(range(4), [c_[2] for c_ in cells])
    g_2x2.set_xlabel('Composition / total (TWh)')
    g_2x2.set_ylim(0.06, 0.245)
    g_2x2.set_ylabel('2035 fossil displaced\n(MWh per MWh)')
    grid(g_2x2)

    # --- h: 2-factor Shapley waterfall decomposition ---
    g_m = s.groupby(['mix', 'total']).D.mean()
    t_s = sh[(sh.kind == 'scale') & (sh.metric == 'D')].set_index('factor')
    start = g_m.loc[('food', 'small')]
    end = g_m.loc[('fleet', 'large')]
    mx_eff = t_s.loc['mix', 'mean']
    tot_eff = t_s.loc['total', 'mean']
    mx_w = sh_w[(sh_w.kind == 'scale') & (sh_w.metric == 'D') & (sh_w.factor == 'mix')].effect
    tot_w = sh_w[(sh_w.kind == 'scale') & (sh_w.metric == 'D') & (sh_w.factor == 'total')].effect
    vals_h = [start, mx_eff, tot_eff, end]
    base_h = [0.0, start, start + mx_eff, 0.0]
    cols_h = [C['blue'], C['purple'], C['sand'], C['teal']]
    for i, (val, bot, col) in enumerate(zip(vals_h, base_h, cols_h)):
        h.bar(i, val, bottom=bot, width=0.58, color=col, zorder=3)
        top = bot + val
        lbl = f'{val:+.3f}\n({val / (end - start) * 100:.1f}%)' if i in (1, 2) else f'{val:.3f}'
        ypos = (start + 0.005) if i == 1 else (max(bot, top) + 0.005)
        h.text(i, ypos, lbl, ha='center', va='bottom', fontsize=7.0)
        if i < 3:
            h.plot([i + 0.29, i + 0.71], [top] * 2, color=GREY, lw=0.6)
    h.errorbar([1, 2], [start + mx_eff, start + mx_eff + tot_eff],
               yerr=[[mx_eff - mx_w.min(), tot_eff - tot_w.min()], [mx_w.max() - mx_eff, tot_w.max() - tot_eff]],
               fmt='none', ecolor=INK, elinewidth=0.75, capsize=2.0, zorder=4)
    h.set_xticks(range(4), ['Food\nsmall', 'Country\nshares', 'Total\nsupply', 'Fleet\nlarge'])
    h.set_ylabel('Fossil displaced (MWh/MWh)')
    h.set_ylim(0, 0.245)
    grid(h)

    for axis_obj, ltr in zip([a, b, d_ax, e, g_w, g_2x2, h], 'abdefgh'):
        label(f, axis_obj, ltr)
    label(f, c, 'c', x_abs=134.5, y_abs=5.0)
    export(f, 'fig3_spatial_scale')


def fig4_boundaries():
    d = read('boundary_base_by_weather')
    s = read('boundary_year_summary')
    co = read('country')
    ht = read('CHP_heat_counterfactuals')
    j = read('boundary_joint_summary')
    cum = read('cumulative')

    # Asymmetric Nature showcase layout (180 mm x 196 mm):
    # Top hero zone: Left stacked pair (a, c) + Right enlarged 2x2 Europe map quartet (b: 110 x 77 mm!)
    f = fig(196)
    a = ax(f, 14, 6.5, 45, 31)
    c = ax(f, 14, 50.5, 45, 29.5)
    b_maps = [ax(f, 68 + 56 * (k % 2), 6.5 + 39.5 * (k // 2), 54, 37.5) for k in range(4)]

    # Middle zone: d (36-weather box+jitter across heat counterfactuals), e (co-product single bars), f (grouped credit bars + gas-grid lead)
    d_ax = ax(f, 14, 102, 41, 34)
    e = ax(f, 77, 102, 36, 34)
    f_ax = ax(f, 127, 102, 51, 34)

    # Bottom zone: g (annual cohort bars + cumulative opportunity-bridge waterfall), h (period-stacked displaced fossil TWh)
    g_yr = ax(f, 14, 152, 51, 33.5)
    g_tot = ax(f, 77, 152, 27, 33.5)
    h = ax(f, 125, 152, 53, 33.5)

    # --- a: h0 (zero net abatement) vs h* (match gas-grid use) ---
    a.axhspan(0, 100, color=C['teal'], alpha=0.08, lw=0)
    q = d.copy()
    q['hmatch'] = q.h_match_injection_primary * 100.0
    q['hzero'] = q.h_zero_primary * 100.0
    band(a, q, 'hmatch', C['teal'], 'Match grid ($h_*$)')
    band(a, q, 'hzero', C['blue'], 'Net zero ($h_0$)', marker='s')
    a.axhline(100, color=GREY, ls='--', lw=0.75)
    a.axhline(0, color=GREY, lw=0.5)
    a.text(2035.1, 152, '148%', fontsize=7.0, color=C['teal'], ha='right', va='bottom', weight='bold')
    a.text(2035.1, 7, '3%', fontsize=7.0, color=C['blue'], ha='right', va='bottom', weight='bold')
    a.set_ylim(-85, 208)
    yearaxis(a, short=True)
    a.set_ylabel('Heat threshold (%)')
    a.legend(loc='upper left', fontsize=7.0, handlelength=1.0, labelspacing=0.18)
    grid(a)

    # --- c: 2035 Net abatement surface (h x lambda) ---
    r35 = s[s.year == 2035].iloc[0]
    heat_coef = 0.488 * 0.243023076 / 0.9
    hx = np.linspace(0, 1.0, 101)
    ly = np.linspace(0, 0.05, 101)
    H_grid, L_grid = np.meshgrid(hx, ly)
    Z = r35.avoided_domestic + H_grid * heat_coef - L_grid * 0.072 * 27.0
    im_c = c.pcolormesh(H_grid * 100.0, L_grid * 100.0, Z, cmap=DIV,
                        norm=colors.TwoSlopeNorm(vmin=Z.min(), vcenter=0.0, vmax=Z.max()),
                        shading='auto', rasterized=True)
    cs = c.contour(H_grid * 100.0, L_grid * 100.0, Z, levels=[0.0], colors=INK, linewidths=0.95)
    c.clabel(cs, fmt={0.0: 'Zero'}, fontsize=7.0)
    c.plot(25.0, 2.5, 'o', color='white', mec=INK, ms=3.4)
    c.text(29.0, 2.45, '25%, 2.5%', fontsize=7.0, va='center', ha='left')
    c.set_xlabel('Effective heat use (%)')
    c.set_ylabel('Methane intensity (%)')
    cb_c = f.colorbar(im_c, cax=ax(f, 14, 89.2, 45, 2.0), orientation='horizontal')
    cb_c.set_ticks([-0.04, 0.0, 0.08, 0.16])
    cb_c.set_label('2035 net abatement (t/MWh)', fontsize=7.0)
    cb_c.ax.tick_params(labelsize=7.0)

    # --- b: Enlarged 2x2 European choropleth showcase of national h* (5-tier diverging feasibility scale) ---
    hb = [
        ('≤50%', '#116A5B'),
        ('50–100%', '#76C7B7'),
        ('100–135%', '#F0C689'),
        ('135–160%', '#E08660'),
        ('>160%', '#B03A2E'),
    ]
    cm_b = colors.ListedColormap([x[1] for x in hb])
    nm_b = colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5)
    badges_b = {
        2028: '2028 · 13/15 ≤ 100%',
        2030: '2030 · 2/15 (IT, GR)',
        2033: '2033 · 1/15 (IT)',
        2035: '2035 · 1/15 (IT 94%)',
    }
    for ax_m, y in zip(b_maps, YEARS):
        t = co[co.year == y].set_index('country').h_star.to_dict()
        cls = {}
        for cnt, hv in t.items():
            k_c = 'UK' if cnt == 'GB' else cnt
            bin_v = 1 + (hv > 0.50) + (hv > 1.00) + (hv > 1.35) + (hv > 1.60)
            cls[k_c] = float(bin_v)
        countrymap(ax_m, cls, cmap=cm_b, norm=nm_b)
        (x0, x1), (y0, y1) = ax_m.get_xlim(), ax_m.get_ylim()
        ax_m.text(x0 + 0.02 * (x1 - x0), y1 - 0.025 * (y1 - y0), badges_b[y],
                  fontsize=7.0, weight='bold', va='top', ha='left', zorder=6,
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.90, pad=0.7))
        if y == 2028:
            for c_k, lon_k, lat_k, col_txt in [('DE', 10.3, 51.0, INK), ('FR', 2.4, 46.6, INK), ('IT', 12.6, 42.3, INK)]:
                xm, ym = laea(lon_k, lat_k)
                val_pct = int(round(t[c_k] * 100))
                ax_m.text(xm, ym, f'{c_k} {val_pct}%', fontsize=7.0, ha='center', va='center', color=col_txt,
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.76, pad=0.35), zorder=6)
        elif y == 2035:
            x_it, y_it = laea(12.6, 42.4)
            ax_m.annotate('IT 94%', xy=(x_it, y_it), xytext=laea(18.6, 43.2), ha='left', va='center', fontsize=7.0, weight='bold',
                          color='#0E5A4D', bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=0.5),
                          arrowprops=dict(arrowstyle='-', lw=0.55, color=INK), zorder=6)
            for c_k, lon_k, lat_k in [('DE', 10.3, 51.1), ('FR', 2.4, 46.6)]:
                xm, ym = laea(lon_k, lat_k)
                val_pct = int(round(t[c_k] * 100))
                ax_m.text(xm, ym, f'{c_k} {val_pct}%', fontsize=7.0, ha='center', va='center', color=INK, weight='bold',
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.82, pad=0.35), zorder=6)
            x_gr, y_gr = laea(22.0, 39.1)
            ax_m.annotate('GR 122%', xy=(x_gr, y_gr), xytext=laea(23.5, 36.4), ha='center', va='center', fontsize=7.0,
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=0.4),
                          arrowprops=dict(arrowstyle='-', lw=0.55, color=INK), zorder=6)

    lg_b = ax(f, 68, 85.0, 110, 6.5)
    lg_b.axis('off')
    lg_b.legend(
        handles=[Patch(facecolor=col, edgecolor='#64748B', lw=0.35, label=lab_h) for lab_h, col in hb],
        title='Effective heat recovery $h_*$ needed to match gas-grid injection:',
        title_fontsize=7.0, loc='center', ncol=5, fontsize=7.0, handlelength=0.85, columnspacing=0.75
    )

    # --- d: 2035 Heat counterfactual comparison via 36-weather box-and-jitter distributions (NO line chart!) ---
    ht35 = ht[ht.year == 2035]
    zc_vals = ht35[(ht35.heat_counterfactual == 'zero_carbon_heat') & np.isclose(ht35.h, 0.0)].net
    d_ax.axhspan(zc_vals.min(), zc_vals.max(), color='#E2E8F0', alpha=0.85, lw=0, zorder=1)
    d_ax.axhline(0.0, color=INK, lw=0.55, zorder=2)
    inj_35_mean = float(j[(j.year == 2035) & np.isclose(j.leakage, 0.025) & np.isclose(j.h, 0.0) &
                          (j.H2_kgCO2_per_kg == 0) & np.isclose(j.import_tCO2_per_MWhe, 0.0) &
                          (j.H2_energy_basis == 'LHV_assumption') & (j.compression_accounting == 'registered_domestic')].injection_net_mean.iloc[0])
    d_ax.axhline(inj_35_mean, color=C['inject'], ls='--', lw=0.85, zorder=2)
    d_ax.text(2.42, inj_35_mean + 0.005, 'Gas grid: 0.191', fontsize=7.0, color=C['inject'], ha='right', va='bottom', weight='bold')

    tiers_h = [0.25, 0.50, 1.00]
    for idx_h, hv in enumerate(tiers_h):
        for dx_h, cf_key, cf_col in [(-0.17, 'gas_boiler_90pct', C['teal']), (0.17, 'electric_heat_pump_COP3_domestic_MEF', C['blue'])]:
            arr_cf = ht35[(ht35.heat_counterfactual == cf_key) & np.isclose(ht35.h, hv)].net.to_numpy()
            px = idx_h + dx_h
            d_ax.boxplot([arr_cf], positions=[px], widths=0.26, patch_artist=True, showfliers=False,
                         boxprops={'facecolor': cf_col, 'alpha': 0.32, 'edgecolor': cf_col, 'lw': 0.7},
                         medianprops={'color': INK, 'lw': 1.0}, whiskerprops={'color': cf_col, 'lw': 0.65},
                         capprops={'color': cf_col, 'lw': 0.65}, zorder=3)
            d_ax.scatter(px + np.linspace(-0.06, 0.06, len(arr_cf)), arr_cf, s=6, color=cf_col, alpha=0.65, lw=0, zorder=4)
    d_ax.set_xticks(range(3), ['25%', '50%', '100%'])
    d_ax.set_xlim(-0.48, 2.50)
    d_ax.set_ylim(-0.022, 0.232)
    d_ax.set_xlabel('Effective heat use (%)')
    d_ax.set_ylabel('2035 net abatement\n(t CO$_2$e per MWh)')
    d_ax.legend(handles=[
        Patch(facecolor=C['teal'], alpha=0.55, edgecolor=C['teal'], label='90% gas boiler'),
        Patch(facecolor=C['blue'], alpha=0.55, edgecolor=C['blue'], label='COP 3 heat pump'),
        Patch(facecolor='#CBD5E1', alpha=0.9, label='Zero-carbon / h=0'),
    ], loc='upper left', bbox_to_anchor=(-0.02, 0.84), fontsize=7.0, handlelength=0.85, labelspacing=0.16)
    grid(d_ax)

    # --- e: 2035 H2 and external-import single credits (36-weather mean & range) ---
    specs_e = [
        ('Primary (h=0)', 0, 0.0, 'LHV_assumption', C['power']),
        ('H$_2$ LHV (10 kg)', 10, 0.0, 'LHV_assumption', C['blue']),
        ('H$_2$ LHV (12 kg)', 12, 0.0, 'LHV_assumption', C['navy']),
        ('H$_2$ HHV (10 kg)', 10, 0.0, 'HHV_assumption', C['purple']),
        ('Imports (0.1)', 0, 0.1, 'LHV_assumption', C['grey']),
        ('Imports (0.3)', 0, 0.3, 'LHV_assumption', C['grey']),
        ('Imports (0.5)', 0, 0.5, 'LHV_assumption', C['teal']),
    ]
    for i, (lab_e, h2_v, imp_v, basis_e, col_e) in enumerate(specs_e):
        sub_e = j[
            (j.year == 2035) & np.isclose(j.leakage, 0.025) & np.isclose(j.h, 0.0) &
            (j.H2_kgCO2_per_kg == h2_v) & (j.H2_energy_basis == basis_e) &
            np.isclose(j.import_tCO2_per_MWhe, imp_v) & (j.compression_accounting == 'registered_domestic')
        ].iloc[0]
        e.barh(i, sub_e.site_net_mean, height=0.58, color=col_e, alpha=0.72)
        e.errorbar(sub_e.site_net_mean, i,
                   xerr=[[sub_e.site_net_mean - sub_e.site_net_min], [sub_e.site_net_max - sub_e.site_net_mean]],
                   fmt='none', ecolor=INK, elinewidth=0.68, capsize=1.6, zorder=4)
    e.axvline(0, color=INK, lw=0.6)
    e.set_yticks(range(len(specs_e)), [x[0] for x in specs_e])
    e.invert_yaxis()
    e.set_xlabel('2035 site net abatement\n(t CO$_2$e per MWh)')
    grid(e, 'x')

    # --- f: Grouped Credit-Build-Up Bar + Gas-Grid Parity Gap Chart (NO cramped numeric heatmap!) ---
    combos = [
        (0, 0.0, 'No co-credit', '#8FA3B5'),
        (10, 0.3, 'H$_2$10 + Imp0.3', C['blue']),
        (12, 0.5, 'H$_2$12 + Imp0.5', C['navy']),
    ]
    hs_f = [0.0, 0.25, 0.5, 1.0]
    xf_pos = np.arange(len(hs_f))
    f_ax.axhline(0.0, color=INK, lw=0.55, zorder=2)
    f_ax.axhline(inj_35_mean, color=C['inject'], lw=1.1, ls='--', zorder=3)
    f_ax.axhspan(0.173, inj_35_mean, xmin=0.72, xmax=0.98, color=C['inject'], alpha=0.14, lw=0, zorder=1)
    for k_idx, (h2_v, imp_v, clab_f, col_f) in enumerate(combos):
        means_f, lo_f, hi_f = [], [], []
        for hv in hs_f:
            rf = j[
                (j.year == 2035) & np.isclose(j.leakage, 0.025) & np.isclose(j.h, hv) &
                (j.H2_kgCO2_per_kg == h2_v) & (j.H2_energy_basis == 'LHV_assumption') &
                np.isclose(j.import_tCO2_per_MWhe, imp_v) & (j.compression_accounting == 'registered_domestic')
            ].iloc[0]
            means_f.append(rf.site_net_mean)
            lo_f.append(rf.site_net_mean - rf.site_net_min)
            hi_f.append(rf.site_net_max - rf.site_net_mean)
        px_f = xf_pos + (k_idx - 1) * 0.25
        f_ax.bar(px_f, means_f, width=0.22, color=col_f, alpha=0.86, label=clab_f, zorder=3)
        f_ax.errorbar(px_f, means_f, yerr=[lo_f, hi_f], fmt='none', ecolor=INK, elinewidth=0.6, capsize=1.5, zorder=4)
    f_ax.text(-0.42, inj_35_mean + 0.005, 'Gas grid: 0.191 (leads all cells)', fontsize=7.0, color=C['inject'], weight='bold', va='bottom', ha='left')
    f_ax.annotate('+0.018\nlead', xy=(3.25, 0.182), xytext=(2.02, 0.172),
                  fontsize=7.0, color='#0E5A4D', weight='bold', ha='center', va='top', linespacing=1.02,
                  arrowprops=dict(arrowstyle='->', lw=0.65, color='#0E5A4D'))
    f_ax.set_xticks(xf_pos, ['0%', '25%', '50%', '100%'])
    f_ax.set_xlim(-0.52, 3.52)
    f_ax.set_ylim(-0.022, 0.232)
    f_ax.set_xlabel('Effective heat use (%)')
    f_ax.set_ylabel('2035 net abatement (t/MWh)')
    f_ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 0.84), fontsize=7.0, handlelength=0.85, labelspacing=0.16)
    grid(f_ax)

    # --- g: Annual Cohort Abatement Bars (g_yr) + 8-Year Cumulative Waterfall Bridge (g_tot) (NO line chart!) ---
    x_yr = np.arange(len(cum))
    g_yr.axhline(0.0, color=INK, lw=0.55, zorder=2)
    g_yr.bar(x_yr - 0.25, cum.inject_net_Mt, width=0.24, color=C['inject'], alpha=0.88, label="Gas grid from '28", zorder=3)
    g_yr.bar(x_yr, cum.switch_net_Mt, width=0.24, color=C['switch'], alpha=0.88, label='Switch at zero', zorder=3)
    g_yr.bar(x_yr + 0.25, cum.keep_net_Mt, width=0.24, color=C['power'], alpha=0.88, label='Keep power', zorder=3)
    for idx_y, n_gen in enumerate(cum.countries_still_generating):
        g_yr.text(idx_y, -0.29, str(int(n_gen)), fontsize=7.0, ha='center', va='center', color=C['switch'], weight='bold')
    g_yr.text(-0.45, -0.14, 'Countries in power:', fontsize=7.0, ha='left', va='center', color=C['switch'])
    g_yr.set_xticks(x_yr, [f"'{str(yr)[2:]}" for yr in cum.year])
    g_yr.set_xlim(-0.58, 7.58)
    g_yr.set_ylim(-0.42, 1.88)
    g_yr.set_xlabel('Year (2028–2035)')
    g_yr.set_ylabel('Annual net abatement\n(Mt CO$_2$e yr$^{-1}$)')
    g_yr.legend(loc='upper left', ncol=2, fontsize=7.0, handlelength=0.8, columnspacing=0.55, labelspacing=0.14)
    grid(g_yr)

    # Right companion sub-panel of g: 2028-2035 cumulative totals + opportunity-gain bridge
    tot_vals = [cum.cum_keep_net_Mt.iloc[-1], cum.cum_switch_net_Mt.iloc[-1], cum.cum_inject_net_Mt.iloc[-1]]
    tot_cols = [C['power'], C['switch'], C['inject']]
    x_tot = np.arange(3)
    g_tot.bar(x_tot, tot_vals, width=0.54, color=tot_cols, alpha=0.88, zorder=3)
    for i_t, v_t in enumerate(tot_vals):
        g_tot.text(i_t, v_t + 0.25, f'{v_t:.1f}', fontsize=7.0, ha='center', va='bottom', color=tot_cols[i_t], weight='bold')
    # Opportunity gain bridge bracket from Keep (1.47) & Switch (6.07) to Early Grid (9.93)
    g_tot.plot([0, 0, 2, 2], [2.8, 11.4, 11.4, 10.9], color=INK, lw=0.65)
    g_tot.text(1.0, 11.55, '+8.5 Mt', fontsize=7.0, ha='center', va='bottom', color=C['inject'], weight='bold')
    g_tot.plot([1, 1, 2], [7.3, 8.6, 8.6], color=MUTED, lw=0.55, ls='--')
    g_tot.text(1.42, 8.75, '+3.9', fontsize=7.0, ha='center', va='bottom', color=MUTED)
    g_tot.set_xticks(x_tot, ['Keep', 'Switch', 'Grid'])
    g_tot.set_ylim(0, 13.0)
    g_tot.set_xlabel('2028–35 total')
    g_tot.set_ylabel('Cumulative (Mt)')
    grid(g_tot)

    # --- h: Cumulative 2028-2035 fossil fuel displaced by period ---
    periods = [((2028, 2030), '2028–30', '#4A6075'), ((2031, 2033), '2031–33', '#8CA0B3'), ((2034, 2035), '2034–35', '#CBD5E1')]
    strats = [('keep_fossil_TWh', 'Keep power', C['power']), ('switch_fossil_TWh', 'Switch', C['switch']), ('inject_fossil_TWh', 'Gas grid', C['inject'])]
    for i, (fk, slab, scol) in enumerate(strats):
        left = 0.0
        for (y0, y1), plab, pcol in periods:
            v = cum[(cum.year >= y0) & (cum.year <= y1)][fk].sum()
            h.barh(i, v, left=left, height=0.56, color=pcol, edgecolor='white', lw=0.45, label=plab if i == 0 else None)
            left += v
        h.text(left + 0.8, i, f'{left:.1f} TWh', fontsize=7.0, va='center', ha='left', weight='bold')
    h.set_yticks(range(3), [s_[1] for s_ in strats])
    for tl, (_, _, scol) in zip(h.get_yticklabels(), strats):
        tl.set_color(scol)
        tl.set_fontweight('bold')
    h.invert_yaxis()
    h.set_xlim(0, 66)
    h.set_xlabel('Cumulative fossil fuel displaced, 2028–2035 (TWh)')
    h.legend(loc='upper right', fontsize=7.0, handlelength=0.85, labelspacing=0.16)
    grid(h, 'x')

    for axis_obj, ltr in zip([a, b_maps[0], c, d_ax, e, f_ax, g_yr, h], 'abcdefgh'):
        label(f, axis_obj, ltr)
    export(f, 'fig4_boundaries')


if __name__ == '__main__':
    jobs = {'1': fig1_trajectory, '2': fig2_mechanism, '3': fig3_spatial_scale, '4': fig4_boundaries}
    for k in sys.argv[1:] or jobs:
        jobs[k]()
