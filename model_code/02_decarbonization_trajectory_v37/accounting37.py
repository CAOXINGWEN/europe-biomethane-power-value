"""Paired dispatch accounting (use case minus same-year, same-weather counterfactual), V36 factors.

pair_metrics(case_arrays, r0_arrays, hydro_zone) returns system totals: fossil fuel displaced by fuel, avoided
emissions (GWP100, GWP20), fossil share s_f, displaced fossil intensity m, EENS change and the complete energy
balance of the added site electricity (closure residual reported). No interpretation is produced here.
"""
from pathlib import Path
import sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common37 import HERE, V36, arrays

P36 = json.loads((V36 / 'inputs/nonpower_parameters.json').read_text())
FOSSIL = ['gas', 'Lignite', 'Hard coal', 'Light oil', 'Heavy oil']
COMB = {'gas': P36['gas_combustion_t_per_MWh'], 'Lignite': .3636, 'Hard coal': .34056, 'Light oil': .26676, 'Heavy oil': .27864}
UP_FIXED = {'Lignite': .010, 'Hard coal': .038, 'Light oil': .07884, 'Heavy oil': .02052}     # V36 postprocess36 UPSTREAM


def gas_upstream(h):
    v = P36['gas_upstream_inventory_g_per_MJ']; g = P36[h]
    return (v['CO2'] + v['CH4'] * g['fossil_CH4'] + v['N2O'] * g['N2O']) * .0036


EF = {h: {k: COMB[k] + (gas_upstream(h) if k == 'gas' else UP_FIXED[k]) for k in FOSSIL} for h in ['GWP100', 'GWP20']}
THERMAL = ['Nuclear', 'Biofuel', 'Lignite', 'Hard coal', 'Light oil', 'Heavy oil']


def generation(a):
    """System annual generation (MWh) by technology and annual fuel (MWh) by fossil fuel."""
    gen = {'gas': float(a['gas'].sum())}; fuel = {'gas': float((a['gas'] / a['gas_efficiency'][None, :]).sum())}
    tech = np.asarray(a['thermal_technology']); eff = np.asarray(a['thermal_efficiency']); td = a['thermal_dispatch']
    for t in THERMAL:
        ix = tech == t; gen[t] = float(td[:, ix].sum())
        if t in FOSSIL: fuel[t] = float((td[:, ix] / eff[ix][None, :]).sum())
    return gen, fuel


def pair_metrics(u, r):
    gu, fu = generation(u); g0, f0 = generation(r)
    B = float(u['resource_q_hourly'].sum())
    dF = {k: f0[k] - fu[k] for k in FOSSIL}
    d = dict(biomethane_MWh=B, **{f'delta_fuel_{k}_MWh': v for k, v in dF.items()})
    d['fossil_fuel_displaced_MWh'] = sum(dF.values())
    for h in ['GWP100', 'GWP20']: d[f'avoided_{h}_t'] = sum(dF[k] * EF[h][k] for k in FOSSIL)
    d['avoided_combustion_t'] = sum(dF[k] * COMB[k] for k in FOSSIL)
    # Energy balance of the added supply: positive entries are where the added electricity went.
    went = {f'reduced_{k}': g0[k] - gu[k] for k in ['gas'] + THERMAL}
    for k in ['hydrogen', 'external_import', 'hydro_generation', 'existing_battery_discharge', 'dsr', 'unserved']:
        went[f'reduced_{k}'] = float(r[k].sum() - u[k].sum())
    for k in ['curtailment', 'hydro_pumping', 'existing_battery_charge']:
        went[f'increased_{k}'] = float(u[k].sum() - r[k].sum())
    went['increased_idsr_consumption'] = float(u['idsr_consumption'].sum() - r['idsr_consumption'].sum())
    if 'flex_consumption' in u and 'flex_consumption' in r:          # amendment_post_01 (S-flex)
        typ = np.asarray(u['flex_type']); eff = np.asarray(u['flex_eff']); dc = u['flex_consumption'].sum(axis=0) - r['flex_consumption'].sum(axis=0)
        went['increased_electrolyser_consumption'] = float(dc[typ == 'Electrolyser'].sum()); went['increased_power_to_heat_consumption'] = float(dc[typ == 'Power to heat'].sum())
        d_extra = dict(extra_hydrogen_MWh=float((dc * eff)[typ == 'Electrolyser'].sum()), flex_consumption_R0_MWh=float(r['flex_consumption'].sum()), flex_consumption_U_MWh=float(u['flex_consumption'].sum()))
    else: d_extra = {}
    d.update({f'balance_{k}_MWh': v for k, v in went.items()})
    d['fossil_generation_reduction_MWh'] = sum(g0[k] - gu[k] for k in FOSSIL)
    d['delta_EENS_MWh'] = float(r['unserved'].sum() - u['unserved'].sum())
    d.update(d_extra)
    return d


def finish_u1(d, site_eff=.4):
    """Per-MWh quantities for a U1 pair; site generation = 0.40 x biomethane."""
    B = d['biomethane_MWh']; site = site_eff * B; out = dict(site_generation_MWh=site)
    out['D_fossil_per_MWh'] = d['fossil_fuel_displaced_MWh'] / B
    for h in ['GWP100', 'GWP20']: out[f'avoided_{h}_t_per_MWh'] = d[f'avoided_{h}_t'] / B
    out['s_f'] = d['fossil_generation_reduction_MWh'] / site
    out['m_t_per_MWh_e'] = d['avoided_GWP100_t'] / d['fossil_generation_reduction_MWh'] if abs(d['fossil_generation_reduction_MWh']) > 1e-6 else np.nan
    out['MEF_t_per_MWh_e'] = d['avoided_GWP100_t'] / site
    bal = sum(v for k, v in d.items() if k.startswith('balance_'))
    out['balance_residual_MWh'] = site - bal; out['balance_residual_relative'] = (site - bal) / site
    return out


if __name__ == '__main__':
    # Self-test against V36 (2035, Central): R1_Central_WSnn minus R0_WSnn.
    led = pd.read_csv(V36 / 'report/country_use_by_weather.csv'); led = led[(led.name.str.match(r'R1_Central_WS\d\d$'))]
    rows = []
    for w in [int(x) for x in sys.argv[1:]] or [1, 13, 20]:
        u = arrays(V36 / f'results/R1_Central_WS{w:02d}.npz'); r = arrays(V36 / f'results/R0_WS{w:02d}.npz')
        m = pair_metrics(u, r); f = finish_u1(m); ref = led[led.weather.eq(w)]
        rows.append(dict(weather=w, D=f['D_fossil_per_MWh'], D_V36=ref.dispatch_fossil_fuel_displacement_MWh.sum() / ref.biomethane_MWh.sum(),
                         avoided=m['avoided_GWP100_t'], avoided_V36=ref.avoided_fossil_emissions_GWP100_t.sum(), s_f=f['s_f'], m=f['m_t_per_MWh_e'],
                         residual_rel=f['balance_residual_relative']))
    print(pd.DataFrame(rows).to_string())
