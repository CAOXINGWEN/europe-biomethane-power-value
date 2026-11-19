"""Per-target-year thermal split, dispatch costs and within-country resource weights (V36 prepare36 logic).

Only the year and the input folder are parameters; formulas, efficiencies and factors are those of V36.
For 2035 the outputs are written to qa/ and compared with the frozen V36 files.
Usage: prepare37.py --years 2028 2030 2033 2035
"""
from pathlib import Path
import sys, json, argparse
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common37 import HERE, ROOT, DASH, V36, now, sha, dump, arrays, input_path

TECH = ['Nuclear', 'Lignite', 'Hard coal', 'Light oil', 'Heavy oil', 'Biofuel']
ETA = {'Nuclear': .33, 'Lignite': .4, 'Hard coal': .4, 'Light oil': .35, 'Heavy oil': .4, 'Biofuel': .4}
EF = {'Nuclear': 0., 'Lignite': .3636, 'Hard coal': .34056, 'Light oil': .26676, 'Heavy oil': .27864, 'Biofuel': 0.}
VOM = {'Nuclear': 9., 'Lignite': 3.3, 'Hard coal': 3.3, 'Light oil': 1.1, 'Heavy oil': 3.3, 'Biofuel': 25.}


def prices(year):
    p = pd.read_excel(DASH / 'Commodity Prices.xlsx', sheet_name='Commodity Prices')
    p = p[p.Year.eq(year) & p.data_version.eq('ERAA 2025 final')]
    return {str(r.Fuel).strip(): float(r.Value) for r in p.itertuples()}


def thermal(year):
    d = arrays(input_path(year, 13)); zones = d['zones']; Z = len(zones); T = 8760
    raw = pd.read_csv(DASH / 'GenerationCapacities.csv'); raw = raw[raw.data_version.eq('ERAA 2025 final') & raw['Target year'].eq(year)]
    cap = np.zeros((T, Z, len(TECH))); gas = np.zeros((T, Z))
    for k, (z, tech) in enumerate(zip(d['audit_thermal_zone'], d['audit_thermal_technology'])):
        if tech == 'Hydrogen': continue
        if tech == 'Natural gas': gas[:, z] += d['audit_thermal_available'][:, k]
        else: cap[:, z, TECH.index(str(tech))] += d['audit_thermal_available'][:, k]
    mappings = []
    for z, zone in enumerate(zones):
        extra = float(raw[raw.Market_Node.eq(zone) & raw.Technology.eq('Natural gas') & raw.Operational_Status.str.contains('Inelastic', na=False)].Value.sum()) * .92
        gas[:, z] += extra
        residual = d['other_thermal_cap'][:, z] - cap[:, z, :].sum(axis=1)
        if abs(residual).max() > 1e-7:
            if abs(residual - residual[0]).max() > 1e-7: raise ValueError('Time-varying unexplained other capacity ' + zone)
            if residual[0] > 1e-7: raise ValueError('Unexpected positive other capacity ' + zone)
            cap[:, z, TECH.index('Biofuel')] += residual
        mappings.append(dict(zone=str(zone), inelastic_gas_addition_MW=extra, legacy_biomass_pool_subtraction_MW=float(-residual.mean())))
    gaserror = float(abs(gas - d['gas_cap']).max()); othererror = float(abs(cap.sum(axis=2) - d['other_thermal_cap']).max())
    if max(gaserror, othererror) > 1e-7 or cap.min() < -1e-7: raise ValueError('Thermal reconciliation failed')
    cap = np.maximum(cap, 0)
    price = prices(year); carbon = price['CO2 price']
    columns = []; cost = []; tz = []; tn = []; eff = []; emissions = []; ledger = []
    for z, zone in enumerate(zones):
        country = str(d['countries'][z]); group = 1 if country in ['BG', 'CZ'] else 3 if country in ['SI', 'RO'] else 4 if country == 'GR' else 2
        lignite = next(v for k, v in price.items() if k.startswith('Lignite G' + str(group)))
        for k, name in enumerate(TECH):
            if cap[:, z, k].max() < 1e-8: continue
            fuel = lignite if name == 'Lignite' else price.get(name, 0.)
            mc = fuel * 3.6 / ETA[name] + EF[name] / ETA[name] * carbon + VOM[name]
            columns.append(cap[:, z, k]); cost.append(mc); tz.append(z); tn.append(name); eff.append(ETA[name]); emissions.append(EF[name])
            ledger.append(dict(year=year, zone=str(zone), technology=name, mean_available_MW=float(cap[:, z, k].mean()), efficiency=ETA[name],
                               combustion_tCO2_per_MWh_fuel=EF[name], fuel_price_EUR2024_per_MWh_fuel=fuel * 3.6, carbon_EUR2024_per_t=carbon,
                               variable_OM_EUR2024_per_MWh=VOM[name], dispatch_cost_EUR2024_per_MWh=mc))
    data = dict(zones=zones, thermal_cap=np.array(columns).T, thermal_zone=np.array(tz, dtype=int), thermal_technology=np.array(tn, dtype='U20'),
                thermal_efficiency=np.array(eff), thermal_combustion_ef=np.array(emissions), thermal_cost=np.array(cost),
                gas_efficiency=np.full(Z, .50), gas_cost=np.full(Z, (price['Gas'] * 3.6 + .20196 * carbon) / .50 + 1.6))
    gross = np.zeros(Z)
    for w in range(1, 37):
        with np.load(input_path(year, w), allow_pickle=False) as a:
            gross += a['gross_demand'].mean(axis=0) / 36
            for k in ['gas_cap', 'other_thermal_cap', 'audit_thermal_available']:
                if not np.array_equal(a[k], d[k]): raise ValueError(f'Weather-dependent thermal input {year}/{w}/{k}')
    weights = np.zeros(Z)
    for country in np.unique(d['countries']):
        ix = d['countries'] == country; weights[ix] = gross[ix] / gross[ix].sum()
    u = pd.read_csv(V36 / 'inputs/uk_resource_v36.csv'); u = u[u['case'].eq('primary_mean_2021_2022') & u.scenario.eq('Central')].iloc[0]
    weights[zones == 'UKNI'] = u.UKNI_MW / u.MW_fuel_mean; weights[zones == 'UK00'] = u.UK00_MW / u.MW_fuel_mean
    rw = dict(zones=zones, countries=d['countries'], weights=weights, mean_gross_demand_MW=gross)
    recon = dict(passed=True, year=year, created_utc=now(), gas_max_absolute_difference_MW=gaserror, other_max_absolute_difference_MW=othererror,
                 mapping_corrections=mappings, prices=price, weather_invariant_thermal_inputs=True)
    return data, rw, pd.DataFrame(ledger), recon


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--years', nargs='+', type=int, required=True); a = ap.parse_args()
    out = HERE / 'inputs/system'
    for y in a.years:
        data, rw, ledger, recon = thermal(y)
        if y == 2035:
            ref = arrays(V36 / 'inputs/thermal_split_2035.npz'); rref = arrays(V36 / 'inputs/resource_zone_weights.npz')
            diff = {k: float(np.abs(np.asarray(data[k], float) - np.asarray(ref[k], float)).max()) for k in data if k != 'zones' and data[k].dtype.kind != 'U'}
            diff['thermal_technology_equal'] = bool(np.array_equal(data['thermal_technology'], ref['thermal_technology']))
            diff['weights'] = float(np.abs(rw['weights'] - rref['weights']).max())
            ok = diff.pop('thermal_technology_equal') and max(diff.values()) <= 1e-9
            dump(HERE / 'qa/reproduce_thermal_split_2035.json', dict(utc=now(), passed=ok, max_abs_difference=diff))
            print('2035 thermal reproduction', ok, max(diff.values()), flush=True); continue
        np.savez_compressed(out / f'thermal_split_{y}.npz', **data)
        np.savez_compressed(out / f'resource_zone_weights_{y}.npz', **rw)
        ledger.to_csv(out / f'thermal_parameters_{y}.csv', index=False)
        recon['outputs_sha256'] = {f'thermal_split_{y}.npz': sha(out / f'thermal_split_{y}.npz'), f'resource_zone_weights_{y}.npz': sha(out / f'resource_zone_weights_{y}.npz')}
        dump(HERE / f'qa/thermal_split_reconciliation_{y}.json', recon)
        cap = data['thermal_cap'].mean(axis=0); tech = data['thermal_technology']
        print(y, {t: round(float(cap[tech == t].sum()) / 1000, 2) for t in np.unique(tech)}, 'GW mean available', flush=True)


if __name__ == '__main__': main()
