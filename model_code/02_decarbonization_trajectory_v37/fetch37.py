"""Download registered public statistics (Eurostat JSON-stat API) into inputs/raw/eurostat and log them.

Downloads are data only: saved verbatim, hashed, and later parsed with json (no code is executed).
"""
from pathlib import Path
import sys, json, hashlib
from urllib.parse import urlencode
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import requests
from common37 import HERE, now, dump

API = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
GEO = ['AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'FR', 'DE', 'EL', 'IT', 'PT', 'RO', 'SI', 'ES', 'EU27_2020']
GEO_ALL = GEO[:-1] + ['CY', 'EE', 'FI', 'HU', 'IE', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'SK', 'SE', 'EU27_2020']
REQUESTS = {
    'demo_pjan_2021_2023': ('demo_pjan', dict(sex='T', age='TOTAL', time=['2021', '2022', '2023', '2024'], geo=GEO + ['UK'])),
    'proj_23np_bsl': ('proj_23np', dict(projection='BSL', sex='T', age='TOTAL', unit='PER', time=['2028', '2030', '2033', '2035'], geo=GEO)),
    'env_wasmun_cd_kg_cap': ('env_wasmun', dict(wst_oper='RCY_C_D', unit='KG_HAB', time=[str(y) for y in range(2018, 2025)], geo=GEO_ALL)),
    'env_wasmun_gen_kg_cap': ('env_wasmun', dict(wst_oper='GEN', unit='KG_HAB', time=[str(y) for y in range(2018, 2025)], geo=GEO_ALL)),
    'nrg_bal_peh_biogas': ('nrg_bal_peh', dict(siec='R5300', nrg_bal='GEP', unit='GWH', time=[str(y) for y in range(2019, 2025)], geo=GEO)),
}


def get(dataset, params):
    q = [('format', 'JSON'), ('lang', 'EN')]
    for k, v in params.items():
        for x in (v if isinstance(v, list) else [v]): q.append((k, x))
    url = API + dataset + '?' + urlencode(q)
    r = requests.get(url, timeout=120, proxies={'http': None, 'https': None}); r.raise_for_status()
    return url, r.content


def main():
    out = HERE / 'inputs/raw/eurostat'; out.mkdir(parents=True, exist_ok=True)
    reg_path = HERE / 'inputs/source_registry.json'
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for key, (ds, params) in REQUESTS.items():
        url, body = get(ds, params); p = out / f'{key}.json'; p.write_bytes(body)
        j = json.loads(body); print(key, len(j.get('value', {})), 'values', j.get('updated', ''), flush=True)
        reg[str(p.relative_to(HERE))] = dict(url=url, downloaded_utc=now(), sha256=hashlib.sha256(body).hexdigest(), dataset_updated=j.get('updated'))
    dump(reg_path, reg)


if __name__ == '__main__': main()
