"""Download public data as inert bytes into V36 only; never execute downloads."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urlencode
import json,hashlib,requests,sys
HERE=Path(__file__).resolve().parents[1]
GEO=['AT','BE','BG','HR','CZ','DK','FR','DE','EL','IT','PT','RO','SI','ES','UK']
API='https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'
PARAMS=[('lang','EN'),('freq','A'),('sinceTimePeriod','2014'),('untilTimePeriod','2023')]+[('geo',g) for g in GEO]
SOURCES=[
 ('eurostat/nrg_bal_c.json',API+'nrg_bal_c?'+urlencode(PARAMS+[('siec','G3000'),('unit','GWH')]),'json'),
 ('eurostat/nrg_ind_id.json',API+'nrg_ind_id?'+urlencode(PARAMS+[('siec','G3000')]),'json'),
 ('ipcc/IPCC2006_stationary_combustion.pdf','https://www.ipcc-nggip.iges.or.jp/public/2006gl/pdf/2_Volume2/V2_2_Ch2_Stationary_Combustion.pdf','pdf'),
 ('jec/JEC_TTW_HDV_v5.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC117564/jec_ttw_v5_hdv_117564_final.pdf','pdf'),
 ('jec/JEC_WTW_v5.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC121213/jec_wtw_v5_121213_final.pdf','pdf'),
 ('jrc/JRC142012_2026_emission_values.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC142012/JRC142012_01.pdf','pdf'),
 ('dea/renewable_fuels_page.html','https://ens.dk/en/analyses-and-statistics/technology-data-renewable-fuels','html'),
 ('policy/ETS2.html','https://climate.ec.europa.eu/areas-action/carbon-markets/ets2-buildings-road-transport-and-additional-sectors_en','html'),
 ('dukes/gas_page.html','https://www.gov.uk/government/statistics/natural-gas-chapter-4-digest-of-united-kingdom-energy-statistics-dukes','html'),
]


def fetch(item):
    filename,url,kind=item;p=HERE/'inputs/raw'/filename;p.parent.mkdir(parents=True,exist_ok=True)
    rec=dict(file=str(p.relative_to(HERE)),url=url,retrieved_utc=datetime.now(timezone.utc).isoformat(),attempts=[])
    if p.exists():
        rec.update(success=True,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),existing=True);return rec
    for direct in [False,True]:
        try:
            session=requests.Session();session.trust_env=not direct
            response=session.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=(12,60))
            rec['attempts'].append(dict(direct=direct,status=response.status_code,bytes=len(response.content)))
            response.raise_for_status();data=response.content
            if kind=='pdf' and not data.startswith(b'%PDF'):raise ValueError('Non-PDF response')
            if kind=='xlsx' and not data.startswith(b'PK'):raise ValueError('Non-XLSX response')
            if kind=='json':
                value=response.json()
                if 'error' in value:raise ValueError(str(value['error']))
            p.write_bytes(data);rec.update(success=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest());return rec
        except Exception as exc:rec['attempts'].append(dict(direct=direct,error=str(exc)))
    rec['success']=False;return rec


def main():
    out=HERE/'inputs/source_registry.json';old=json.loads(out.read_text()) if out.exists() else []
    todo=[x for x in SOURCES if not (HERE/'inputs/raw'/x[0]).exists()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,x) for x in todo]):
            row=future.result();old=[x for x in old if x.get('file')!=row['file']]+[row]
            out.write_text(json.dumps(old,ensure_ascii=False,indent=1)+'\n')
            print(json.dumps({k:row.get(k) for k in ['file','success','bytes']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
