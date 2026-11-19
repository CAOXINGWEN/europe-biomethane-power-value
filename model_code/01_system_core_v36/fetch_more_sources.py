"""Follow verified publisher links; retain every transfer attempt in the registry."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import urlencode,urljoin
import sys,json,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fetch_sources import fetch,HERE,API


def main():
    reg=HERE/'inputs/source_registry.json';old=json.loads(reg.read_text())
    sources=[
        ('dukes/DUKES_4.1_2026.xlsx','https://assets.publishing.service.gov.uk/media/6a69f4332c4b5ba9d1242a68/DUKES_4.1.xlsx','xlsx'),
        ('dukes/DUKES_4.1.1_2026.xlsx','https://assets.publishing.service.gov.uk/media/6a69f47e701b2d693e5b9420/DUKES_4.1.1.xlsx','xlsx'),
        ('dea/renewable_fuels_2026.xlsx','https://ens.dk/media/6444/download','xlsx'),
        ('dea/renewable_fuels_2026.pdf','https://ens.dk/media/7823/download','pdf'),
        ('jec/jec_v5_page.html','https://joint-research-centre.ec.europa.eu/welcome-jec-website/jec-publications/jec-version-5-2020_en','html'),
        ('eurostat/hicp_2020_2024.json',API+'prc_hicp_aind?'+urlencode([('lang','EN'),('unit','INX_A_AVG'),('coicop','CP00'),('geo','EA20'),('time','2020'),('time','2024')]),'json'),
        ('jec/JEC_TTW_HDV_v5.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC117564/jec_ttw_v5_hdv_117564_final.pdf?download=1','pdf'),
        ('jrc/JRC142012_2026_emission_values.pdf','https://publications.jrc.ec.europa.eu/repository/bitstream/JRC142012/JRC142012_01.pdf?download=1','pdf'),
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(fetch,x) for x in sources]):
            row=f.result();old.append(row);reg.write_text(json.dumps(old,ensure_ascii=False,indent=1)+'\n');print(json.dumps({k:row.get(k) for k in ['file','success','bytes']}),flush=True)
    p=HERE/'inputs/raw/jec/jec_v5_page.html'
    if p.exists():
        links=re.findall(r'href=["\']([^"\']+)["\']',p.read_text())
        for u in links:
            if any(x in u.lower() for x in ['hdv','wtt','well-to-tank']):print('JEC_LINK',urljoin('https://joint-research-centre.ec.europa.eu',u),flush=True)


if __name__=='__main__':main()
