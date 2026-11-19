"""Public primary literature retrieval; independent of the frozen LP experiment.
No credentials, paywalls, author contact, or external writes. Preserve unsuccessful attempts.
"""
import sys,json,hashlib,datetime,subprocess,concurrent.futures
from pathlib import Path
from urllib.parse import quote
import requests
HERE=Path(__file__).resolve().parents[1]
OUT=HERE/'literature'; RAW=OUT/'raw';RAW.mkdir(parents=True,exist_ok=True)
UA={'User-Agent':'Research-literature-verification/1.0 (public scholarly metadata)'}
ITEMS={
 'negro2025':('10.1016/j.apenergy.2024.124687','https://iris.polito.it/retrieve/handle/11583/2993548/d4fa678b-7a86-4516-a6e2-d783c9c25948/1-s2.0-S0306261924020701-main.pdf'),
 'pasciucco2023':('10.1016/j.jclepro.2023.136762','https://arpi.unipi.it/retrieve/01d7dcaf-230d-4fee-a83f-37d491deb45c/Valorization%20of%20biogas.pdf'),
 'millinger2025':('10.1038/s41560-024-01693-6','https://www.nature.com/articles/s41560-024-01693-6.pdf'),
 'tea_lca2026':('10.3390/su18031483','https://www.mdpi.com/2071-1050/18/3/1483/pdf'),
 'iea_hydrogen2024':(None,'https://iea.blob.core.windows.net/assets/3ece5ee4-7537-4992-8ba6-b83c994c3fd4/GlobalHydrogenReview2024.pdf'),
 'ce_delft_biogas':(None,'https://cedelft.eu/wp-content/uploads/sites/2/2021/03/CE_Delft_3G84_Biogas_beyond_2020_final_report.pdf'),
}
QUERIES=[
 'Beyond energy crops and subsidised electricity A study on sustainable biogas production and utilisation in advanced energy markets',
 'Life cycle assessment of biogas infrastructure options on a regional scale',
 'biogas biomethane electricity decarbonisation marginal emissions heat recovery',
 'biogas electricity biomethane future electricity mix life cycle assessment',
]
def get(key,url):
 rec=dict(key=key,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  r=requests.get(url,headers=UA,timeout=60);rec.update(status=r.status_code,final_url=r.url,content_type=r.headers.get('content-type',''))
  ext='.pdf' if r.content.startswith(b'%PDF') else '.json' if 'json' in rec['content_type'] else '.html'
  p=RAW/(key+ext);p.write_bytes(r.content);rec.update(path=str(p.relative_to(HERE)),sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content))
  if ext=='.pdf':
   subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  rec['success']=r.ok
 except Exception as e:rec.update(success=False,error=str(e))
 return rec
def main():
 tasks=[]
 for k,(doi,url) in ITEMS.items():
  tasks.append((k,url))
  if doi:
   tasks.append((k+'_crossref','https://api.crossref.org/works/'+quote(doi,safe='')))
   tasks.append((k+'_openalex','https://api.openalex.org/works/https://doi.org/'+doi))
 for i,q in enumerate(QUERIES):tasks.append((f'query{i+1}_crossref','https://api.crossref.org/works?rows=5&query.bibliographic='+quote(q)))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows=list(pool.map(lambda x:get(*x),tasks))
 (OUT/'retrieval_manifest.json').write_text(json.dumps(dict(queries=QUERIES,records=rows),ensure_ascii=False,indent=2))
 print(json.dumps([{'key':r['key'],'status':r.get('status'),'success':r['success']} for r in rows]),flush=True)
if __name__=='__main__':main()
