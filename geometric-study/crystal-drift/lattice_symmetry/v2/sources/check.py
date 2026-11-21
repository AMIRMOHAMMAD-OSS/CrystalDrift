"""Record primary-source metadata, method locators, and linked code for the source guide."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

HERE=Path(__file__).resolve().parent
GUIDE=json.loads((HERE/"guide.json").read_text(encoding="utf8"))
HEADERS={"User-Agent":"CrytalDrift scholarly method audit/1.0"}


def get(url):
    r=requests.get(url,headers=HEADERS,timeout=(8,25))
    r.raise_for_status()
    return r.content


def html(body):
    return BeautifulSoup(body,"html.parser")


def values(doc,name):
    return [t.get("content","").strip() for t in doc.find_all("meta",attrs={"name":name})]


def check(item):
    key,ident,locator,term,note=item
    arxiv=not ident.startswith(("10.","http"))
    base=("https://arxiv.org/abs/"+ident if arxiv else "https://doi.org/"+ident if ident.startswith("10.") else ident)
    record=dict(key=key,url=base,locator=locator,term=term,interpretation=note)
    try:
        if ident.startswith("10."):
            obj=json.loads(get("https://api.crossref.org/works/"+ident))["message"]
            record.update(title=obj["title"][0],first_author=obj["author"][0]["family"],
                          year=obj["published"]["date-parts"][0][0])
        try:
            body=get(base)
            page=html(body)
            title=values(page,"citation_title") or values(page,"DC.Title")
            if title:record["title"]=title[0]
            if "title" not in record:record["title"]=page.title.get_text(" ",strip=True)
            authors=values(page,"citation_author")
            if authors:record["first_author"]=authors[0]
            date=values(page,"citation_date") or values(page,"citation_publication_date")
            if date:record["year"]=int(re.search(r"\d{4}",date[0])[0])
            record["landing_sha256"]=hashlib.sha256(body).hexdigest()
        except Exception as exc:
            if "title" not in record:raise
            record["landing_status"]=str(exc)
            page=html(b"")
        content=page
        if arxiv:
            try:
                b=get("https://arxiv.org/html/"+ident)
                candidate=html(b)
                if candidate.select_one("article") or candidate.select_one(".ltx_document"):
                    content=candidate
                    record["read_fulltext_sha256"]=hashlib.sha256(b).hexdigest()
                    record["fulltext_url"]="https://arxiv.org/html/"+ident
            except requests.RequestException:pass
        record["linked_repositories"]=sorted({urljoin(base,a["href"]).split("#")[0].rstrip(".,)") for a in content.find_all("a",href=True) if re.match(r"https?://github\.com/[^/]+/[^/?#]+",a["href"],re.I)})[:10]
        hits=[]
        for tag in content.find_all(["p","div","section","td"]):
            if tag.name in ("div","section") and not ("ltx_para" in tag.get("class",[]) or tag.get("id")=="Abs1"):continue
            s=" ".join(tag.stripped_strings)
            if term.lower() in s.lower() and len(s)<5000:
                parent=tag if tag.get("id") else tag.find_parent(id=True)
                hits.append(parent.get("id") if parent else "main text")
        record["matching_sections"]=list(dict.fromkeys(hits))[:12]
        if not record["matching_sections"] and arxiv:
            try:
                pdf=get("https://arxiv.org/pdf/"+ident)
                exe=shutil.which("pdftotext")
                text=subprocess.run([exe,"-layout","-","-"],input=pdf,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=45).stdout.decode("utf8","replace")
                # PDF text extraction can insert hyphens and newlines mid-word.
                record["matching_pdf_pages"]=[i for i,s in enumerate(text.split("\f"),1) if term.lower() in s.lower()][:12]
                record["pdf_sha256"]=hashlib.sha256(pdf).hexdigest()
            except Exception as exc:record["pdf_status"]=str(exc)
        record["metadata_verified"]=bool(record.get("title") and record.get("first_author"))
    except Exception as exc:record["error"]=str(exc)
    print(key,str(record.get("title",record.get("error")))[:86],flush=True)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retry",action="store_true")
    args=p.parse_args()
    out=HERE/"primary-records.json"
    old={r["key"]:r for r in json.loads(out.read_text(encoding="utf8"))} if args.retry and out.exists() else {}
    jobs=[r for r in GUIDE if not old.get(r[0],{}).get("metadata_verified")]
    with ThreadPoolExecutor(4) as pool:
        records={**old,**{r["key"]:r for r in pool.map(check,jobs)}}
    out.write_text(json.dumps([records[r[0]] for r in GUIDE],indent=2,ensure_ascii=False)+"\n",encoding="utf8")
