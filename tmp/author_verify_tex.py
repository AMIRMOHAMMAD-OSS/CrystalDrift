from pathlib import Path
from collections import Counter
from xml.etree import ElementTree as ET
import re, sys, shutil, hashlib, json, zipfile, subprocess, os
project=Path(__file__).resolve().parent.parent
source=Path(sys.argv[1]).resolve()
build=Path(sys.argv[2]).resolve()
build.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
cfg=str(project/'tmp/crystal-audit-20261120/miktexstartup.ini')
env['MIKTEX_USERSTARTUPFILE']=cfg
env['MIKTEX_COMMONSTARTUPFILE']=cfg
def run(args,output):
 with (build/output).open('wb') as o:
  proc=subprocess.run(args,cwd=build,stdout=o,stderr=subprocess.STDOUT,env=env)
 if proc.returncode: raise RuntimeError((build/output).read_text(errors='replace')[-2400:])
def pdftool(args,output):
 with (build/output).open('wb') as o:
  proc=subprocess.run(args,cwd=build,stdout=o,stderr=subprocess.PIPE,env=env)
 if proc.returncode: raise RuntimeError(proc.stderr.decode(errors='replace')[-500:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if source.is_file():
 with zipfile.ZipFile(source) as z:
  assert z.testzip() is None
  z.extractall(build)
else:
 for f in source.iterdir():
  if f.is_file() and f.suffix in ('.tex','.pdf') and f.name not in ('main.pdf','CrytalDrift.pdf') and not f.name.startswith('CrytalDrift_'):shutil.copy2(f,build/f.name)
files={p.name:sha(p) for p in build.iterdir() if p.is_file()}
passes=[]
for i in (1,2):
 run(['pdflatex','-disable-installer','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','main.tex'],f'console-{i}.txt')
 log=(build/'main.log').read_text(errors='replace')
 passes.append({'pass':i,'log_sha256':sha(build/'main.log'),'undefined':'There were undefined references' in log})
pdftool(['pdftotext','-layout','-enc','UTF-8','main.pdf','-'],'body.txt')
pdftool(['pdftotext','-bbox','-enc','UTF-8','main.pdf','-'],'body.xml')
pdftool(['pdffonts','main.pdf'],'fonts.txt')
run(['pdftoppm','-r','50','-png','main.pdf','page'],'render.log')
pages=[]
raw=(build/'body.xml').read_text(encoding='utf8')
raw=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',raw)
for page in ET.fromstring(raw).findall('.//{*}page'):
 words=page.findall('{*}word')
 w,h=map(float,[page.attrib['width'],page.attrib['height']])
 bounds=(min(float(x.attrib['xMin']) for x in words),min(float(x.attrib['yMin']) for x in words),max(float(x.attrib['xMax']) for x in words),max(float(x.attrib['yMax']) for x in words))
 pages.append({'words':len(words),'width':w,'height':h,'bounds':bounds,'in_page':bounds[0]>=0 and bounds[1]>=0 and bounds[2]<=w and bounds[3]<=h})
warnings=re.findall(r'(?:Overfull \\[hv]box|LaTeX(?: Font)? Warning:|Package [a-zA-Z-]+ Warning:)[^\n]*',log)
data={'files':files,'source':str(source),'passes':passes,'pages':pages,'warnings':warnings,'pdf_sha256':sha(build/'main.pdf'),'no_unresolved':not re.search(r'There were undefined|Warning: (Citation|Reference).*undefined|multiply defined',log)}
(build/'build-report.json').write_text(json.dumps(data,indent=2),encoding='utf8')
run([sys.executable,str(project/'tmp/author_layout_contact.py'),str(build)],'page-inventory.txt')
print(len(pages),'pages',len(files),'source files',warnings, data['no_unresolved'],flush=True)
