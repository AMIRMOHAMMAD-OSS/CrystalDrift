"""Build the current manuscript from its actual arXiv source and publish after QA."""
from pathlib import Path
from datetime import datetime
import hashlib, json, os, re, shutil, subprocess, sys, tempfile, zipfile

project=Path(__file__).resolve().parent.parent
manuscript=project/'CrytalDrift_Manuscript'
zip_target=manuscript/'CrytalDrift_arXiv_Source.zip'
pdf_target=manuscript/'CrytalDrift.pdf'
release=project/'geometric-study/crystal-drift/lattice_symmetry/v3/release/CrytalDrift-arxiv-source.zip'
stage=Path(tempfile.mkdtemp(prefix='crytaldrift-theorems-',dir=project/'tmp'))
print(f'Build workspace: {stage}',flush=True)

files={}
def load(name):
    if name in files:return
    path=manuscript/name
    assert path.resolve().is_relative_to(manuscript.resolve())
    files[name]=data=path.read_bytes()
    if path.suffix != '.tex':return
    src=re.sub(r'(?<!\\)%[^\n]*','',data.decode('utf8'))
    for child in re.findall(r'\\(?:input|include)\{([^}]+)\}',src):
        load(child if Path(child).suffix else child+'.tex')
    for child in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',src):
        load(child)
load('main.tex')
figs={f for f in files if f.endswith('.pdf')}
with zipfile.ZipFile(release) as z:
    assert figs=={f for f in z.namelist() if f.endswith('.pdf')}, figures if False else figs
    assert len(figs)==12 and all(files[f]==z.read(f) for f in figs)
newzip=stage/zip_target.name
with zipfile.ZipFile(newzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for mem,data in sorted(files.items()):
        zi=zipfile.ZipInfo(mem,(2026,11,21,0,0,0)); zi.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(zi,data)
with zipfile.ZipFile(newzip) as z:
    assert z.testzip() is None and len(z.namelist())==len(files)
build=stage/'extracted'
subprocess.run([sys.executable,project/'tmp/author_verify_tex.py',newzip,build],check=True)
report=json.loads((build/'build-report.json').read_text())
assert report['no_unresolved'] and not report['warnings'] and all(p['in_page'] for p in report['pages']), report['warnings']
assert all((manuscript/name).read_bytes()==files[name] for name in files),'Author source changed during compilation'
pages_file=(build/'body.txt').read_text(encoding='utf8')
pages=pages_file.split('\f')
for i,t in enumerate(pages,1):
    for line in t.splitlines():
        if re.match(r'^\s*(Theorem|Proposition|Lemma|Corollary)\s+[A-Za-z]?\.?[0-9]',line): print(f'Page {i}: {line.strip()[:145]}',flush=True)
notes=[]
for name in (zip_target.name,pdf_target.name):
    orig=manuscript/name
    if orig.exists():shutil.copyfile(orig,stage/('prior-'+name))
tmp_pdf=stage/'final.pdf'
shutil.copyfile(build/'main.pdf',tmp_pdf)
try:
    os.replace(newzip,zip_target)
    os.replace(tmp_pdf,pdf_target)
except Exception:
    for path in (zip_target,pdf_target):
        old=stage/('prior-'+path.name)
        if old.exists():shutil.copyfile(old,path)
    raise
for target in (zip_target,pdf_target):
    print(target.name, target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest())
print('QA path:',build, 'figures:',len(figs),'pages:',len(report['pages']),'source members:',len(files),flush=True)
