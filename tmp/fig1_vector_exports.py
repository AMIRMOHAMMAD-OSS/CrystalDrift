from pathlib import Path
import datetime, html, math, os, re, shutil, subprocess, sys, tempfile, xml.etree.ElementTree as ET
from PIL import Image,ImageChops
project=Path(__file__).resolve().parent.parent
folder=project/'CrystalDrift_Figure1_Design'
source=folder/'Figure1_Periodic_Drift.svg'
stage=Path(tempfile.mkdtemp(prefix='fig1-exports-',dir=project/'tmp'))
print(stage,flush=True)
data=source.read_text(encoding='utf-8-sig')
root=ET.fromstring(data)
xmin,ymin,w,h=map(float,root.get('viewBox').replace(',',' ').split())
assert w>0 and h>0
csswidth=960.0
cssheight=csswidth*h/w
page_css=f'@page{{size:{csswidth:.5f}px {cssheight:.5f}px;margin:0}}html,body{{padding:0;margin:0;background:white;width:{csswidth:.5f}px;height:{cssheight:.5f}px;overflow:hidden;line-height:0;}}body>svg{{display:block;width:100%;height:100%;}}'
wrapper=stage/'plate.html'
svg=data[data.index('<svg'):]
wrapper.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Periodic drifting field</title><style>'+page_css+'</style></head><body>'+svg+'</body></html>',encoding='utf8')
edge=r'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
dest=stage/'plate.pdf'
args=[edge,'--headless','--disable-gpu','--no-first-run','--no-pdf-header-footer','--user-data-dir='+str(stage/'edge-profile'),'--print-to-pdf='+str(dest),wrapper.as_uri()]
with (stage/'browser.log').open('wb') as f:
 subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=100,creationflags=subprocess.CREATE_NO_WINDOW)
assert dest.stat().st_size>1000
env=os.environ.copy()
cfg=str(project/'tmp/crystal-audit-20261120/miktexstartup.ini');env['MIKTEX_USERSTARTUPFILE']=cfg;env['MIKTEX_COMMONSTARTUPFILE']=cfg
def readcmd(cmd): return subprocess.run(cmd,capture_output=True,env=env,check=True).stdout.decode('utf8',errors='replace')
info=readcmd(['pdfinfo',dest])
assert re.search(r'Pages:\s*1\b',info),info
size=re.search(r'Page size:\s*([\d.]+) x ([\d.]+) pts',info);assert size,info
pdfw,pdfh=map(float,size.groups())
assert abs((pdfw/pdfh)/(w/h)-1)<.012
fonts=readcmd(['pdffonts',dest]);text=readcmd(['pdftotext','-layout','-enc','UTF-8',dest,'-'])
(stage/'labels.txt').write_text(text,encoding='utf8')
(stage/'fonts.txt').write_text(fonts,encoding='utf8')
subprocess.run(['pdftoppm','-r','360','-png','-singlefile',str(dest),str(stage/'plate')],env=env,check=True)
png=stage/'plate.png'
with Image.open(png) as im:
 im.verify()
with Image.open(png) as im:
 rgb=im.convert('RGB');white=Image.new('RGB',rgb.size,'white');bbox=ImageChops.difference(rgb,white).getbbox();assert bbox and rgb.width>=3500
 im.thumbnail((1450,1100));im.save(stage/'qa.png')
 assert bbox[2]-bbox[0]>.65*rgb.width and bbox[3]-bbox[1]>.60*rgb.height
 print('Canvas:',rgb.size,'ink bounds:',bbox)
print('PDF:',pdfw,pdfh,'points. Embedded fonts:',fonts.count(' yes '),'text chars:',len(text.strip()),flush=True)
assert len(text.strip())>15 and ('yes' in fonts),fonts
assert source.read_text(encoding='utf-8-sig')==data,'SVG changed during render'
shutil.copyfile(dest,source.with_suffix('.pdf'));shutil.copyfile(png,source.with_suffix('.png'))
for f in (source,source.with_suffix('.pdf'),source.with_suffix('.png')): print(f.name,f.stat().st_size,flush=True)
