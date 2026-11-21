from pathlib import Path
from PIL import Image, ImageDraw
import sys,re
sys.stdout.reconfigure(encoding='utf8', errors='replace')
p = Path(sys.argv[1]); fs=sorted(p.glob('page-*.png'),key=lambda f:int(f.stem.split('-')[-1]))
pages=(p/'body.txt').read_text(encoding='utf8').split('\f')
if not pages[-1].strip():pages.pop()
for start in range(0,len(fs),8):
 ims=Image.new('RGB',(4*246,2*330),'#e5e5e5');d=ImageDraw.Draw(ims)
 for j,path in enumerate(fs[start:start+8]):
  with Image.open(path) as im:
   im.thumbnail((237,305));x=4+(j%4)*246;y=4+(j//4)*330;ims.paste(im,(x,y))
   d.text((x+2,y+306),f'Page {start+j+1}',fill='black')
 ims.save(p/f'overview-{start+1:02}.jpg',quality=87)
for i,page in enumerate(pages):
 lines=[s.strip() for s in page.splitlines() if s.strip()]; chars=sum(len(s) for s in lines)
 figs=[s[:70] for s in lines if re.match(r'(Figure|Table)\s+\d+[:.]',s)]
 print(i+1,chars,repr(lines[:1]),'; '.join(figs))
print('sheets',len(fs))

