"""Render a submission PDF to page images and contact sheets for visual inspection."""
import argparse
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf",type=Path)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--renderer",default=shutil.which("pdftoppm"))
    args=parser.parse_args()
    args.directory.mkdir(exist_ok=True,parents=True)
    subprocess.run([args.renderer,"-scale-to","1080","-png",str(args.pdf),str(args.directory/"page")],check=True)
    pages=sorted(args.directory.glob("page-*.png"))
    for start in range(0,len(pages),9):
        sheet=Image.new("RGB",(870,1200),"#dbe1e5")
        draw=ImageDraw.Draw(sheet)
        for j,file in enumerate(pages[start:start+9]):
            with Image.open(file) as original:
                page=original.convert("RGB")
                page.thumbnail((280,370))
                x=(j%3)*290+(290-page.width)//2
                y=(j//3)*400+9
                sheet.paste(page,(x,y))
                draw.text((x, y+page.height+2),file.stem,fill="#1b3447")
        sheet.save(args.directory/f"contact-{start//9+1:02}.png")
    print(len(pages),"pages rendered")

if __name__=="__main__":
    main()
