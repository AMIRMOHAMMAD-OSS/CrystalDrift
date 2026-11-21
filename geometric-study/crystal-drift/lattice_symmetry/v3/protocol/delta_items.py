"""List immutable new editions and pending independent readbacks, avoiding old volume uploads."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT=Path(r"\\wsl.localhost\Ubuntu\home\amir\crytaldrift-v3-evidence\recovered-ephemeral\crytaldrift-v3-20261121")
PUBLIC=("FINAL-BACKUP.json","FINAL-FILES.ndjson","FINAL-README.md","restore-final.py","CrytalDrift.pdf","CrytalDrift-arxiv-source.zip","MORNING.pdf")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("kind",choices=("upload","verify"))
    parser.add_argument("--slot",type=int,default=0)
    parser.add_argument("--slots",type=int,default=1)
    parser.add_argument("--root",type=Path,default=DEFAULT)
    args=parser.parse_args()
    base=json.loads((HERE/"drive-catalog.json").read_text())
    old={v["name"] for v in base["volumes"]}
    pending={}
    final=args.root/"drive-backup-final"
    # The sealed interim endpoint was lost to a WSL restart before publication.
    # Its identifiers stay reserved. Only the published progress packs and new
    # packs from the persistent final stage belong in the remote edition.
    for stage in (args.root/"drive-backup-progress",final):
        cached=HERE/('drive-'+stage.name.rsplit('-',1)[-1]+'-catalog.json')
        if cached.exists():
            for row in json.loads(cached.read_text())['volumes']:
                n=row['name']
                if n.startswith('pack-') and n not in old:pending[n]=stage
            continue
        if stage.exists():
            for p in stage.glob("pack-*.zip"):
                if p.name not in old:pending[p.name]=stage
    sealed=(final/"FINAL-README.md").is_file()
    if sealed:
        for n in PUBLIC:pending[n]=final
    ready=[];missing=0
    for n,stage in sorted(pending.items()):
        slot=int(n[5:10]) if n.startswith("pack-") else PUBLIC.index(n)
        if slot%args.slots!=args.slot:continue
        receipt=HERE/"drive-receipts"/(n+".json")
        if args.kind=="upload":
            if receipt.exists():continue
            ready.append({"name":n,"stage":stage.name});missing+=1
        elif not (HERE/"drive-verified"/(n+".json")).exists():
            missing+=1
            if receipt.exists():ready.append(json.loads(receipt.read_text()))
    print(json.dumps({"final":sealed,"missing":missing,"ready":ready[:10]}))
if __name__=="__main__":main()
