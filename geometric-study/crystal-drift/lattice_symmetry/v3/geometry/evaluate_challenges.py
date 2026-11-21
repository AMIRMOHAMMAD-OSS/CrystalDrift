"""Run both frozen models' small protected panels serially and reduce every completed score."""
import argparse,json,os,subprocess,sys
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"

HERE=Path(__file__).resolve().parent;V3=HERE.parent;ROOT=V3.parents[1];E=V3/"evaluation"
def main():
    p=argparse.ArgumentParser();p.add_argument("--matcher",required=True)
    p.add_argument("--matrix",type=Path,nargs="+",help="Additional frozen update registers to run in place of the first two matrices.")
    p.add_argument("--panels",nargs="+",default=["arrangement","arrangement-control","family","shared"])
    arg=p.parse_args()
    first=E/"first-frozen.json";control=E/"controlled-frozen.json"
    def tags(f):return [r["tag"] for r in json.loads(f.read_text())["methods"]]
    matrices=arg.matrix or [first,control]
    for register in matrices:
        assert json.loads(register.read_text())["frozen"] is True
    def run(*cmd):
        subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)
    for panel in arg.panels:
        allnames=list(dict.fromkeys(name for matrix in matrices for name in tags(matrix)))
        for matrix in matrices:
            # Already sealed candidate shards are reused after validating their signatures.
            run(sys.executable,E/"paired.py",matrix,"--panel",panel,"--threads","1")
        run(arg.matcher,E/"shards.py",panel,*allnames,"--workers","1")
        for tag in allnames:
            assert (E/f"scores/{panel}/{tag}/outcomes.npz").is_file(),(panel,tag)
        run(sys.executable,E/"analysis.py",panel)
        run(sys.executable,"-m","lattice_symmetry.v3.geometry.conditional",panel,*allnames)
        for tag in allnames:
            if tag!="prior":
                run(sys.executable,"-m","lattice_symmetry.v3.geometry.family_effects",panel,tag,tag)
        groups={tag.rsplit("-",1)[0] for tag in allnames if tag.rsplit("-",1)[-1] in {"11","23","37"}}
        for group in sorted(groups):
            seeds=[f"{group}-{seed}" for seed in (11,23,37)]
            if all((E/f"scores/{panel}/{tag}/outcomes.npz").is_file() for tag in seeds):
                run(sys.executable,"-m","lattice_symmetry.v3.geometry.family_effects",panel,group+"-three-seeds",*seeds)
    print("Completed frozen challenge panels",flush=True)
if __name__=="__main__":main()
