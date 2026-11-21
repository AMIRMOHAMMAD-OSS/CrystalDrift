"""Write matched PBE single-point inputs from archived MgGa3 geometries."""
import json
from pathlib import Path
import re
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from data import unpack_gram

PSEUDOS={12:"Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",31:"Ga.pbe-dn-kjpaw_psl.1.0.0.UPF"}
SYMBOLS={12:"Mg",31:"Ga"}


def main():
    # The stricter of the published recommended cutoffs, with rounded floors.
    cutoffs=[]
    for filename in PSEUDOS.values():
        raw=(HERE/"pseudos"/filename).read_text()
        w=re.search(r'wfc_cutoff\s*=\s*"([^"]+)"',raw)
        r=re.search(r'rho_cutoff\s*=\s*"([^"]+)"',raw)
        if w and r:cutoffs.append((float(w[1]),float(r[1])))
    ecut=int(np.ceil(max([65]+[w for w,r in cutoffs])/5)*5)
    density=int(np.ceil(max([8*ecut]+[r for w,r in cutoffs])/10)*10)
    with np.load(ROOT/"expansion/denovo/pools/learned-drift.npz") as pool:
        first = next(i for i,row in enumerate(pool["z"][:16]) if 0 < (row>0).sum() <= 4 and row.max() <= 36)
    assert first == 6
    for method in ("learned-drift","learned-flow16"):
        identifier=f"{method}-{first+1:04d}"
        inp=json.loads((ROOT/f"expansion/physical/inputs/{identifier}.json").read_text())
        obs=json.loads((ROOT/f"expansion/physical/records/{identifier}.json").read_text())
        assert inp["z"]==[12,31,31,31]
        for frame,k,f in (("raw",inp["k"],inp["f"]),("relaxed",obs["final_k"],obs["final_f"])):
            cell=np.linalg.cholesky(unpack_gram(k))
            for mesh in (2,3):
                name=f"{identifier}-{frame}-k{mesh}"
                # Paths pseudo and scratch are supplied by run.sh at execution time.
                text=f"""&CONTROL
 calculation='scf', prefix='{name}', pseudo_dir='PSEUDODIR',
 outdir='SCRATCHDIR', disk_io='none', tprnfor=.true., tstress=.false.,
/
&SYSTEM
 ibrav=0, nat=4, ntyp=2, ecutwfc={ecut}, ecutrho={density},
 input_dft='PBE', occupations='smearing', smearing='mv', degauss=0.02,
 nspin=1, nosym=.true.,
/
&ELECTRONS
 conv_thr=1.0d-8, electron_maxstep=160, mixing_beta=0.25,
/
ATOMIC_SPECIES
Mg 24.305 {PSEUDOS[12]}
Ga 69.723 {PSEUDOS[31]}
CELL_PARAMETERS angstrom
"""
                text+="\n".join(" ".join(f"{x:.12f}" for x in row) for row in cell)
                text+="\nATOMIC_POSITIONS crystal\n"
                text+="\n".join(SYMBOLS[z]+" "+" ".join(f"{x:.12f}" for x in row) for z,row in zip(inp["z"],f))
                text+=f"\nK_POINTS automatic\n{mesh} {mesh} {mesh} 0 0 0\n"
                (HERE/f"{name}.in").write_text(text)
    print("ecutwfc",ecut,"ecutrho",density,flush=True)

if __name__=="__main__":main()
