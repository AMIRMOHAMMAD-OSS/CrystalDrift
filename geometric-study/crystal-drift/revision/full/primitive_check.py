"""Check the exact screen prerequisite for all targets and selected change-of-cell cases."""
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure,Lattice
from match_full import HERE,ROOT,make_structure,target_context,candidate,pin
from data import pack_gram


def pack(s):
    z=np.asarray(s.atomic_numbers,np.int64);order=np.argsort(z,kind='stable')
    g=s.lattice.matrix@s.lattice.matrix.T
    return pack_gram(g),s.frac_coords[order]%1,z[order],g


def main():
    pin()
    with np.load(HERE/'mp20/cohort.npz') as f:src={k:f[k] for k in ('ids','k','f','z')}
    change=[]
    for i,mid in enumerate(src['ids']):
        truth=make_structure(src['k'][i],src['f'][i],src['z'][i])
        # Same public reduction sequence as pymatgen 2024.4.13's matcher.
        primitive=truth.get_reduced_structure(reduction_algo='niggli').get_primitive_structure()
        if len(primitive)!=len(truth):change.append({'id':str(mid),'native':len(truth),'primitive':len(primitive)})
    A=np.array([[4.1,0.,0.],[1.2,5.4,0.],[-.9,1.1,6.2]])
    f=np.array([[.05,.13,.25],[.39,.49,.67],[.83,.26,.54]])
    base=Structure(Lattice(A),[3,8,8],f)
    U=np.array([[1,2,-1],[0,1,1],[0,0,1]])
    rebased=Structure(Lattice(U@A),[3,8,8],(f@np.linalg.inv(U)+[.26,.88,.11])%1)
    supercell=base.copy();supercell.make_supercell([2,1,1])
    native_perturbed=supercell.copy()
    native_perturbed.translate_sites([1],[.17,.18,.21],to_unit_cell=True)
    scaled=Structure(Lattice(1.3*A),[3,8,8],f)
    other=Structure(Lattice(np.diag([3.,4.,12.])),[3,8,8],[[.2,.2,.3],[.78,.4,.1],[.8,.82,.9]])
    tests=[]
    for name,truth,pred in [('integer shear and translation',base,rebased),('native scale',base,scaled),
            ('larger-cell primitive reference',supercell,supercell.copy()),
            ('perturbed correspondence in supercell',supercell,native_perturbed),
            ('larger-cell primitive prediction',native_perturbed,supercell),
            ('unequal triclinic and orthogonal structures',base,other)]:
        kt,ft,zt,gt=pack(truth);kp,fp,zp,gp=pack(pred)
        ctx=target_context(kt,ft,zt);rms,strict,forwarded=candidate(ctx,kp,fp,zp,gp,True)
        tests.append(dict(name=name,native_target=len(zt),primitive_target=len(ctx[1]),rms=float(rms) if np.isfinite(rms) else None,
                          strict=bool(strict),forwarded=bool(forwarded),native_ratio=float(pred.volume/truth.volume)))
    output=dict(release_target_count=len(src['ids']),native_site_count_retained=len(src['ids'])-len(change),
                reference_reduction_changes=change,fixtures=tests)
    (HERE/'mp20/primitive-prerequisites.json').write_text(json.dumps(output,indent=1)+'\n')
    print('Reduced-count changes',len(change),'of',len(src['ids']),'fixtures',len(tests),flush=True)


if __name__=='__main__':main()
