"""Independent numerical checks for the actual arms, finite field, gradients and witnesses."""
import argparse
import itertools
import json
from pathlib import Path
from platform import platform
from time import perf_counter

import numpy as np
import torch
from lattice_symmetry.v2.physical import Generator
from scaled.family import within_direction
from .descriptors import (ARMS, SLOTS, SHELLS, RECIPROCAL_CUTOFF, bank_direction, combine,
                          drift_loss, feature_blocks, fit_stats, reciprocal_indices, reciprocal_shells)

HERE = Path(__file__).resolve().parent
# The six prospectively screened feature maps. The later four-shell sensitivity has
# its own independent implementation/precision check in radial_check.py.
SCREEN_ARMS = ("pair","center0","angular","variance","contacts","reciprocal")


def fixtures():
    """P2_1 with repeated same-species orbits and an independent inversion special position."""
    rng = np.random.default_rng(519203)
    records = []
    for _ in range(9):
        L = np.array([[5.24,0,.15],[0,6.01,0],[-.43,0,5.72]]) * rng.uniform(.88,1.16)
        generators = rng.uniform(.05,.94,size=(3,3))
        mates = generators @ np.diag([-1.,1.,-1.]) + [0,.5,0]
        F = np.stack((generators,mates%1),axis=1).reshape(-1,3)
        records.append((L,F,np.array([3,3,8,8,3,3]),np.array([4,4,9,9,12,12])))
    return records


def change(record,rng,which):
    L,F,Z,O = record
    U = np.eye(3,dtype=int)
    U[0,1] = [11,-4,3][which%3]
    U[1,2] = [-2,1,2][which%3]
    if which%2:
        U[[1,2]] = U[[2,1]]
    Q, _ = np.linalg.qr(rng.normal(size=(3,3)))
    if (np.linalg.det(Q) < 0) != (which%2 == 0):
        Q[:,0] *= -1
    t = rng.uniform(-1,1,size=3)
    p = rng.permutation(len(F))
    labels = {o: int(v) for o,v in zip(np.unique(O),rng.choice(400, len(set(O)), replace=False))}
    Fp = (F[p] @ np.linalg.inv(U) + t) % 1 + rng.integers(-2,3,F.shape)
    return (U@L@Q,Fp,Z[p],np.array([labels[o] for o in O[p]])),(U,Q,p,t)


def pack(records,dtype,device,gradient=False):
    L,F,Z,O = zip(*records)
    L = torch.as_tensor(np.stack(L),device=device,dtype=dtype).requires_grad_(gradient)
    slots = max(map(len,F))+1
    f = np.zeros((len(L),slots,3)); z = np.zeros((len(L),slots),dtype=int)
    o = np.full((len(L),slots), -47,dtype=int)
    for i,(fi,zi,oi) in enumerate(zip(F,Z,O)):
        # Padding deliberately interrupts some real-site sequences.
        active = np.delete(np.arange(slots),i%slots)[:len(fi)]
        f[i,active] = fi; z[i,active] = zi; o[i,active] = oi
    return (L,torch.as_tensor(f,device=device,dtype=dtype).requires_grad_(gradient),
            torch.as_tensor(z,device=device,dtype=torch.long),torch.as_tensor(o,device=device,dtype=torch.long))


def diff(a,b):
    return float((a-b).abs().max().detach())


def weights(t):
    return np.exp(1-1/(1-t))[:,None] * np.exp(-.5*((t[:,None]-np.linspace(0,1,SHELLS))*2)**2)


def independent_contacts(record):
    L,F,Z,O = record
    groups = list(np.unique(O)); species, ranks = np.unique(Z,return_inverse=True)
    bound = int(np.ceil(4/np.linalg.svd(L,compute_uv=False).min() + np.ptp(F,axis=0).max()))
    shifts = np.asarray(list(itertools.product(range(-bound,bound+1),repeat=3)))
    parts = [[] for _ in range(SLOTS*(SLOTS+1)//2*4)]
    for a,b in itertools.combinations(groups,2):
        ai,bi = np.flatnonzero(O==a), np.flatnonzero(O==b)
        lo,hi = sorted([ranks[ai[0]],ranks[bi[0]]])
        cls = ((hi*(hi+1)//2+lo)*2+int(len(ai)==len(bi)))*2+int(min(len(ai),len(bi))>1)
        dyad = np.zeros(SHELLS)
        for i,j in itertools.product(ai,bi):
            t = np.sum(((F[j]-F[i]+shifts)@L/4)**2,axis=1)
            dyad += weights(t[t<1]).sum(0)/(len(ai)*len(bi))
        parts[cls].append(dyad)
    result = np.zeros((len(parts),2*SHELLS))
    for i,v in enumerate(parts):
        if v:
            result[i] = np.r_[np.mean(v,axis=0),np.square(v).mean(axis=0)]
    return result.ravel()


def independent_reciprocal(record,truncated=False):
    L,F,Z,_ = record
    bound = 1 if truncated else int(np.ceil(np.linalg.norm(L,2)*RECIPROCAL_CUTOFF))
    h = np.asarray([a for a in itertools.product(range(-bound,bound+1),repeat=3) if any(a)])
    g = np.linalg.solve(L,h.T).T
    t = np.sum((g/RECIPROCAL_CUTOFF)**2,axis=1)
    h,t = h[t<1],t[t<1]
    phases = np.exp(2j*np.pi*(F@h.T))
    s = np.zeros((SLOTS,len(h)),dtype=complex)
    for i,z in enumerate(np.unique(Z)):
        s[i] = phases[Z==z].mean(0)
    a,b = np.triu_indices(SLOTS)
    return ((s[a]*s[b].conj()).real@weights(t)/(abs(np.linalg.det(L))*4*np.pi/3*
            RECIPROCAL_CUTOFF**3)).ravel()


def witness(dtype,device):
    def record(L,shift):
        return (np.diag(L),np.array([[0.,0,0],shift]),np.array([3,8]),np.array([4,12]))
    cases = {
        "variance_blind_to_contact": [record([7,8,9],[x,.01,0]) for x in (.18,.36)],
        "fixed_geometry_different_subgroups": [
            (np.eye(3)*6,np.array([[.1,0,0],[.9,0,0]]),np.array([8,8]),np.array(partition))
            for partition in ([0,1],[0,0])],
        "identical_local_shells": [record([22,13,15],[x,.03,0]) for x in (.24,.38)],
        "homometric_nonidentifiability": [(np.diag([50.,5,5]),np.column_stack(
            [np.array(s)/12,np.zeros((4,2))]), np.full(4,6),np.array([3,8,2,9]))
            for s in ((0,1,4,6),(0,1,3,7))]}
    out = {}
    for name,rr in cases.items():
        args = pack(rr,dtype,device)
        vals = {a:feature_blocks(*args,arm=a) for a in ("variance","contacts","reciprocal")}
        gaps = {k:float((v[0]-v[1]).norm()) for rows in vals.values() for k,v in rows.items()}
        out[name] = gaps
    assert out["variance_blind_to_contact"]["variance"] == 0 and out["variance_blind_to_contact"]["contacts"]>.002
    partition=out["fixed_geometry_different_subgroups"]
    assert partition["base"]==partition["reciprocal"]==partition["variance"]==0 and partition["contacts"]>0.1
    assert out["identical_local_shells"]["base"] == 0 and out["identical_local_shells"]["reciprocal"]>.001
    assert max(out["homometric_nonidentifiability"].values()) < (1e-11 if dtype==torch.float64 else 1e-4)
    return out


def edge_fixtures(dtype,device):
    records = [
        (np.diag([9.,10,11]),np.array([[0,0,0],[4/9,0,0],[.5,.5,.5]]),np.array([6,6,8]),np.arange(3)),
        (np.diag([1.,1.,1.]),np.array([[0,0,0]]),np.array([14]),np.array([99])),
        (np.diag([2./RECIPROCAL_CUTOFF,6,7]),np.array([[0,0,0],[.5,.5,.5]]),
         np.array([3,8]),np.array([123,47]))]
    result = []
    for i,r in enumerate(records):
        # This includes a radial boundary, empty reciprocal sphere, no inter-orbit pairs,
        # fixed inversion centers and the reciprocal h=(2,0,0) at the physical cutoff.
        raw = []
        for e in (-1e-6,0,1e-6):
            scaled = (r[0]*(1+e),*r[1:])
            args = pack([scaled],dtype,device,True)
            xs = [feature_blocks(*args,arm=k)[k] for k in ("contacts","reciprocal")]
            value = torch.cat(xs,1)
            grads = torch.autograd.grad(value.square().sum(),args[:2],allow_unused=True)
            assert torch.isfinite(value).all() and all(v is None or torch.isfinite(v).all() for v in grads)
            raw.append(value.detach())
        result.append(max(diff(raw[0],raw[1]),diff(raw[1],raw[2])))
    assert max(result) < 1e-4
    return result


def verify(dtype,device):
    torch.manual_seed(337)
    rng = np.random.default_rng(697301)
    records = fixtures()
    transformed,actions = zip(*(change(row,rng,i) for i,row in enumerate(records)))
    original = pack(records,dtype,device)
    changed = pack(transformed,dtype,device)
    tol = 3e-9 if dtype==torch.float64 else 2e-2
    report = {"descriptor":{}, "finite_bank":{}, "decoder":{}, "affinity_fault_tests":{}}
    raw1 = {}; raw2 = {}
    for arm in SCREEN_ARMS:
        x,y = feature_blocks(*original,arm=arm),feature_blocks(*changed,arm=arm)
        raw1[arm],raw2[arm] = x,y
        report["descriptor"][arm] = {k:diff(value,y[k]) for k,value in x.items()}
        assert max(report["descriptor"][arm].values()) < tol, (arm,report)
    for arm,fn in (("contacts",independent_contacts),("reciprocal",independent_reciprocal)):
        vals = np.stack([fn(r) for r in records])
        error = np.abs(vals-raw1[arm][arm].detach().cpu().numpy()).max()
        report["descriptor"][arm]["numpy_oracle"] = float(error)
        assert error < tol
    # Frozen illustrative standardizers use independent synthetic calibration structures.
    calibration = [(l*1.06,(f*.91)%1,z,o) for l,f,z,o in records]
    for arm in SCREEN_ARMS:
        stats = fit_stats(feature_blocks(*pack(calibration,dtype,device),arm=arm))
        def experiment(raw):
            y = combine(raw,stats).reshape(3,3,-1)
            return (y[0][None],y[1][None],y[2][None])
        banks = experiment(raw1[arm]); alt = experiment(raw2[arm])
        for law in ("laplace","gaussian"):
            v = bank_direction(*banks,law)
            reps = diff(v,bank_direction(*alt,law))
            swap = diff(v,-bank_direction(banks[0],banks[2],banks[1],law))
            live = banks[0].detach().requires_grad_(True)
            grad = torch.autograd.grad(drift_loss(live,*banks[1:],law),live)[0]
            analytic = diff(grad,-v/live.shape[1])
            row = dict(representation=reps,bank_swap=swap,stopped_gradient=analytic,
                       single_sample_norm=float(v.norm()))
            if law=="laplace":
                row["v2_field_parity"] = diff(v,within_direction(*banks))
            report["finite_bank"][f"{arm}_{law}"] = row
            assert max(reps,swap,analytic,row.get("v2_field_parity",0)) < tol
    # Equivariant physical generator is independently exercised as a composition with D/E.
    model = Generator().to(device=device,dtype=dtype)
    for arm in ("contacts","reciprocal"):
        stats = fit_stats(feature_blocks(*pack(calibration,dtype,device),arm=arm))
        def composed(args):
            L,F = model(*args[:3]); y = combine(feature_blocks(L,F,*args[2:],arm=arm),stats)
            q,p,n = y.reshape(3,3,-1).unbind(0)
            loss = drift_loss(q[None],p[None],n[None])
            return torch.cat([v.flatten() for v in torch.autograd.grad(loss,model.parameters())]),L,F
        grad,L,F = composed(original); gradp,Lp,Fp = composed(changed)
        relgrad = float((grad-gradp).norm()/grad.norm().clamp_min(1e-12))
        cell_err, site_err = 0.,0.
        for i,(u,q,p,t) in enumerate(actions):
            up,qp = [torch.as_tensor(a,device=device,dtype=dtype) for a in (u,q)]
            cell_err = max(cell_err,diff(up@L[i]@qp,Lp[i]))
            expect = F[i,original[2][i]>0][p] @ torch.linalg.inv(up) + F.new_tensor(t)
            torus = (Fp[i,changed[2][i]>0]-expect+.5).remainder(1)-.5
            site_err = max(site_err,float(torus.abs().max()))
        report["decoder"][arm] = dict(relative_parameter_gradient=relgrad,cell=cell_err,torus=site_err)
        assert max(relgrad,cell_err,site_err) < tol
    rawgap = np.linalg.norm(records[0][0]-transformed[0][0])
    boxgap = np.linalg.norm(independent_reciprocal(records[0],True)-
                            independent_reciprocal(transformed[0],True))
    assert rawgap > 1 and boxgap > 1e-3
    report["affinity_fault_tests"] = dict(raw_cell_norm=float(rawgap),cubical_index_error=float(boxgap))
    report["witnesses"] = witness(dtype,device)
    report["boundary_feature_changes"] = edge_fixtures(dtype,device)
    if dtype==torch.float64:
        args = pack(records[:1],dtype,device,True)
        dx = torch.linspace(.03,.44,args[1].numel(),device=device,dtype=dtype).reshape_as(args[1])
        dl = torch.linspace(-.031,.041,9,device=device,dtype=dtype).reshape_as(args[0])
        for arm in ("contacts","reciprocal"):
            def objective(l,f):
                return feature_blocks(l,f,*args[2:],arm=arm)[arm].square().sum()
            gl,gf = torch.autograd.grad(objective(*args[:2]),args[:2])
            analytic = (gl*dl).sum()+(gf*dx).sum()
            h = 1e-5
            numeric = (objective(args[0]+h*dl,args[1]+h*dx)-objective(args[0]-h*dl,args[1]-h*dx))/(2*h)
            fd = float((numeric-analytic).abs())
            report["descriptor"][arm]["directional_derivative_error"] = fd
            assert fd < 5e-7
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("cpu","cuda"),default="cpu")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.device=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    start=perf_counter()
    res=dict(device=args.device,hardware=torch.cuda.get_device_name() if args.device=="cuda" else platform(),
             torch=torch.__version__,numpy=np.__version__,fixture_seed=519203,transform_seed=697301,
             exact_arithmetic_group="GL(3,Z) x O(3), passive origin, site/orbit relabelling and wraps",
             tolerance={"float64":3e-9,"float32":2e-2}, cases={})
    for dt in (torch.float64,torch.float32):
        res["cases"][str(dt)] = verify(dt,args.device)
    res["seconds"] = perf_counter()-start
    target = args.output or HERE/f"verification-{args.device}.json"
    target.write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8")
    print(target, res["seconds"])


if __name__ == "__main__":
    main()
