"""Time the published, unconditional, 100-step CrystalFlow sampler on 10,000 fresh draws."""
import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from crystalflow import HERE, ROOT, EncoderCache, fast_edges, load_model
from prepare_mp20 import pack_gram


def site_distribution(source):
    tree = ast.parse((source / "scripts/generation.py").read_text())
    assignment, = [x for x in tree.body if isinstance(x, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "train_dist" for t in x.targets)]
    # Read only the literal used in the MP-20 branch of the authors' sample dataset.
    for key, value in zip(assignment.value.keys, assignment.value.values):
        if ast.literal_eval(key) == "mp_20":
            p = np.array(ast.literal_eval(value))
            assert len(p) == 21 and p[0] == 0 and np.isclose(p.sum(), 1)
            return p
    raise ValueError("Pinned source lacks the MP-20 atom-count distribution")


@torch.inference_mode()
def propose(state, counts, steps, audit=False):
    model, emb, decode, cache = state
    b = len(counts)
    device = next(model.parameters()).device
    num = torch.as_tensor(counts, device=device)
    graph = torch.from_numpy(np.repeat(np.arange(b), counts)).to(device)
    l = torch.randn(b, 6, device=device) * .1
    l[:, -1] += 1
    nsites = int(counts.sum())
    x = torch.rand(nsites, 3).to(device)
    types = model.type_encoding.get_rd_encoded_types(nsites, device=device)
    edges = fast_edges(counts, device)
    original_edges = model.gen_edges
    if cache is not None:
        model.gen_edges = lambda num, coord, mat, g: (edges, (coord[edges[1]] - coord[edges[0]]) % 1.)
    try:
        for i in range(1, steps + 1):
            if cache is not None:
                cache.cache.clear()
            times = emb(torch.full((b,), i / steps, device=device))
            cell = decode(l)
            dl, dx, dt = model(times, types, x, l, num, graph, lattices_mat=cell)
            l = l + dl / steps
            x = (x + dx / steps) % 1.
            types = types + dt / steps
    finally:
        model.gen_edges = original_edges
    types_finite = torch.isfinite(types).all(1).cpu().numpy()
    atoms = model.type_encoding.decode_types(types).cpu().numpy()
    coords = x.cpu().numpy()
    q = l.double().cpu().numpy()
    k = np.column_stack((q[:, 3] + q[:, 4] + q[:, 5], -q[:, 3] + q[:, 4] + q[:, 5],
                         -2 * q[:, 4] + q[:, 5], np.sqrt(2) * q[:, :3]))
    mat = np.zeros((b, 3, 3))
    mat[:, np.arange(3), np.arange(3)] = k[:, :3]
    i, j = np.triu_indices(3, 1)
    mat[:, i, j] = mat[:, j, i] = k[:, 3:] / np.sqrt(2)
    finite = np.isfinite(mat).all((1, 2))
    lattices = np.full((b, 3, 3),np.nan)
    try:
        lam, vec = np.linalg.eigh(mat[finite])
        lattices[finite] = (vec * np.exp(lam)[:, None, :]) @ vec.transpose(0, 2, 1)
    except np.linalg.LinAlgError:
        for row in np.flatnonzero(finite):
            try:
                lam,vec=np.linalg.eigh(mat[row])
                lattices[row]=(vec*np.exp(lam))@vec.T
            except np.linalg.LinAlgError:
                pass
    if audit:
        np.testing.assert_allclose(lattices, decode(l).cpu().numpy(), atol=5e-4, rtol=5e-5)
    z = np.zeros((b, 20), np.int16)
    f = np.zeros((b, 20, 3), np.float32)
    offset = 0
    for row, count in enumerate(counts):
        group = slice(offset, offset + count)
        order = np.argsort(atoms[group], kind="stable")
        z[row, :count] = atoms[group][order]
        f[row, :count] = coords[group][order]
        offset += count
    type_ok=np.logical_and.reduceat(types_finite,np.r_[0,np.cumsum(counts)[:-1]])
    valid = type_ok & np.isfinite(k).all(1) & np.isfinite(f).all((1, 2)) & np.isfinite(lattices).all((1, 2))
    return z, k, f, lattices, f @ lattices, valid


def main(opts):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    params = json.loads((opts.weights / "model.json").read_text())
    assert params["type_encoding"] == "table" and params["decoder"]["pred_type"]
    assert not params.get("from_cubic", False)
    counts_p = site_distribution(opts.source)
    net, emb, build = load_model(opts.source, opts.weights, opts.device)
    from check_sampler import author_adapter
    adapter=author_adapter(opts.source,net,emb,build,True)
    @torch.inference_mode()
    def native(counts):
        b=len(counts)
        dev=next(net.parameters()).device
        batch=SimpleNamespace(num_graphs=b,num_nodes=int(counts.sum()),
            num_atoms=torch.as_tensor(counts,device=dev),
            batch=torch.as_tensor(np.repeat(np.arange(b),counts),device=dev))
        result,_=adapter.sample(batch,N=100)
        L=result["lattices"].double().cpu().numpy()
        coords=result["frac_coords"].cpu().numpy()
        atoms=result["atom_types"].cpu().numpy()
        k=np.full((b,6),np.nan);z=np.zeros((b,20),np.int16);f=np.zeros((b,20,3),np.float32)
        valid=np.isfinite(L).all((1,2))
        good=np.flatnonzero(valid)
        if len(good):
            gram=L[good]@L[good].transpose(0,2,1)
            try:k[good]=pack_gram(gram)
            except (ValueError,np.linalg.LinAlgError):
                for j,g in zip(good,gram):
                    try:k[j]=pack_gram(g)
                    except (ValueError,np.linalg.LinAlgError):valid[j]=False
        first=0
        for i,count in enumerate(counts):
            group=slice(first,first+count);order=np.argsort(atoms[group],kind="stable")
            z[i,:count]=atoms[group][order]
            f[i,:count]=coords[group][order]
            first+=count
        valid &= np.isfinite(f).all((1,2))&np.isfinite(k).all(1)
        return z,k,f,L,f@L,valid
    if opts.pilot:
        checked=native(np.array([2,4],dtype=np.int64))
        assert checked[-1].all() and (checked[0]>0).sum()==6
        print("Verified native 100-step joint endpoint",flush=True)
        return
    if opts.device != "cuda":raise ValueError("The matched timing workload requires CUDA")
    out = ROOT / "expansion/denovo/pools"
    batch = 24
    rng = np.random.RandomState(241125)
    steps = 100
    for _ in range(3):
        native(rng.choice(len(counts_p), batch, p=counts_p))
    torch.cuda.synchronize()
    torch.manual_seed(241127)
    rng = np.random.RandomState(241127)
    start = time.perf_counter()
    counts = rng.choice(len(counts_p), 10000, p=counts_p)
    count_seconds = time.perf_counter() - start
    chunks = []
    for i in range(0, 10000, batch):
        chunks.append(native(counts[i:i + batch]))
        if i and i%1200==0:print("Completed",i+len(counts[i:i+batch]),"draws",round(time.perf_counter()-start,1),"s",flush=True)
    z, k, f, L, cart, valid = map(np.concatenate, zip(*chunks))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ids = [f"crystalflow-replay-{i:05d}" for i in range(10000)]
    archive=out / "crystalflow-joint-local.npz"
    temp=archive.with_suffix(".partial.npz")
    np.savez_compressed(temp, ids=ids, z=z, k=k, f=f, lattice=L,
                        valid_output=valid, num_sites_drawn=counts)
    temp.replace(archive)
    print("Wrote 10,000 raw proposals", elapsed, flush=True)
    times = []
    torch.manual_seed(241128)
    rng = np.random.RandomState(241128)
    for _ in range(131):
        before = time.perf_counter()
        native(rng.choice(len(counts_p), 1, p=counts_p))
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - before))
    record = dict(arm="crystalflow-joint-local", draws=10000, batch=batch, final_batch=16, warmup_batches=3,
                  size_prior="Pinned CrystalFlow scripts/generation.py train_dist[mp_20]", nfe=steps,
                  decoder_params=sum(p.numel() for p in net.parameters()), coord_anneal=False,
                  type_encoding="Released TypeTableModule, 13 rows and 15 columns", b1_ms=times[3:],
                  seed=241127, serial_seed=241128, site_count_seconds=count_seconds,
                  elapsed_s=elapsed, per_second=10000 / elapsed, sampler="Pinned unmodified CSPFlow.sample and CSPNet fully connected graph, native PyTorch scatter mean",
                  threads=1, precision="float32, TF32 disabled", device=torch.cuda.get_device_name(),
                  torch=torch.__version__, checkpoint="ixsluo/CrystalFlow v1.0.0-alpha.1 / DNG-mp-20",
                  cost_boundary="From count sampling to host species, fractional sites, canonical lattice and Cartesian sites")
    (out / "crystalflow-joint-local.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Median B1 milliseconds", np.median(times[3:]), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--pilot", action="store_true")
    main(p.parse_args())
