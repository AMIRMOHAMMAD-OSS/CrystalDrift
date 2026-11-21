"""Generate unfiltered multielement structures without consulting the test set."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scaled"))
from family import SetNetwork, draw
from prior import CompositionPrior


def cells(k):
    s = np.zeros((len(k), 3, 3), float)
    s[:, range(3), range(3)] = k[:, :3]
    for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
        s[:, a, b] = s[:, b, a] = k[:, col] / np.sqrt(2)
    w, q = np.linalg.eigh(s)
    return (q * np.exp(w)[:, None, :]) @ q.swapaxes(-1, -2)


def setup(method, device):
    fit = "drift" if method == "drift" else "flow"
    path = ROOT / f"scaled/runs/mp20/{fit}-seed11.pt"
    state = torch.load(path, weights_only=True, map_location=device)
    net = SetNetwork().to(device).eval()
    net.load_state_dict(state["model"])
    return net, state, fit, path


def generate(method, kind, prior, n, batch, device, net, state, fit):
    rng = np.random.default_rng(22112026)
    geom = np.random.default_rng(71831)
    zs = []
    ks = []
    fs = []
    Ls = []
    Xs = []
    clock = []
    for start in range(0, n, batch):
        t0 = time.perf_counter()
        size = min(n - start, batch)
        z = prior.sample(rng, size, kind)
        k = geom.standard_normal((size, 6)).astype(np.float32)
        f = geom.random((size, 20, 3), dtype=np.float32)
        at = torch.as_tensor(z, dtype=torch.long, device=device)
        t = torch.as_tensor(k, device=device)
        u = torch.as_tensor(f, device=device)
        with torch.inference_mode():
            k, f = draw(net, fit, at, t, u, 16 if method == "flow16" else 1)
            k = (k * state["scale"] + state["mean"]).cpu().numpy()
            f = f.cpu().numpy()
        L = cells(k)
        X = f @ L
        clock.append(time.perf_counter() - t0)
        zs.append(z)
        ks.append(k)
        fs.append(f)
        Ls.append(L)
        Xs.append(X)
    z, k, f, L, X = map(np.concatenate, (zs, ks, fs, Ls, Xs))
    good = np.isfinite(k).all(-1) & np.isfinite(f).all((1, 2)) & np.isfinite(L).all((1, 2))
    return dict(
        ids=np.array([f"d{i+1:05d}" for i in range(n)]),
        z=z,
        k=k,
        f=f,
        lattice=L,
        valid_output=good,
        batch_seconds=clock,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--arms", nargs="+", default=["learned-drift", "empirical-drift", "learned-flow16"]
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    out = HERE / "pools"
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT / "scaled/mp20-data/train.npz"
    with np.load(path) as d:
        train = d["z"]
    with np.load(ROOT / "scaled/mp20-data/val.npz") as d:
        val = d["z"]
    prior = CompositionPrior(train)
    val_report = {}
    for a in (0.5, 4.0, 32.0):
        prior.alpha = a
        prior.cache.clear()
        val_report[str(a)] = prior.validate(val)
    selected = min(val_report, key=lambda a: val_report[a]["conditional_nll_nats"])
    prior.alpha = float(selected)
    prior.cache.clear()
    summary = dict(
        selected_shrinkage=prior.alpha,
        validation=val_report,
        train_rows=len(train),
        species=prior.elements.tolist(),
        distinct_patterns=len(prior.keys),
        train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        conditional="exact sorted unit-cell multiplicities",
    )
    (HERE / "prior-fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    for arm in args.arms:
        kind, method = arm.split("-")
        net, state, fit, checkpoint = setup(method, args.device)
        payload = generate(method, kind, prior, 128, args.batch, args.device, net, state, fit)
        t0 = time.perf_counter()
        payload = generate(method, kind, prior, 10000, args.batch, args.device, net, state, fit)
        total = time.perf_counter() - t0
        np.savez_compressed(out / (arm + ".npz"), **payload)
        serial = generate(method, kind, prior, 128, 1, args.device, net, state, fit)
        meta = dict(
            arm=arm,
            device=args.device,
            torch=torch.__version__,
            numpy=np.__version__,
            python=platform.python_version(),
            seed_fit=11,
            composition_seed=22112026,
            geometry_seed=71831,
            n=10000,
            finite=int(payload["valid_output"].sum()),
            batch=args.batch,
            elapsed_s=total,
            completed_batch_seconds=payload["batch_seconds"],
            b1_ms=np.array(serial["batch_seconds"]).__mul__(1000).tolist(),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source=str(checkpoint.relative_to(ROOT)),
            seconds_in_model_and_sampling=float(sum(payload["batch_seconds"])),
            nfe=16 if method == "flow16" else 1,
            cpu_threads=torch.get_num_threads(),
        )
        (out / (arm + ".json")).write_text(json.dumps(meta, indent=1) + "\n")
        print(arm, total, meta["finite"], flush=True)


if __name__ == "__main__":
    main()
