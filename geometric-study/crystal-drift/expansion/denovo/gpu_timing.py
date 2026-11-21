"""Same-GPU compact de novo timing on the preserved composition sequence at batch 24."""

import argparse, json, time
import numpy as np
import torch
from generate import ROOT, HERE, setup, cells, generate
from prior import CompositionPrior
from family import draw

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--match-noise",
    action="store_true",
    help="Replay the exact CPU draw-order lattice and site noise",
)
args = p.parse_args()
torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
with np.load(ROOT / "scaled/mp20-data/train.npz") as t:
    prior = CompositionPrior(t["z"], 4.0)
for arm in ("learned-drift", "empirical-drift", "learned-flow16"):
    kind, method = arm.split("-")
    net, state, fit, path = setup(method, "cuda")
    with np.load(HERE / f"pools/{arm}.npz") as pool:
        expected = pool["z"]
        if args.match_noise:
            expected_k, expected_f = pool["k"], pool["f"]
    for _ in range(3):
        generate(method, kind, prior, 24, 24, "cuda", net, state, fit)
    rng = np.random.default_rng(22112026)
    geom = np.random.default_rng(71831)
    t0 = time.perf_counter()
    z = np.concatenate([prior.sample(rng, min(128, 10000 - i), kind) for i in range(0, 10000, 128)])
    np.testing.assert_array_equal(z, expected)
    prior_time = time.perf_counter() - t0
    if args.match_noise:
        noises = [
            (
                geom.standard_normal((min(128, 10000 - i), 6)).astype(np.float32),
                geom.random((min(128, 10000 - i), 20, 3), dtype=np.float32),
            )
            for i in range(0, 10000, 128)
        ]
        input_k, input_f = map(np.concatenate, zip(*noises))
    arr = []
    for i in range(0, 10000, 24):
        a = torch.as_tensor(z[i : i + 24], dtype=torch.long, device="cuda")
        n = len(a)
        k_noise = (
            input_k[i : i + n]
            if args.match_noise
            else geom.standard_normal((n, 6)).astype(np.float32)
        )
        f_noise = (
            input_f[i : i + n] if args.match_noise else geom.random((n, 20, 3), dtype=np.float32)
        )
        k = torch.as_tensor(k_noise, device="cuda")
        f = torch.as_tensor(f_noise, device="cuda")
        with torch.inference_mode():
            k, f = draw(net, fit, a, k, f, 16 if method == "flow16" else 1)
            k = (k * state["scale"] + state["mean"]).cpu().numpy()
            f = f.cpu().numpy()
        L = cells(k)
        X = f @ L
        arr.append((k, L, f, X))
    if args.match_noise:
        out_k, out_L, out_f, out_X = map(np.concatenate, zip(*arr))
    total = time.perf_counter() - t0
    b = generate(method, kind, prior, 131, 1, "cuda", net, state, fit)
    out = dict(
        arm=arm,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        batch=24,
        draws=10000,
        elapsed_s=total,
        composition_seconds=prior_time,
        b1_ms=(1000 * np.array(b["batch_seconds"][3:])).tolist(),
        seed=22112026,
        geometry_seed=71831,
        nfe=16 if method == "flow16" else 1,
    )
    suffix = "-matched" if args.match_noise else ""
    if args.match_noise:
        dk = np.max(abs(out_k - expected_k), axis=1)
        delta = out_f - expected_f
        df = np.max(abs(delta - np.rint(delta)) * (expected > 0)[:, :, None], axis=(1, 2))
        np.savez_compressed(
            HERE / f"pools/{arm}-gpu-parity.npz",
            lattice_chart_max_difference=dk,
            fractional_max_difference=df,
        )
        out.update(
            matched_latents=True,
            chart_linf=float(dk.max()),
            site_periodic_linf=float(df.max()),
            above_tolerance=int(((dk > 1e-4) | (df > 1e-4)).sum()),
            tolerance=1e-4,
        )
    (HERE / f"pools/{arm}-gpu{suffix}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(arm, total, flush=True)
