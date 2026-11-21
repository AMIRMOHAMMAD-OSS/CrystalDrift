"""Fit the one-pass crystal generator from the public processed MP-20 arrays."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from model import CrystalGenerator, chemistry, crystal_features, drifting_field, within_formula_field


HERE = Path(__file__).resolve().parent


def training_data(data_folder, method):
    prefix = "multi-" if method == "multi" else ""
    with np.load(data_folder / (prefix + "train.npz")) as d:
        z = torch.from_numpy(d["z"].astype("int64"))
        k = torch.from_numpy(d["k"].astype("float32"))
        f = torch.from_numpy(d["f"].astype("float32"))
    if f.ndim == 2:
        f = torch.stack((torch.zeros_like(f), f), dim=1)
    mean, scale = k.mean(0), k.std(0, unbiased=False).clamp_min(1e-5)
    real = crystal_features((k - mean) / scale, f, z, method)
    return z, real, chemistry(z, k.dtype), mean, scale


def noise(z):
    return torch.randn(len(z), 6), torch.rand(*z.shape, 3)


def train(data_folder, method, seed, updates, batch_size):
    torch.manual_seed(seed)
    rep = method.split("-")[0]
    exact = method.endswith("-exact")
    z, real, chem, mean, scale = training_data(data_folder, rep)
    if exact:
        if batch_size % 4:
            raise ValueError("Within-formula batches use four queries per cell composition.")
        _, inverse = torch.unique(z, dim=0, return_inverse=True)
        members = [torch.where(inverse == i)[0] for i in range(int(inverse.max()) + 1)]
        counts = torch.tensor([len(group) for group in members])
        index_table = torch.nn.utils.rnn.pad_sequence(members, batch_first=True)
    model = CrystalGenerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    log = []
    start = time.perf_counter()
    for step in range(updates):
        if exact:
            group = inverse[torch.randint(len(z), (batch_size // 4,))]
            qi = index_table[group, 0].repeat_interleave(4)
            local = (torch.rand(len(group), 4) * counts[group, None]).long()
            ri = index_table[group[:,None], local].reshape(-1)
        else:
            qi, ri = torch.randint(len(z), (2, batch_size))
        query_z, ref_z = z[qi], z[ri]
        with torch.no_grad():
            k_ref, f_ref = model(ref_z, *noise(ref_z))
            negative = crystal_features(k_ref, f_ref, ref_z, rep)
        k, f = model(query_z, *noise(query_z))
        h = crystal_features(k, f, query_z, rep)
        if exact:
            shape = (len(group), 4, h.shape[-1])
            direction = within_formula_field(h.detach().reshape(shape),real[ri].reshape(shape),
                                             negative.reshape(shape)).reshape_as(h)
        else:
            direction = drifting_field(h.detach(), real[ri], negative, chem[qi], chem[ri])
        target = h.detach() + direction
        loss = 0.5 * (h - target).square().sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            if not torch.isfinite(norm):
                raise ValueError(f"Nonfinite gradient on update {step + 1}")
            log.append(dict(update=step+1, loss=loss.item(), gradient_norm=norm.item()))
    return model.eval(), mean, scale, time.perf_counter() - start, log


def evaluation_source(z, draws, seed):
    rng = np.random.default_rng(seed + 100000)
    cell = rng.normal(size=(len(z), draws, 6)).astype("float32")
    if z.shape[1] == 2:
        # Common AB source with the flow comparator, using the first site as origin.
        relative = rng.random((len(z), draws, 3)).astype("float32")
        f = np.stack((np.zeros_like(relative), relative), axis=-2)
    else:
        f = rng.random((len(z), draws, z.shape[1], 3)).astype("float32")
    return (torch.from_numpy(np.repeat(z.astype("int64"), draws, axis=0)),
            torch.from_numpy(cell.reshape(-1, 6)), torch.from_numpy(f.reshape(-1, z.shape[1], 3)))


@torch.inference_mode()
def generate(model, source, mean, scale):
    # Chunk evaluation to keep peak CPU memory bounded for larger candidate pools.
    cells, positions = [], []
    for i in range(0, len(source[0]), 2048):
        z, u, v = (x[i:i+2048] for x in source)
        k, f = model(z, u, v)
        cells.append(k * scale + mean)
        if z.shape[1] == 2:
            f = (f[:,1] - f[:,0]).remainder(1)
        positions.append(f)
    return torch.cat(cells), torch.cat(positions)


def run(method, args):
    out = args.output / method
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data / ("multi-test.npz" if method.startswith("multi") else "test.npz")) as d:
        z, ids = d["z"], d["ids"]
    for seed in args.seeds:
        model, mean, scale, elapsed, log = train(args.data, method, seed, args.updates, args.batch_size)
        source = evaluation_source(z, args.draws, seed)
        generate(model, source, mean, scale)
        timings, single = [], []
        for _ in range(5):
            start = time.perf_counter()
            k, f = generate(model, source, mean, scale)
            timings.append(time.perf_counter() - start)
        for index in np.linspace(0, len(source[0])-1, 32, dtype=int):
            one = tuple(x[index:index+1] for x in source)
            start = time.perf_counter()
            generate(model, one, mean, scale)
            single.append(time.perf_counter() - start)
        assert torch.isfinite(k).all() and torch.isfinite(f).all() and ((f >= 0) & (f < 1)).all()
        np.savez_compressed(out / f"seed{seed}.npz", ids=ids, z=z,
                            k=k.numpy().reshape(len(z), args.draws, 6),
                            f=f.numpy().reshape(len(z), args.draws, *f.shape[1:]))
        torch.save(dict(model=model.state_dict(), mean=mean, scale=scale, method=method, seed=seed),
                   out / f"seed{seed}.pt")
        stats = dict(seed=seed, updates=args.updates, batch=args.batch_size, draws=args.draws,
                     train_seconds=elapsed, parameters=sum(p.numel() for p in model.parameters()),
                     sampling_nfe=1, samples=len(source[0]), batch_seconds=timings,
                     microseconds_per_crystal=statistics.median(timings)*1e6/len(source[0]),
                     median_single_crystal_microseconds=statistics.median(single)*1e6,
                     torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                     threads=args.threads, training=log)
        (out / f"seed{seed}.json").write_text(json.dumps(stats, indent=2)+"\n", encoding="utf8")
        print(method, seed, "training seconds", round(elapsed,2), flush=True)


def replay_validation(method, args):
    out = args.validation_output / method
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data / ("multi-val.npz" if method.startswith("multi") else "val.npz")) as split:
        z, ids = split["z"], split["ids"]
    for seed in args.seeds:
        checkpoint = torch.load(args.output / method / f"seed{seed}.pt", map_location="cpu", weights_only=True)
        model = CrystalGenerator().eval()
        model.load_state_dict(checkpoint["model"])
        source = evaluation_source(z, args.draws, seed)
        k,f = generate(model,source,checkpoint["mean"],checkpoint["scale"])
        np.savez_compressed(out / f"seed{seed}.npz", ids=ids,z=z,k=k.numpy().reshape(len(z),args.draws,6),
                            f=f.numpy().reshape(len(z),args.draws,*f.shape[1:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("ab","raw","multi","ab-exact","multi-exact"),
                        default=["ab","raw","multi","ab-exact","multi-exact"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,23,37])
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "drift-results")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-output", type=Path, default=HERE / "drift-validation")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for method in args.methods:
        if args.validation_only:
            replay_validation(method,args)
        else:
            run(method, args)
