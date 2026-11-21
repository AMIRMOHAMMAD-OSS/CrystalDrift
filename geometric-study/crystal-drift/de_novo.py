"""Generate new cell compositions and candidate geometries without a target structure."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from data import ELEMENTS, geometry_features
from model import CrystalGenerator
from sample import cif_text


HERE = Path(__file__).resolve().parent


def cell_key(z):
    first, second = np.unique(z[z > 0], return_counts=True)
    return int(first[0]), int(first[1]), int(second[0]), int(second[1])


def reduced_key(key):
    a, b, na, nb = key
    divisor = math.gcd(na, nb)
    return a, b, na // divisor, nb // divisor


def make_prior(z):
    """Half empirical joint law, half stoichiometry-conditioned independent substitution."""
    keys = Counter(cell_key(row) for row in z)
    stoichiometries = defaultdict(Counter)
    for (a,b,na,nb), frequency in keys.items():
        stoichiometries[na,nb][a,b] += frequency
    total = sum(keys.values())
    out = {k: n / (2 * total) for k,n in keys.items()}
    for (na,nb), group in stoichiometries.items():
        low, high = Counter(), Counter()
        for (a,b),n in group.items():
            low[a] += n
            high[b] += n
        proposals = {(a,b): n*m for a,n in low.items() for b,m in high.items() if a < b}
        mass = sum(group.values()) / (2 * total * sum(proposals.values()))
        for (a,b),v in proposals.items():
            key = (a,b,na,nb)
            out[key] = out.get(key, 0.) + mass*v
    support = sorted(out)
    weights = np.array([out[key] for key in support])
    assert np.isclose(weights.sum(), 1.0)
    return support, weights / weights.sum(), keys


def formula_text(key):
    a,b,na,nb = key
    return ELEMENTS[a] + (str(na) if na>1 else "") + ELEMENTS[b] + (str(nb) if nb>1 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--relax-prefix", type=int, default=32)
    parser.add_argument("--seed", type=int, default=71413)
    parser.add_argument("--out", type=Path, default=HERE/"de-novo")
    parser.add_argument("--checkpoint", type=Path, default=HERE/"drift-results/multi-exact/seed11.pt")
    args = parser.parse_args()
    if not 1 <= args.relax_prefix <= args.count:
        parser.error("The relaxation prefix must be between 1 and --count.")
    torch.set_num_threads(1)
    with np.load(HERE/"data/multi-train.npz") as d:
        support, weights, train_counts = make_prior(d["z"])
    rng = np.random.default_rng(args.seed)
    keys = [support[i] for i in rng.choice(len(support), args.count, p=weights)]
    z = np.zeros((args.count,8), dtype=np.int64)
    for i, (a,b,na,nb) in enumerate(keys):
        z[i,:na], z[i,na:na+nb] = a, b
    state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model = CrystalGenerator().eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        k, f = model(torch.from_numpy(z),torch.from_numpy(rng.normal(size=(len(z),6)).astype("float32")),
                     torch.from_numpy(rng.random((len(z),8,3)).astype("float32")))
        k = (k*state["scale"] + state["mean"]).numpy()
        f = f.numpy()
    ids = np.array([f"draw-{i+1:04d}" for i in range(args.count)])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"cifs").mkdir(exist_ok=True)
    props = geometry_features(k,f,z)
    train_reduced = {reduced_key(key) for key in train_counts}
    annotations = []
    for i, (target_id, key) in enumerate(zip(ids,keys)):
        chemical_formula = formula_text(key)
        entry = dict(id=target_id, formula=chemical_formula, exact_cell_in_training=key in train_counts,
                     reduced_formula_in_training=reduced_key(key) in train_reduced, volume_per_atom=props[i,0],
                     shortest_contact=float(np.min(props[i,1:])), predefined_relaxation=i<args.relax_prefix)
        annotations.append(entry)
        if i < args.relax_prefix:
            count = key[2]+key[3]
            (args.out/"cifs"/f"{target_id}.cif").write_text(
                cif_text(target_id,z[i,:count],k[i],f[i,:count]),encoding="ascii")
    with (args.out/"manifest.csv").open("w",encoding="utf8",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    arrays = dict(ids=ids,z=z,k=k,f=f)
    np.savez_compressed(args.out/"all-draws.npz",**arrays)
    prefix = args.out/"candidates.npz"
    np.savez_compressed(prefix,**{name: values[:args.relax_prefix] for name,values in arrays.items()})
    summary = dict(seed=args.seed,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   prefix_sha256=hashlib.sha256(prefix.read_bytes()).hexdigest(),count=len(z),
                   prefix=args.relax_prefix,training_cell_compositions=len(train_counts),prior_support=len(support),
                   unique_cell_compositions=len(set(keys)),unique_reduced_compositions=len({reduced_key(k) for k in keys}),
                   unseen_cell_composition_draws=sum(k not in train_counts for k in keys),
                   unseen_reduced_formula_draws=sum(reduced_key(k) not in train_reduced for k in keys),
                   with_sub_half_angstrom_contacts=sum(r["shortest_contact"] < 0.5 for r in annotations))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print(summary)


if __name__=="__main__":
    main()
