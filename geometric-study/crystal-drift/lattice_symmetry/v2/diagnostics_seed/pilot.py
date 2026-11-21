"""Formula-only prototype seeds, crossed with a frozen one-evaluation angular generator.

Run `python pilot.py generate` in the release torch environment, then
`python pilot.py score` in the release pinned pymatgen environment.
"""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
CSP = HERE.parent / "csp"
ROOT = HERE.parents[2]
DATA = ROOT / "scaled/mp20-data"
SEED = 11
DRAWS = 20
ARMS = ("prior", "angular", "hybrid_prior", "hybrid_angular")


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("z", "ids", "k", "f")}


def pattern(z):
    return tuple(np.unique(z[z > 0], return_counts=True)[1])


class DonorBank:
    def __init__(self, train, expected_log_volume):
        self.train = train
        self.log_volume = expected_log_volume
        self.by_pattern = defaultdict(list)
        self.by_size = defaultdict(list)
        for i, row in enumerate(train["z"]):
            self.by_pattern[pattern(row)].append(i)
            self.by_size[np.count_nonzero(row)].append(i)

    def neighbors(self, z, rng, size):
        possible = self.by_pattern.get(pattern(z), self.by_size[np.count_nonzero(z)])
        rows = np.array(possible)
        distance = np.abs(self.train["z"][rows].astype(int) - z.astype(int)).sum(1)
        rows = rows[np.lexsort((rng.random(len(rows)), distance))]
        return rows[:size]


def generate(confirmation, test):
    sys.path.insert(0, str(CSP))
    from model import (Prior, checkpoint_path, count_design, digest, geometry, new_model,
                       pack_gram, torch)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    train = load(DATA / "train.npz")
    split = "test" if test else "val"
    val = load(DATA / (split + ".npz"))
    assert not set(train["ids"]) & set(val["ids"])
    prefix = "crytaldrift-onecall-transfer-confirm:" if test else ""
    order = sorted(range(len(val["ids"])), key=lambda i: hashlib.sha256((prefix+str(val["ids"][i])).encode()).digest())
    lo = 0 if test else 256 if confirmation else 128
    n = 384 if test else 128
    targets = np.array(order[lo:lo+n])
    out = HERE/"test384" if test else HERE/"confirmation" if confirmation else HERE
    out.mkdir(exist_ok=True)
    prior = Prior.load(CSP / "prior.npz")
    design, _ = count_design(train["z"])
    bank = DonorBank(train, design @ prior.coefficients)
    path = checkpoint_path("angular", SEED)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["prior_sha256"] == digest(CSP / "prior.npz")
    model = new_model("angular").eval()
    model.load_state_dict(checkpoint["model"])
    def apply_net(row, k, f):
        with torch.inference_mode():
            cell, sites = geometry(model, "angular", torch.as_tensor(row).long(),
                                   torch.as_tensor(k).float(), torch.as_tensor(f).float())
        cell = cell.double().numpy()
        return pack_gram(cell @ cell.transpose(0, 2, 1)), sites.numpy()

    all_k, all_f, all_donors, all_exact = [], [], [], []
    for raw_idx in targets:
        z = val["z"][raw_idx]
        zz = np.repeat(z[None], DRAWS, axis=0)
        base_rng = np.random.default_rng(np.random.SeedSequence([30711, SEED, int(raw_idx)]))
        donor_rng = np.random.default_rng(np.random.SeedSequence([30712, SEED, int(raw_idx)]))
        k, f = prior.draw(zz, base_rng)
        ka, fa = apply_net(zz, k, f)
        donor = bank.neighbors(z, donor_rng, DRAWS // 2)
        assert len(set(map(int, donor))) == len(donor)
        slots = np.arange(0, len(donor) * 2, 2)
        kh, fh = k.copy(), f.copy()
        kh[slots], fh[slots] = train["k"][donor], train["f"][donor]
        comp_design, _ = count_design(z[None])
        log_ratio = float((comp_design @ prior.coefficients)[0]) - bank.log_volume[donor]
        kh[slots, :3] += log_ratio[:, None] / 3
        kha, fha = ka.copy(), fa.copy()
        if len(donor):
            kha[slots], fha[slots] = apply_net(zz[slots], kh[slots], fh[slots])
        donor_ids = np.full(DRAWS, -1); donor_ids[slots] = donor
        cells = [k, ka, kh, kha]
        coords = [f, fa, fh, fha]
        if confirmation or test:
            kanchor = kha.copy()
            delta_logvol = (kh[slots,:3] - kha[slots,:3]).sum(1)
            kanchor[slots,:3] += delta_logvol[:,None]/3
            cells.append(kanchor)
            coords.append(fha)
        all_k.append(np.stack(cells))
        all_f.append(np.stack(coords))
        all_donors.append(donor_ids)
        all_exact.append(bool((train["z"][donor] == z).all(1).any()))
    np.savez_compressed(out/"proposals.npz", idx=targets, ids=val["ids"][targets], z=val["z"][targets],
                        k=np.asarray(all_k), f=np.asarray(all_f), donor_train_index=all_donors,
                        exact_train_composition=all_exact)
    setup = {"checkpoint":str(path.relative_to(ROOT)),"checkpoint_sha256":digest(path),
             "prior_sha256":digest(CSP/"prior.npz"), "training_sha256":digest(DATA/"train.npz"),
             "evaluation_split":split, "evaluation_split_sha256":digest(DATA/(split+".npz")),
             "selection":f"SHA256('{prefix}' + ID), sorted, [{lo},{lo+n})",
             "input":"exact cell composition only", "rng":[30711,30712], "weight_seed":SEED,
             "draws":DRAWS, "targets":len(targets), "arms":list(ARMS)+(["volume_anchor"] if confirmation or test else [])}
    if test:
        setup["protocol_sha256"] = digest(HERE/"TEST_PROTOCOL.md")
        assert setup["protocol_sha256"] == "5a67143255fd7bb6c4692da82651dac00c51cbf8d2f18bbe866ea46dee64e717"
    (out/"setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup))


def score(confirmation,test):
    assert version("pymatgen") == "2024.4.13"
    sys.path.insert(0, str(ROOT/"revision/full"))
    from match_full import candidate, target_context, unpack_gram
    out=HERE/"test384" if test else HERE/"confirmation" if confirmation else HERE
    arms=json.loads((out/"setup.json").read_text())["arms"]
    with np.load(out/"proposals.npz") as p:
        idx, ids, ks, fs, seen, donor = (p[k] for k in ("idx", "ids", "k", "f", "exact_train_composition", "donor_train_index"))
    val = load(DATA/("test.npz" if test else "val.npz"))
    assert np.array_equal(ids, val["ids"][idx])
    shape = ks.shape[:3]
    match = np.full(shape, np.nan); log_ratio = np.zeros(shape); fit = np.zeros(shape, bool)
    errors = np.zeros(shape, bool)
    for i, target in enumerate(idx):
        z = val["z"][target]; species = z[z > 0].astype(np.int64)
        context = target_context(val["k"][target], val["f"][target], z)
        memo = {}
        for j in range(len(arms)):
            k, f = ks[i,j], fs[i,j]
            gram = unpack_gram(k)
            log_ratio[i,j] = k[:,:3].sum(1,dtype=float) - val["k"][target,:3].sum(dtype=float)
            for b in range(DRAWS):
                key=(k[b].tobytes(),f[b].tobytes())
                if key not in memo:
                    try:
                        rms, strict, _ = candidate(context, k[b], f[b], species, gram[b], (i*811+j*47+b)%997 == 0)
                        memo[key] = rms, strict, False
                    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                        memo[key] = np.nan, False, True
                match[i,j,b], fit[i,j,b], errors[i,j,b] = memo[key]
    if test:
        for name, source in (("archived_angular",CSP/"test/angular-seed11/outcomes.npz"),
                             ("uflow",ROOT/"revision/full/external/mp20/complete-crosswalk-k20.npz")):
            with np.load(source) as data:
                table={str(mid):i for i,mid in enumerate(data["ids"])}
                take=np.array([table[str(mid)] for mid in ids])
                assert np.array_equal(data["ids"][take],ids)
                match=np.concatenate((match,data["rms"][take,None,:]),axis=1)
                fit=np.concatenate((fit,data["strict_fit"][take,None,:]),axis=1)
                log_ratio=np.concatenate((log_ratio,data["log_volume_ratio"][take,None,:]),axis=1)
                archive_errors=(data["error"][take,None,:]!=0 if "error" in data else np.zeros((len(ids),1,DRAWS),bool))
                errors=np.concatenate((errors,archive_errors),axis=1)
            arms.append(name)
    np.savez_compressed(out/"scores.npz", ids=ids, arms=arms, rms=match, max_site_fit=fit,
                        log_volume_ratio=log_ratio, errors=errors, known=seen)
    hits = np.isfinite(match) & ~errors
    native = hits & (np.abs(log_ratio) <= np.log(1.1))
    rows={}
    large=(val["z"][idx]>0).sum(1)>=9
    for name, mask in (("all", np.ones(len(idx),bool)),("exact_train_composition",seen),("unseen_train_composition",~seen),("nine_plus_atoms",large)):
        rows[name] = {"N":int(mask.sum())}
        for j, arm in enumerate(arms):
            rows[name][arm] = {str(n): {"match_count":int(hits[mask,j,:n].any(1).sum()),
                                        "native_count":int(native[mask,j,:n].any(1).sum())} for n in (1,5,20)}
    comparisons = {}
    pairs=[(3,1),(2,0),(3,2),(1,0)]
    if confirmation or test:
        pairs.extend(((4,3),(4,1),(4,2)))
    if test:
        pairs.extend(((4,5),(4,6)))
    for a,b in pairs:
        comparisons[arms[a]+" minus "+arms[b]] = {}
        for key,data in (("scaled",hits),("native",native)):
            gain=data[:,a].any(1).astype(int)-data[:,b].any(1)
            rng=np.random.default_rng(51039)
            boot=rng.choice(gain, (10000,len(idx)),replace=True).mean(1)
            comparisons[arms[a]+" minus "+arms[b]][key] = {"gains":int((gain>0).sum()),
                    "losses":int((gain<0).sum()),"percentage_points":100*gain.mean(),
                    "paired_percentile_95":(100*np.quantile(boot,[.025,.975])).tolist()}
    result={"counts":rows,"paired_at_20":comparisons,"matcher_exceptions":int(errors.sum()),
            "prototype_slots":int((donor>=0).sum()),"number_of_targets":len(idx),"pymatgen":version("pymatgen")}
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(rows["all"]),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("generate","score"))
    parser.add_argument("--confirmation",action="store_true",help="Separate validation IDs, adding the prespecified volume anchor.")
    parser.add_argument("--test",action="store_true",help="Locked 384-test-ID head-to-head after validation.")
    args=parser.parse_args()
    assert not (args.test and args.confirmation)
    (generate if args.action=="generate" else score)(args.confirmation,args.test)
