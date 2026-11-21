"""One-pass kernel ablation on the previously frozen 384 by five formula-only panel."""
import hashlib
import json
import sys
import copy
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent / "diagnostics_template_prediction"
sys.path.insert(0, str(TRANSFER))
from confirm import CONF, K, N
from experiment import (ROOT, V2, SETTINGS, Structure, StructureMatcher, Prior, apply,
                        model_and_hash, native, np, records, torch, unpack_gram)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate():
    prior = Prior(records())
    models = {}
    checkpoints = {}
    for method in ("angular-C", "orbit-C"):
        models[method], checkpoints[method] = model_and_hash(method)
    path = V2 / "wyckoff" / "training-orbit" / "angular-template-256.pt"
    models["orbit-C-256"] = copy.deepcopy(models["orbit-C"])
    models["orbit-C-256"].load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    checkpoints["orbit-C-256"] = {"checkpoint":str(path.relative_to(ROOT)), "sha256": digest(path), "nfe":1}

    raw_choices = json.loads((CONF / "symbolic-choices.json").read_text())
    with np.load(CONF / "predictions.npz") as f:
        ids = f["ids"]
        old_l, old_f, old_z = (f[s] for s in ("L","F","Z"))
    assert len(ids) == N and len(raw_choices) == N*K
    with np.load(ROOT / "scaled/mp20-data/test.npz") as table:
        indices = {str(mid): i for i, mid in enumerate(table["ids"])}
        order = np.array([indices[str(mid)] for mid in ids])
        zs = table["z"][order]
    L = np.full_like(old_l, np.nan)
    F = np.full_like(old_f, np.nan)
    Z = np.zeros_like(old_z)
    errors = []
    for i, mid in enumerate(ids):
        choices = raw_choices[K*i:K*(i+1)]
        assert [(r["id"], r["draw"]) for r in choices] == [(mid, k+1) for k in range(K)]
        rng = np.random.default_rng(np.random.SeedSequence([70693,int(order[i])]))
        seeds = [prior.draw(r["group"], r["pattern"], rng) for r in choices]
        outputs = [seeds] + [apply(models[key], seeds) for key in ("angular-C", "orbit-C", "orbit-C-256")]
        for j, row in enumerate(choices):
            for a, pool in enumerate(outputs):
                try:
                    obj = native(pool[j], row["group"], zs[i])
                except ValueError as e:
                    if a < 2:
                        assert not old_z[i,j,a].any()
                    else:
                        errors.append({"id":str(mid), "draw":j+1, "arm":a-2,"error":str(e)})
                    continue
                if a < 2:
                    mask = old_z[i,j,a] > 0
                    np.testing.assert_array_equal(old_z[i,j,a,mask], obj.atomic_numbers)
                    np.testing.assert_allclose(old_l[i,j,a], obj.lattice.matrix, rtol=1e-6, atol=1e-6)
                    res = old_f[i,j,a,mask] - obj.frac_coords
                    assert np.max(np.abs(res-np.rint(res))) < 2e-6
                else:
                    a -= 2
                    L[i,j,a] = obj.lattice.matrix
                    F[i,j,a,:len(obj)] = obj.frac_coords
                    Z[i,j,a,:len(obj)] = obj.atomic_numbers
    out = HERE / "transfer-orbit-predictions.npz"
    np.savez_compressed(out, ids=ids, order=order, L=L, F=F, Z=Z)
    return dict(checkpoints=checkpoints, conversion_failures=errors,
                source_choices_sha256=digest(CONF/"symbolic-choices.json"),
                source_predictions_sha256=digest(CONF/"predictions.npz"),
                generated_sha256=digest(out))


def score():
    with np.load(HERE / "transfer-orbit-predictions.npz") as store:
        ids, order, L, F, Z = (store[s] for s in ("ids","order","L","F","Z"))
    with np.load(ROOT / "scaled/mp20-data/test.npz") as store:
        np.testing.assert_array_equal(store["ids"][order], ids)
        tl = np.linalg.cholesky(unpack_gram(store["k"][order]))
        tz, tf = store["z"][order], store["f"][order]
    matcher = StructureMatcher(**SETTINGS)
    result = np.full((N,K,2),np.nan)
    vol = np.full_like(result, np.nan)
    for i in range(N):
        mask = tz[i] > 0
        target = Structure(tl[i], tz[i,mask], tf[i,mask])
        for j in range(K):
            for arm in range(2):
                valid = Z[i,j,arm] > 0
                if not valid.any():
                    continue
                candidate = Structure(L[i,j,arm],Z[i,j,arm,valid],F[i,j,arm,valid])
                outcome = matcher.get_rms_dist(candidate, target)
                if outcome is not None:
                    result[i,j,arm] = outcome[0]
                vol[i,j,arm] = np.log(candidate.volume/target.volume)
    with np.load(CONF / "scores.npz") as scores:
        np.testing.assert_array_equal(scores["ids"],ids)
        raw = {key:scores[key] for key in ("symbolic_prior","symbolic_angular")}
        volumes = scores["log_volume_ratio"]
        seen = scores["seen"]
    raw["orbit_C_selected"], raw["orbit_C_equal_updates"] = result[:,:,0], result[:,:,1]
    v = dict(symbolic_prior=volumes[:,:,0], symbolic_angular=volumes[:,:,1],
             orbit_C_selected=vol[:,:,0], orbit_C_equal_updates=vol[:,:,1])
    np.savez_compressed(HERE / "transfer-orbit-scores.npz",ids=ids,**raw,logvolumes_new=vol)
    summary = {}
    for k in (1,5):
        hits = {name:np.isfinite(a[:,:k]).any(1) for name,a in raw.items()}
        data = {}
        for name,h in hits.items():
            matched = np.where(np.isnan(raw[name][:,:k]),np.inf,raw[name][:,:k]).min(1)
            data[name] = dict(hits=int(h.sum()), mr=float(h.mean()),
                              mean_matched_rms=float(np.mean(matched[np.isfinite(matched)])),
                              with_native_volume=int((np.isfinite(raw[name][:,:k]) &
                                  (np.abs(v[name][:,:k]) <= np.log(1.1))).any(1).sum()),
                              seen_hits=int(h[seen].sum()), unseen_hits=int(h[~seen].sum()))
        for name in ("orbit_C_selected","orbit_C_equal_updates"):
            for control in ("symbolic_angular","symbolic_prior"):
                x,y = hits[name],hits[control]
                delta = x.astype(float)-y
                rng = np.random.default_rng(111589)
                lo,hi = np.quantile(delta[rng.integers(0,N,(5000,N))].mean(axis=1),[.025,.975])
                data[f"{name}_minus_{control}"] = dict(difference=float(delta.mean()),
                    ci95=[float(lo),float(hi)],only_orbit=int((x&~y).sum()),only_control=int((~x&y).sum()))
        summary[k] = data
    return summary


def native_difference():
    with np.load(HERE / "transfer-orbit-scores.npz") as s, np.load(CONF / "scores.npz") as old:
        a = np.isfinite(s["symbolic_angular"]) & (np.abs(old["log_volume_ratio"][:,:,1]) <= np.log(1.1))
        b = np.isfinite(s["orbit_C_selected"]) & (np.abs(s["logvolumes_new"][:,:,0]) <= np.log(1.1))
        medians = {"ordinary":float(np.nanmedian(old["log_volume_ratio"][:,:,1])),
                   "orbit_selected":float(np.nanmedian(s["logvolumes_new"][:,:,0]))}
    intervals = {}
    for k in (1,5):
        x, y = b[:,:k].any(1), a[:,:k].any(1)
        d = x.astype(float)-y
        rng = np.random.default_rng(111589)
        bounds = np.quantile(d[rng.integers(0,len(x),(5000,len(x)))].mean(1),[.025,.975])
        intervals[k] = dict(difference=float(d.mean()), ci95=bounds.tolist(), orbit_only=int((x&~y).sum()),
                            ordinary_only=int((y&~x).sum()))
    return {"median_signed_candidate_log_volume_ratio":medians,
            "volume_qualified_recovery_selected_minus_ordinary":intervals}


if __name__ == "__main__":
    torch.set_num_threads(1)
    receipt = generate()
    print("Predictions frozen. Scoring both orbit checkpoints.",flush=True)
    body = dict(targets=N, draws=K, split="official test, existing prespecified confirmation panel",
                order="proposal first, reference second", target_geometry_access="scoring only",
                compared_arms=["orbit_C_selected_512", "orbit_C_equal_updates_256"],
                **receipt, results=score(), native_difference=native_difference())
    (HERE/"transfer-orbit-summary.json").write_text(json.dumps(body,indent=2)+"\n", encoding="utf8")
    print(json.dumps(body["results"][5],indent=2),flush=True)
