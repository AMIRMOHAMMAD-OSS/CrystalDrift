"""Compute protocol-A geometric and matching sensitivities on fixed, ordered full-MP-20 predictions."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("prior", "radial", "angular")
SEEDS = (11, 23, 37)
K = (1, 5, 20)
REPS = 5000
BOOT_SEED = 822037
FIELDS = ("rms", "strict_fit", "log_volume_ratio", "error")
NATIVE = "native_volume"
MAX_SITE = "maximum_site"
DIAG = ("absolute_log_volume", "log_shape_spectrum", "fractional_site_rms")
ENDPOINTS = [(m, k) for m in (MAX_SITE, NATIVE) for k in K] + [(m, None) for m in DIAG]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def geometry_diagnostic():
    source = ROOT / "expansion/training/diagnose.py"
    spec = importlib.util.spec_from_file_location("fixed_site_diagnostic", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.site_displacement


def shape_spectrum(packed):
    # Eigenvalues of 1/2 log(G), with the isotropic volume component removed.
    a = np.asarray(packed, dtype=np.float64)
    s = np.zeros(a.shape[:-1] + (3, 3))
    for i in range(3):
        s[..., i, i] = a[..., i]
    for j, (i, k) in enumerate(((0, 1), (0, 2), (1, 2)), start=3):
        s[..., i, k] = s[..., k, i] = a[..., j] / np.sqrt(2)
    d = np.linalg.eigvalsh(s)
    return d - d.mean(axis=-1, keepdims=True)


def load_scores(base, ids):
    joined = base / "outcomes.npz"
    shards = sorted((base / "scores").glob("*.npz"))
    if joined.exists():
        files = [(joined, 0, len(ids))]
    else:
        files = [(p, *map(int, p.stem.split("-"))) for p in shards]
        ranges = [(i, min(i + 64, len(ids))) for i in range(0, len(ids), 64)]
        if [(start, end) for _, start, end in files] != ranges:
            return None
    pieces = {field: [] for field in FIELDS}
    manifest = hashlib.sha256()
    for path, start, end in files:
        with np.load(path) as data:
            np.testing.assert_array_equal(data["ids"], ids[start:end])
            for field in FIELDS:
                x = data[field]
                assert x.shape == (end-start, 20)
                pieces[field].append(x)
        manifest.update(f"{path.name} {digest(path)}\n".encode())
    source = dict(kind="outcomes.npz" if joined.exists() and len(files)==1 else "64-target shards",
                  files=len(files), ordered_name_sha256_manifest_digest=manifest.hexdigest())
    return {k: np.concatenate(v) for k, v in pieces.items()}, source


def target_geometry(base, reference, spectrum, site_error):
    with np.load(base / "crosswalk.npz") as cross:
        np.testing.assert_array_equal(cross["ids"], reference["ids"])
        np.testing.assert_array_equal(cross["z"], reference["z"])
    kg = np.load(base / "k.npy", mmap_mode="r")
    fg = np.load(base / "f.npy", mmap_mode="r")
    assert kg.shape == (9046,20,6) and fg.shape == (9046,20,20,3)
    logv = kg[:, :, :3].sum(axis=-1,dtype=np.float64) - reference["k"][:, None, :3].sum(-1,dtype=np.float64)
    delta = shape_spectrum(kg) - spectrum[:, None, :]
    shape = np.linalg.norm(delta,axis=-1).mean(1)
    site = np.array([site_error(z, f, truth) for z, f, truth in zip(reference["z"], fg[:,0], reference["f"])])
    assert np.isfinite(site).all() and np.isfinite(shape).all() and np.isfinite(logv).all()
    return logv, shape, site


def accumulate(wait):
    cp = ROOT / "revision/full/mp20/cohort.npz"
    with np.load(cp) as d:
        ref = {k:d[k] for k in ("ids","k","z","f","release_index")}
    with np.load(ROOT/"scaled/mp20-data/test.npz") as d:
        assert len(set(d["ids"]))==9046
        np.testing.assert_array_equal(d["ids"][ref["release_index"]],ref["ids"])
    spec = shape_spectrum(ref["k"])
    site = geometry_diagnostic()
    x = np.zeros((len(ref["ids"]),len(ARMS),len(SEEDS),len(ENDPOINTS)))
    needed = {}
    receipts = {}
    for j, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            tag = f"{arm}-seed{seed}"
            base = HERE/"test"/tag
            meta = json.loads((base/"ready.json").read_text())
            assert (meta["targets"],meta["draws"],meta["seed"],meta["arm"])==(9046,20,seed,arm)
            for file, key in (("k.npy","cell_sha256"),("f.npy","sites_sha256")):
                assert digest(base/file)==meta[key]
            v, shape, site_values = target_geometry(base,ref,spec,site)
            x[:,j,s,-3:] = np.column_stack((np.abs(v).mean(1),shape,site_values))
            needed[tag] = (base,j,s,v)
            receipts[tag] = dict(raw=meta, crosswalk_sha256=digest(base/"crosswalk.npz"))
            print(tag,"geometry ready",flush=True)
    while needed:
        for tag, (base,j,s,logv) in list(needed.items()):
            outcome = load_scores(base,ref["ids"])
            if outcome is None:continue
            scores, stamp = outcome
            np.testing.assert_allclose(scores["log_volume_ratio"],logv,atol=4e-7,rtol=1e-7)
            ok = scores["error"]==0
            accepted = ok & np.isfinite(scores["rms"])
            strict = ok & scores["strict_fit"]
            qualified = accepted & (np.abs(scores["log_volume_ratio"])<=np.log(1.1))
            assert not (strict & ~accepted).any()
            for b,k in enumerate(K):
                x[:,j,s,b] = strict[:,:k].any(1)
                x[:,j,s,len(K)+b] = qualified[:,:k].any(1)
            receipts[tag]["score_source"] = stamp
            receipts[tag]["exceptions"] = int((~ok).sum())
            del needed[tag]
            print(tag,"ordered scores ready",flush=True)
        if needed:
            if not wait:raise FileNotFoundError("Incomplete score sets: "+", ".join(needed))
            time.sleep(30)
    return x, receipts, cp


def summarize(x):
    # All method, endpoint and seed blocks share each bootstrap's target weights.
    n = len(x)
    actual = x.mean(axis=0)
    by_target = x.mean(axis=2).reshape(n,-1)
    rng = np.random.default_rng(BOOT_SEED)
    reps = []
    for begin in range(0,REPS,32):
        size = min(32,REPS-begin)
        w = rng.multinomial(n,np.full(n,1/n),size=size)
        reps.append((w @ by_target / n).reshape(size,len(ARMS),len(ENDPOINTS)))
    boots = np.concatenate(reps)
    values = actual.mean(axis=1)
    out = []
    for b, (metric,k) in enumerate(ENDPOINTS):
        for j, arm in enumerate(ARMS):
            rows = [(arm,"",values[j,b],boots[:,j,b])]
            if j < ARMS.index("angular"):
                h = ARMS.index("angular")
                rows.append(("angular",arm,values[h,b]-values[j,b],boots[:,h,b]-boots[:,j,b]))
            for method,comp,value,boot in rows:
                lo,hi = np.quantile(boot,[.025,.975])
                out.append(dict(metric=metric,k=k,method=method,minus=comp,seed="mean",targets=n,
                                hits="",value=value,low=lo,high=hi))
            for s, seed in enumerate(SEEDS):
                value = actual[j,s,b]
                out.append(dict(metric=metric,k=k,method=arm,minus="",seed=seed,targets=n,
                                hits=int(x[:,j,s,b].sum()) if k else "",value=value,low="",high=""))
    return out


def latex(rows):
    cases = {(r["metric"],r["k"],r["method"],r["minus"]):r for r in rows if r["seed"]=="mean"}
    def val(m,k,arm,comp=""):
        return cases[m,k,arm,comp]
    def paired(d):
        return f"{100*d['value']:+.2f} [{100*d['low']:+.2f},{100*d['high']:+.2f}]"
    body = [r"\section{Sensitivity to maximum site displacement and native volume}\label{app:csp-sensitivity}",
        r"We rescore the ordered raw proposals under two stricter criteria on the same 9,046 official composition-conditioned targets. The maximum-site criterion uses the public \texttt{StructureMatcher.fit} acceptance under the prescribed primitive-cell reduction, volume scaling, exact species, lattice tolerance 0.3, site tolerance 0.5 and angle tolerance $10^\circ$. It requires an accepted mapping whose largest normalized displacement passes the site tolerance. The volume-qualified criterion requires the standard RMS match and $|\log(V_{\rm draw}/V_{\rm ref})|\leq\log(1.10)$ on the same draw, using the original cell volumes before matcher scaling. Their compositions, and hence site counts, are identical.",
        r"\begin{table}[!htbp]\centering\scriptsize",
        r"\caption{Raw recovery (\%) under stricter structure criteria. Differences are percentage points, with 95\% paired target-bootstrap intervals. Each target carries the same three training or prior seeds 11, 23 and 37 and up to twenty ordered draws.}\label{tab:csp-sensitivity}",
        r"\setlength{\tabcolsep}{3.3pt}\begin{tabular}{lrrrrrr}\hline",
        r"Criterion & $k$ & Prior & Radial & Angular & Angular $-$ prior & Angular $-$ radial\\\hline"]
    for m, label in ((MAX_SITE,"Maximum site"),(NATIVE,"Native volume")):
        for k in K:
            a = [label if k==1 else "",str(k)]
            a.extend(f"{100*val(m,k,arm)['value']:.2f}" for arm in ARMS)
            a.extend(paired(val(m,k,"angular",arm)) for arm in ("prior","radial"))
            body.append(" & ".join(a)+r"\\")
    body.append(r"\hline\end{tabular}\end{table}")
    fmt=lambda metric: ", ".join(f"{val(metric,None,a)['value']:.3f}" for a in ARMS)
    body.append(r"Across twenty raw draws per target and seed, the mean absolute log-volume errors for the prior, radial and angular maps were "+fmt(DIAG[0])+", respectively. The corresponding mean shape errors were "+fmt(DIAG[1])+r". Shape error is the Euclidean distance between sorted eigenvalues of $\frac12\log(LL^\top)$ after subtraction of their mean, computed in each deposited basis. This removes uniform scale and Cartesian orientation. A general unimodular rebasing can change this diagnostic.")
    body.append(r"On the first draw of every target and seed, the mean fractional motif RMS errors in the same order were "+fmt(DIAG[2])+r". For this diagnostic we use the deposited basis, align a center of the rarest species, alternate same-species assignments with a common origin refinement for up to four rounds, and retain the best feasible displacement. Each displacement is wrapped componentwise. This diagnostic is dimensionless and depends on the basis and alignment search. It includes targets with no matcher hit. The archive records the individual-seed results, absolute intervals and paired intervals for all three geometry diagnostics.")
    return "\n".join(body)+"\n"


def main(wait):
    x, receipts, cp = accumulate(wait)
    rows = summarize(x)
    with (HERE/"geometry-sensitivity.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    paper=HERE.parent/"paper"/"sensitivity.tex"
    paper.write_text(latex(rows),encoding="utf8")
    audit=dict(targets=9046,draws=20,seeds=SEEDS,methods=ARMS,
        matching=dict(pymatgen="2024.4.13",ltol=.3,stol=.5,angle_tol_deg=10,primitive_cell=True,scale=True,
                      native_volume="abs(log(V_draw / V_ref)) <= log(1.10) AND finite accepted RMS on same draw",
                      maximum_site="public StructureMatcher.fit-equivalent saved strict_fit; matcher errors count as misses"),
        geometry=dict(volume="mean absolute per-atom log-volume ratio, 20 draws; identical composition in each pair",
                      shape="20 draws; centered eigenvalues of the deposited-basis half-log native Gram, Euclidean error",
                      site="first draw on every target, deposited-basis wrapped fractional RMS after feasible assignment"),
        uncertainty=dict(unit="whole target IDs, paired across seeds/arms and all prefixes",replicates=REPS,
                         seed=BOOT_SEED,interval="2.5/97.5 percentile",point="equal arithmetic mean of the 3 seeds"),
        sha256=dict(official_crosswalk=digest(cp),site_procedure=digest(ROOT/"expansion/training/diagnose.py"),
                    csv=digest(HERE/"geometry-sensitivity.csv"),paper=digest(paper),script=digest(Path(__file__))),
        scored_draws=9046*20*len(ARMS)*len(SEEDS),receipts=receipts)
    (HERE/"geometry-sensitivity-settings.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--wait",action="store_true",help="Wait for outstanding ordered-score shards after reading completed cells.")
    main(p.parse_args().wait)

