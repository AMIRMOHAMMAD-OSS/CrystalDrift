"""CPU formula-to-array timings for the baseline and prototype-seeded one-call samplers."""
import csv
import json
import platform
import sys
import time

from pilot import CSP, DATA, DRAWS, DonorBank, HERE, SEED, load, np

sys.path.insert(0, str(CSP))
from model import Prior, checkpoint_path, count_design, digest, geometry, new_model, pack_gram, torch, unpack_gram


def now_ms():
    return time.perf_counter_ns() * 1e-6


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = new_model("angular").eval()
    checkpoint = checkpoint_path("angular", SEED)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
    prior = Prior.load(CSP/"prior.npz")
    train = load(DATA/"train.npz")
    tic = now_ms()
    train_design, _ = count_design(train["z"])
    bank = DonorBank(train, train_design @ prior.coefficients)
    index_ms = now_ms()-tic
    with np.load(HERE/"test384/proposals.npz") as take:
        ids, z, raw = take["ids"], take["z"], take["idx"]
    protocols = ("angular", "volume_anchor", "hybrid_prior")

    @torch.inference_mode()
    def run(row, ix, n, arm):
        begin = now_ms()
        base_rng = np.random.default_rng(np.random.SeedSequence([30711, SEED, int(ix)]))
        rows = np.repeat(row[None], n, axis=0)
        k, f = prior.draw(rows, base_rng)
        start_lookup = now_ms()
        slots = np.zeros(n, dtype=bool)
        if arm != "angular":
            donor_rng = np.random.default_rng(np.random.SeedSequence([30712, SEED, int(ix)]))
            donor = bank.neighbors(row, donor_rng, (n+1)//2)
            slots[0:2*len(donor):2] = True
            k[slots], f[slots] = train["k"][donor], train["f"][donor]
            x, _ = count_design(row[None])
            k[slots,:3] += ((x @ prior.coefficients)[0]-bank.log_volume[donor])[:,None]/3
        lookup_ms = now_ms()-start_lookup
        if arm != "hybrid_prior":
            output_k = np.empty((n,6), np.float64)
            output_f = np.empty_like(f)
            for b in range(n):
                cell, sites = geometry(model, "angular", torch.as_tensor(rows[b:b+1]).long(),
                                       torch.as_tensor(k[b:b+1]).float(), torch.as_tensor(f[b:b+1]).float())
                cell = cell.double().numpy()
                packed = pack_gram(cell @ cell.transpose(0,2,1))
                if slots[b]:
                    packed[:,:3] += (k[b,:3].sum(dtype=float)-packed[:,:3].sum())/3
                output_k[b], output_f[b] = packed[0], sites.numpy()[0]
        else:
            output_k, output_f = k, f
        lattice = np.linalg.cholesky(unpack_gram(output_k))
        wrapped = np.where(rows[:,:,None] > 0, np.remainder(output_f,1), 0.).astype(np.float64)
        cartesian = wrapped @ lattice
        elapsed = now_ms()-begin
        if not np.isfinite(lattice).all() or not np.isfinite(cartesian).all():
            raise ValueError("Invalid output arrays.")
        return elapsed, lookup_ms, output_k, output_f

    for j in range(48,54):
        for arm in protocols:
            run(z[j], raw[j], 2, arm)
    # Validate the single-candidate inference route against previously scored K20 arrays.
    with np.load(HERE/"test384/proposals.npz") as p:
        for j in (0,1,2):
            for name, path_arm in (("angular",1),("volume_anchor",4),("hybrid_prior",2)):
                _,_,k,f=run(z[j],raw[j],DRAWS,name)
                np.testing.assert_allclose(k,p["k"][j,path_arm],rtol=2e-5,atol=2e-5)
                d=f-p["f"][j,path_arm]
                np.testing.assert_allclose(d-d.round(),0,atol=2e-5)
    measurements=[]
    for repeat in range(2):
        for i in range(48):
            order = protocols[(i+repeat)%3:] + protocols[:(i+repeat)%3]
            for n in ((1,DRAWS) if (i+repeat)%2 else (DRAWS,1)):
                for arm in order:
                    elapsed,lookup,_,_=run(z[i], raw[i], n, arm)
                    measurements.append(dict(material_id=ids[i],repeat=repeat,k=n,arm=arm,
                                              n_atoms=int(np.count_nonzero(z[i])),elapsed_ms=elapsed,lookup_ms=lookup))
    folder = HERE/"test384"
    with (folder/"cpu-clock.csv").open("w",newline="") as stream:
        out=csv.DictWriter(stream,fieldnames=measurements[0].keys())
        out.writeheader(); out.writerows(measurements)
    result={"targets":48,"repetitions":2,"torch_version":torch.__version__,"threads":torch.get_num_threads(),
            "interop_threads":torch.get_num_interop_threads(),"platform":platform.platform(),
            "index_build_ms_excludes_disk_load":index_ms,"training_rows":len(train["z"]),
            "checkpoint_sha256":digest(checkpoint),"nfe":dict(angular=1,volume_anchor=1,hybrid_prior=0),
            "scope":"Wall time from supplied cell composition through RNG, fitted prior, optional train-only donor lookup, volume adjustment, serial CPU batch-1 neural calls and float64 host lattice, wrapped fractional and Cartesian arrays. Index/model loaded once. One fresh donor ranking per formula episode, shared over its candidates. First 48 declared test-subset IDs, six disjoint warmup IDs; interleaved order; no CPU isolation or dedicated machine. Excludes model/index loading, CIF writing, scoring or relaxation."}
    for n in (1,DRAWS):
        result[str(n)]={}
        for arm in protocols:
            rows=[r for r in measurements if r["arm"]==arm and r["k"]==n]
            elapsed=np.array([r["elapsed_ms"] for r in rows])
            per_id=np.array([np.median([r["elapsed_ms"] for r in rows if r["material_id"]==mid]) for mid in ids[:48]])
            result[str(n)][arm]={"median_target_ms":float(np.median(per_id)),"median_ms_per_draw":float(np.median(per_id)/n),
                         "p25_p75_target_ms":np.percentile(per_id,[25,75]).tolist(),
                         "total_draws_per_s":float(len(rows)*n*1000/elapsed.sum()),
                         "median_prototype_lookup_and_adjustment_ms":float(np.median([r["lookup_ms"] for r in rows]))}
    (folder/"cpu-clock.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
