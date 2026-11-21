"""Apply the pinned Cartesian uFlowCSP EMA to the common hash-selected test cases."""
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
SCALED = HERE.parent
sys.path[:0] = [str(SCALED), str(SCALED.parent)]
from data import pack_gram
from experiment import select, sha
from uflowcsp_replay import RESULTS, SOURCE_COMMIT, stage_sources


def predict(cohort, model, transport, element, args, version_info):
    path = SCALED / (cohort + "-data") / "test.npz"
    with np.load(path, allow_pickle=False) as table:
        selected = select(table["ids"], 128)
        ids, z = table["ids"][selected], table["z"][selected]
    train_ids = set(json.loads((RESULTS / "train-ids.json").read_text()))
    test_ids = set(json.loads((RESULTS / "test-ids.json").read_text()))
    if cohort == "mp20":
        assert set(ids) <= test_ids and set(ids).isdisjoint(train_ids)
    def electronegativity(q):
        x = element.from_Z(int(q)).X
        return (float(x) if x is not None and np.isfinite(x) else 100 + q * .001, q)
    ordered = np.zeros_like(z, dtype=np.int64)
    for i, row in enumerate(z):
        occupied = row[row > 0]
        ordered[i, :len(occupied)] = sorted(occupied, key=electronegativity)
    directory = HERE / cohort
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "crosswalk.csv").open("w", newline="", encoding="utf8") as out:
        writer = csv.writer(out)
        writer.writerow(["id","input_z","official_sorted_z","in_official_train_ids","in_official_test_ids"])
        for material, original, official in zip(ids, z, ordered):
            writer.writerow([material, " ".join(map(str,original[original>0])),
                             " ".join(map(str,official[official>0])), material in train_ids, material in test_ids])
    all_k, all_f = [], []
    repeated = np.repeat(ordered, 5, axis=0)
    # Initialization on meta has no CPU RNG draw. Reset for each independent replay.
    torch.manual_seed(2026)
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(repeated), args.batch):
            active_types = repeated[i:i+args.batch]
            n, width = active_types.shape
            types = torch.zeros(n, 100, dtype=torch.long)
            types[:, :width] = torch.from_numpy(active_types)
            active = (types[:, :width] != 0).unsqueeze(-1)
            cond = dict(atom_types=types, mask=types!=0, spacegroup=torch.zeros(n,dtype=torch.long),
                        generate_atom_types=False)
            noise = transport.sample_prior(n, 100, 12, "cpu")
            output = transport.sample(model, noise, cond, num_steps=1)
            cell = (output[:,:width,3:].masked_fill(~active,0).sum(1)/active.sum(1)).reshape(n,3,3)
            frac = torch.remainder(output[:,:width,:3] @ torch.linalg.pinv(cell),1)
            restore = np.argsort(np.where(active_types==0,120,active_types),kind="stable",axis=1)
            np.testing.assert_array_equal(np.take_along_axis(active_types,restore,axis=1),z[np.arange(i,i+n)//5])
            all_f.append(np.take_along_axis(frac.masked_fill(~active,0).numpy(),restore[:,:,None],axis=1))
            lattice = cell.double().numpy()
            all_k.append(pack_gram(lattice @ lattice.swapaxes(-1,-2)))
            if i % (args.batch * 10) == 0:
                print(cohort,i+n,"/",len(repeated),flush=True)
    elapsed = time.perf_counter()-start
    k = np.concatenate(all_k).reshape(len(ids),5,6)
    f = np.concatenate(all_f).reshape(len(ids),5,20,3)
    assert np.isfinite(k).all() and np.isfinite(f).all() and (f*(z[:,None,:,None]==0)==0).all()
    product = directory / "uflow-seed2026-test.npz"
    np.savez_compressed(product,ids=ids,z=z,k=k,f=f)
    report = dict(cohort=cohort, target_count=len(ids), candidates=len(repeated), draws=5, seed=2026,
                  batch=args.batch, threads=args.threads, torch=torch.__version__, **version_info,
                  elapsed_seconds=elapsed, source_sha256=sha(path), prediction_sha256=sha(product),
                  weights_sha256=args.weights_digest, source_commit=SOURCE_COMMIT,
                  checkpoint="https://doi.org/10.6084/m9.figshare.33974422.v1",
                  selected_ids_in_official_train=len(set(ids)&train_ids),
                  selected_ids_in_official_test=len(set(ids)&test_ids), model_steps=1, padding=100,
                  active_sites_min=int((z>0).sum(1).min()),active_sites_max=int((z>0).sum(1).max()),
                  split_note="WBM and MP use different record identifiers. ID comparisons cannot exclude structural overlap.")
    (directory/"replay.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(cohort, "wrote", product, round(elapsed,1), "seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--site",type=Path,nargs="*",default=[])
    parser.add_argument("--cohort",choices=["mp20","wbm"],nargs="+",default=["mp20","wbm"])
    parser.add_argument("--batch",type=int,default=12)
    parser.add_argument("--threads",type=int,default=2)
    args = parser.parse_args()
    sys.path.extend(map(str,args.site))
    from pymatgen.core import Element
    import timm
    assert timm.__version__=="1.0.21"
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    weights_hash = hashlib.sha256()
    with args.weights.open("rb") as source:
        for block in iter(lambda:source.read(8*1024*1024),b""):
            weights_hash.update(block)
    args.weights_digest = weights_hash.hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        stage_sources(Path(tmp))
        from src.models.denoisers.atomic_features import build_atomic_feature_table
        from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import MeanFlowSiTRawFormulaAtomwiseSystem,build_sg_to_crystal_system
        from src.models.meanflow_transport import MeanFlowTransport
        with torch.device("meta"):
            model = MeanFlowSiTRawFormulaAtomwiseSystem(input_size=100, atom_type_vocab_size=120,
                        hidden_size=768,depth=12,num_heads=12,predict_atom_types=False,
                        use_spacegroup_conditioning=False,use_formula_embedding=True,
                        use_atomwise_features=True,use_crystal_system_embedding=True)
        with safe_open(args.weights,framework="pt",device="cpu") as weights:
            model.load_state_dict({k:weights.get_tensor(k) for k in weights.keys()},strict=True,assign=True)
        model.atomic_feature_table = build_atomic_feature_table(120)
        model.sg_to_crystal_system = build_sg_to_crystal_system(230)
        model.eval()
        transport = MeanFlowTransport(torus_coords=False,lattice_per_structure=False,proj_lattice=False,
                                      proj_coords_anchor=False,proj_coords_rotavg=False)
        info=dict(pymatgen=version("pymatgen"),timm=timm.__version__,parameters=sum(v.numel() for v in model.parameters()))
        for cohort in args.cohort:
            predict(cohort,model,transport,Element,args,info)


if __name__ == "__main__":
    main()
