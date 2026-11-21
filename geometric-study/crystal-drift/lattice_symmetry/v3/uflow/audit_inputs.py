"""Verify eligible IDs, raw MP-20 formulas, static labels and every optimizer exposure."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    protocol=json.loads((HERE.parent/"protocol/splits.json").read_text())
    eligible=protocol["train_ids"]
    official=set(json.loads((ROOT/"uflowcsp-results/train-ids.json").read_text()))
    assert set(eligible)<=official
    assert set(eligible).isdisjoint(set(protocol["family_ids"])|set(protocol["shared_ids"])|
                                   set(protocol["dev_ids"])|set(protocol["confirm_ids"])|set(protocol["test_ids"]))
    cache={str(k):[] for k in ("ids","z","sg","original_group")}
    for path in sorted((HERE/"train-cache").glob("*.npz")):
        with np.load(path) as data:
            z=data["z"]
            assert np.allclose(data["x"][z==0],0)
            for k in cache:
                cache[k].append(data[k])
    c={k:np.concatenate(values) for k,values in cache.items()}
    assert list(c["ids"])==eligible
    with np.load(ROOT/"scaled/mp20-data/train.npz") as archive:
        lookup={str(mid):i for i,mid in enumerate(archive["ids"])}
        original_z=archive["z"][[lookup[mid] for mid in eligible]]
    np.testing.assert_array_equal(np.sort(original_z,axis=1),np.sort(c["z"],axis=1))
    # The model only embeds one of seven crystal systems, plus the null class.
    system=lambda n: np.searchsorted([2,15,74,142,167,194,230],n,side="left")+1
    report=dict(eligible=len(eligible),atoms=int(np.sum(c["z"]>0)),coarse_class_counts=np.bincount(system(c["sg"]),minlength=8)[1:].tolist(),
             csv_vs_fitted_group_disagreements=int(np.sum(c["sg"]!=c["original_group"])),
             csv_vs_fitted_system_disagreements=int(np.sum(system(c["sg"])!=system(c["original_group"]))),
             duplicate_ids=len(eligible)-len(set(eligible)),excluded_challenge_ids_seen=0,formula_mismatches=0,
             native_training_set_size=len(official),input_prep="pymatgen Niggli CifParser; author's symmetry ordering at 0.1 Angstrom; auxiliary group from spglib",
             crystal_system_source="Training-only; spglib 0.1 Angstrom. Authors use PyXtal after spglib refinement.",
             native_structured_prior=False,
             comparator_pretrained=False,external_chemistry="four fixed elemental columns from frozen authors' table",
             fits={})
    inventory=json.loads((HERE.parent/'models/cache/inventory.json').read_text())
    schedule=np.array(inventory['order']);rng=np.random.default_rng(11)
    common=np.concatenate([x[rng.permutation(len(x))] for x in np.array_split(schedule,np.arange(128,len(schedule),128))]).astype('<i4')
    reference=json.loads((HERE.parent/'models/paired-anchor-order.json').read_text())
    sequence_hash=hashlib.sha256(common.tobytes()).hexdigest()
    assert sequence_hash==reference['angular']['sha256']==reference['meanflow_strict']['sha256']
    ids=np.array(inventory['eligible'])[common]
    report['common_order']=dict(rows=len(ids),first_look=103*128,indices_sha256=sequence_hash,
        ordered_ids_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        same_anchor_positions_as_angular_and_strict_meanflow=len(ids))
    for record in sorted((HERE/"fits").glob("*/complete.json")):
        if "smoke" in record.parent.name:continue
        config=json.loads((record.parent/"config.json").read_text())
        fit=json.loads(record.read_text())
        if record.parent.name.endswith("-common"):
            allowed=set(json.loads((HERE.parent/"models/cache/inventory.json").read_text())["eligible"])
            subset=np.array([mid in allowed for mid in eligible])
            assert fit["rows"]==len(allowed) and int(np.sum(c["z"][subset]>0))==fit["training_atoms"]
        else:
            assert fit["rows"]%len(eligible)==0
            assert fit["training_atoms"]==report["atoms"]*(fit["rows"]//len(eligible))
        report["fits"][record.parent.name]=dict(**fit,parameters=config["parameters"])
    (HERE/"inputs-and-fits.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
