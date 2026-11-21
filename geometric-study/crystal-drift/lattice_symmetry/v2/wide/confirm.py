"""Follow the independently recorded validation confirmation after seed 11."""
import run
import numpy as np
from pathlib import Path

def main():
    train=run.train;stage=run.stage;model=run.model
    for seed in (23,37):
        train.train(("angular",seed,256,"cpu"))
        stage.validate(["angular"],[seed],run.STEPS)
    for seed in (23,37):run.run(seed)
    diff=[]
    ids=None
    for seed in (11,23,37):
        pair=[]
        for arm in ("angular","angular-wide"):
            with np.load(model.HERE/"fits"/f"{arm}-{seed}"/"val-00256.npz") as d:
                if ids is None:ids=d["ids"].copy()
                assert np.array_equal(ids,d["ids"])
                pair.append(d["energy"])
        diff.append(pair[1]-pair[0])
    delta=np.array(diff)
    idx=np.random.default_rng(81831).integers(len(ids),size=(5000,len(ids)))
    def summary(a):
        t=a.mean(0)
        return dict(mean=float(t.mean()),paired_95=np.quantile(t[idx].mean(1),[.025,.975]).tolist())
    model.save_npz(run.HERE/"paired-validation.npz",ids=ids,seeds=[11,23,37],difference=delta)
    model.write_json(run.HERE/"confirmation.json",dict(per_seed=delta.mean(1).tolist(),all_fits=summary(delta),
                                                       independent_fits=summary(delta[1:])))


if __name__=="__main__":main()
