"""Interleaved warm CPU latency at one proposal per formula (no per-query result cache)."""
from time import perf_counter
import json
from confirm import CONF,ROOT,records,Prior,donor_index,model_and_hash,proposals,native,apply,np,torch
from experiment import load_service

torch.set_num_threads(1)
rows=records();prior=Prior(rows)
with np.load(ROOT/"scaled/mp20-data/train.npz") as tr:
    donors=donor_index(rows,tr["z"])
net,_=model_and_hash("angular-C");a=load_service("angular",11)
with np.load(CONF/"predictions.npz") as p:
    ids=p["ids"][:128]
with np.load(ROOT/"scaled/mp20-data/test.npz") as raw:
    ix={str(s):i for i,s in enumerate(raw["ids"])}
    zs=raw["z"][[ix[s] for s in ids]]


def sample(z,arm,rng):
    if arm=="original_A":
        k,f=a(z[None],rng)
        return len(f)
    g,p,_=proposals(z,donors,1)[0]
    s=prior.draw(g,p,rng)
    if arm=="symbolic_C":
        s=apply(net,[s])[0]
    return len(native(s,g,z))


arms=["original_A","symbolic_prior","symbolic_C"]
for i in range(8):
    for arm in arms:
        sample(zs[i],arm,np.random.default_rng(2026+i))
times={key:[] for key in arms}
for i,z in enumerate(zs):
    for j in range(3):
        arm=arms[(i+j)%3];rng=np.random.default_rng(103819+i)
        tick=perf_counter();sample(z,arm,rng);times[arm].append((perf_counter()-tick)*1000)
summary={"scope":"batch=1 CPU one thread, 128 warm interleaved samples, in-memory donor search included, index construction/model load and scoring excluded. Host shared with other work.","milliseconds":{arm:dict(zip(("q25","median","q75"),np.percentile(v,[25,50,75]))) for arm,v in times.items()}}
(CONF/"cpu-clock.json").write_text(json.dumps(summary,indent=2)+"\n")
print(summary)
