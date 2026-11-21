"""Release the reserved accelerator after all registered clocks have materialized."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
V3=Path(__file__).resolve().parents[1]
E=V3/"evaluation";T=E/"timing"
matrix=(E/"clock-frozen.json").read_bytes()
signature=hashlib.sha256(matrix).hexdigest()
settings=json.loads((T/"settings.json").read_text())
assert settings["matrix_sha"]==signature
for m in settings["method_order"]:
    with np.load(T/(m+"-10k.npz")) as d:
        assert len(d["ids"])==10000 and str(d["matrix_sha"])==signature
    with np.load(T/(m+".npz")) as d:assert d["elapsed_ms"].shape==(64,20) and (np.diff(d["elapsed_ms"],axis=1)>=0).all()
with np.load(T/"cpu-first.npz") as d:assert d["first_ms"].shape==(64,len(settings["method_order"])) and str(d["matrix_sha"])==signature
with np.load(T/"selector-cost.npz") as d:assert d["ms"].shape==(64,3) and int(d["calls"])==64
(V3/"models/timing-finished.json").write_text(json.dumps(dict(utc=time.time(),matrix_sha=signature,methods=settings["method_order"],
    gpu_batch_one_targets=64,ordered_draws=20,batched_attempts_each=10000,cpu_threads=1),indent=2))
print("Dedicated clock verified and device released.")

