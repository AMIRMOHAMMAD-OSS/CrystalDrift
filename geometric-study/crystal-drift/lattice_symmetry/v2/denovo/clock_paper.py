"""Check the complete hardware replay and join cost to a named raw pool."""
import csv
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    rows=[]
    for name in ("fixed-basis","radial","angular"):
        def load(p):return json.loads(p.read_text(encoding="utf8"))
        cuda=load(HERE/f"pools/{name}-cuda.json")
        cpu=load(HERE/f"pools/{name}-cpu.json")
        result=load(HERE/f"panel/{name}.json")
        assert cuda["checkpoint_sha256"]==cpu["checkpoint_sha256"] and cuda["cpu_pool_sha256"]==cpu["data_sha256"]
        assert len(cuda["batch_seconds"])==len(cuda["sizes"])==79 and sum(cuda["sizes"])==10000
        np.testing.assert_allclose(sum(cuda["batch_seconds"]),cuda["elapsed_seconds"])
        assert cuda["finite"]==10000 and len(cuda["b1_ms"])==128
        rows.append(dict(method=name,device=cuda["device"],seconds=cuda["elapsed_seconds"],b1_ms=cuda["median_b1_ms"],
                         structures_per_second=10000/cuda["elapsed_seconds"],valid_unique=result["valid_unique"],
                         estimated_valid_unique_per_gpu_minute=result["valid_unique"]*60/cuda["elapsed_seconds"],
                         max_chart=cuda["max_chart_difference"],max_wrapped_fractional=cuda["max_fractional_difference"]))
    with (HERE/"gpu-costs.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert len(set(r["device"] for r in rows))==1
    def seq(key,fmt):
        return ", ".join(format(r[key],fmt) for r in rows)
    def sci(value):
        coefficient,exponent=f"{value:.1e}".split("e")
        return "$"+coefficient+r"\times10^{"+str(int(exponent))+"}$"
    s=r"\paragraph{Common GPU workload.} On the RTX 3060 Laptop GPU, fixed-basis, radial and angular generation took "+seq("seconds",".2f")+" seconds for 10,000 structures at batch 128. Their warm batch-one medians were "+seq("b1_ms",".2f")+" ms. The clock includes drawing the composition, sampling a physical seed, decoding one network evaluation, transferring to host memory and constructing Cartesian coordinates. It ends before fingerprinting or relaxation. Applying the corresponding CPU-scored valid-unique counts to these clocks gives estimated yields of "+seq("estimated_valid_unique_per_gpu_minute",",.0f")+r" per GPU minute. CPU and GPU streams had identical chemistry, with worst case differences of "+sci(max(r['max_chart'] for r in rows))+" in a log-metric coordinate and "+sci(max(r['max_wrapped_fractional'] for r in rows))+" in a wrapped fractional coordinate across the three paired 10,000-cell replays."
    (HERE.parent/"paper/denovo-clock.tex").write_text(s+"\n",encoding="utf8")


if __name__=="__main__":main()
