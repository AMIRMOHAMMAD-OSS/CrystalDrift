"""Typeset CSP outcomes from the verified raw-record reductions."""
import csv
import json
import numpy as np
from report import HERE,ROOT,PAPER,NEW,LEGACY,LABEL

def read(name):
    with (HERE/name).open(newline="") as f:return list(csv.DictReader(f))

def main():
    q=read("quality.csv");p=read("paired-draws.csv");t=read("speed.csv");c=read("timing-curves.csv")
    lock=json.loads((HERE/"selection.json").read_text())
    audit=json.loads((HERE/"verified.json").read_text())
    def getq(m,k,key):return float(next(r for r in q if r["method"]==m and int(r["k"])==k)[key])
    def cost(m):return next(r for r in t if r["method"]==m)
    def deadline(m):return next(r for r in c if r["method"]==m and r["axis"]=="ms" and r["budget"]=="100")
    tex=[(HERE/"methods.tex").read_text(encoding="utf8"),
         r"\begin{center}\footnotesize\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrr}\hline Objective & Parameters & 256 & 1,024 & 3,392 & Selected & CPU h & Peak RSS (MiB) \\ \hline"]
    cpu=0.
    for m in (v for v in NEW if v!="prior"):
        state=lock[m];fits=[json.loads((HERE/f"fits/{m}-{s}/update-03392.json").read_text()) for s in (11,23,37)]
        cpu+=sum(r["worker_cpu_seconds"] for r in fits)
        sc=[np.mean(state["validation_energy_by_seed"][str(u)]) for u in (256,1024,3392)]
        cpu_h=sum(r["worker_cpu_seconds"] for r in fits)/3600
        peak=max(r["peak_working_set_bytes"] for r in fits)/2**20
        tex.append(f"{LABEL[m]} & {fits[0]['parameters']:,} & {sc[0]:.3f} & {sc[1]:.3f} & {sc[2]:.3f} & {state['updates']:,} & {cpu_h:.3f} & {peak:.0f} "+r"\\")
    tex.extend([r"\hline\end{tabular}}\end{center}",
    r"\begin{figure}[!htbp]\centering\includegraphics[width=.94\linewidth]{csp-validation.pdf}\caption{Validation scores at the three matched update budgets. Thin lines show the three fits and heavy lines show their means. Open circles mark the frozen selections. All 45 points use the same 128 validation targets and five draws per target. Each full physical angular fit deteriorated by the final budget. Backbone width and optimizer family were fixed across the new fits.}\label{fig:v2-csp-validation}\end{figure}",
    r"\begin{table}[!htbp]\centering\small\caption{Raw CSP on all 9,046 MP-20 test targets. MR uses all targets. RMS averages the best accepted normalized displacement per recovered target within each seed, with equal seed weights. $S$ denotes neural evaluations per candidate. At 1, 5 and 20 draws each target uses $S,5S,20S$ evaluations. B1 is median warm batch-one latency.}\label{tab:v2csp}",
    r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrrr}\hline & & & \multicolumn{3}{c}{MR (\%)}&\multicolumn{3}{c}{Mean matched RMS}&\multicolumn{2}{c}{Wall clock}\\",
    r"Model & $S$ & $kS$ & $k=1$ & 5 & 20 & 1 & 5 & 20 & B1 (ms) & min/10k\\ \hline"])
    for m in (*NEW,*LEGACY):
        nfe=0 if m in ("prior","retrieval") else 16 if m in ("flow16","old-flow16") else 1
        row=[LABEL[m],str(nfe),f"{nfe}/{5*nfe}/{20*nfe}",*[f"{100*getq(m,k,'mr'):.2f}" for k in (1,5,20)],
             *[f"{getq(m,k,'mean_normalized_rmse'):.3f}" for k in (1,5,20)],
             f"{float(cost(m)['b1_ms']):.2f}",f"{float(cost(m)['seconds_per_10000'])/60:.3f}"]
        tex.append(" & ".join(row)+r"\\")
    tex.extend([r"\hline\end{tabular}}\end{table}",
    r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{csp-recovery.pdf}\caption{Recovery on the same 128 hash-selected MP-20 targets at equal completed counts and equal elapsed proposal time. Each target owns a serial stream capped at 20 proposals. The shaded pointwise 95\% interval resamples IDs for the full angular generator. Source tables include intervals for every curve. The vertical line marks 100 ms.}\label{fig:v2csp}\end{figure}"])
    pv={(r["second"],int(r["updates"])):r for r in read("paired-validation.csv")}
    d256=pv["radial",256];d3392=pv["radial",3392]
    def summary_delta(r):
        return f"{float(r['energy_delta']):+.3f} "+r"\(["+f"{float(r['lower']):+.3f},{float(r['upper']):+.3f}"+r"]\)"
    tex.append("On the validation energy score, angular minus pair-shell at equal exposure was "+summary_delta(d256)+" after 256 updates and "+summary_delta(d3392)+" after 3,392. Lower scores favor the first method. Thus the early angular advantage reversed by the full scheduling pass.")
    diagnostic=[]
    for k in (1,5,20):
        recovered=[];displacements=[]
        for seed in (11,23,37):
            with np.load(ROOT/f"expansion/training/test/flow1-seed{seed}/outcomes.npz") as op:
                rr=op["rms"][:,:k];hit=np.isfinite(rr).any(1)
                recovered.append(hit.mean());displacements.append(np.where(np.isfinite(rr),rr,np.inf).min(1)[hit].mean())
        diagnostic.append((100*np.mean(recovered),np.mean(displacements)))
    dr="/".join(f"{val[0]:.2f}" for val in diagnostic);rms="/".join(f"{val[1]:.3f}" for val in diagnostic)
    tex.append("One Euler update of the earlier compact velocity checkpoint is a separate diagnostic. With its original same-target candidates, MR at 1/5/20 draws was "+dr+r"\% and mean matched normalized RMS was "+rms+". Its latency belongs to the earlier timing stream.")
    pair={r["second"]:r for r in p if r["k"]=="20"}
    def difference(m):
        v=pair[m]
        return r"\("+f"{100*float(v['mr_delta']):+.2f}"+r"\;["+f"{100*float(v['mr_low']):+.2f},{100*float(v['mr_high']):+.2f}"+r"]\)"
    tex.append(r"At twenty draws, angular training changed the recovered fraction by "+difference("radial")+" percentage points relative to pair-shell drift and by "+difference("power0")+r" relative to center power alone. Brackets give paired 95\% intervals. The change from the fitted prior was "+difference("prior")+". With the same angular field, changing from the fixed-basis backbone to the physical network gave "+difference("fixed-angular")+" points. Relative to the earlier fixed-basis release, the change was "+difference("old-drift")+".")
    tex.append("Angular and pair-shell drift both selected 256 updates. Their test contrast isolates the feature change at the same update count. The center-power control selected 1,024 updates, so its test contrast also includes the selected training duration.")
    hits=[int(r["hits"]) for r in read("seeds.csv") if r["method"]=="angular" and r["k"]=="20"]
    angular_summary=next(r for r in q if r["method"]=="angular" and r["k"]=="20")
    hittext=", ".join(f"{n:,}" for n in hits)
    tex.append("At twenty draws the angular matched-RMS denominators in seeds 11, 23 and 37 were "+hittext+". Its mean matched RMS was "+f"{float(angular_summary['mean_normalized_rmse']):.4f} "+r"\(["+f"{float(angular_summary['rms_low']):.4f},{float(angular_summary['rms_high']):.4f}"+r"]\). Because recovery changes with the method and draw count, these conditional displacement averages can describe different target populations.")
    v=[r for r in read("sensitivities.csv") if r["method"]=="angular" and r["k"]=="20"]
    mx=100*np.mean([float(r["max_site_mr"]) for r in v]);nv=100*np.mean([float(r["native_volume_mr"]) for r in v])
    tex.append(f"Two stricter checks on the same angular draws give {mx:.2f}"+r"\% recovery with the maximum-site matcher and "+f"{nv:.2f}"+r"\% when the accepted RMS draw must also have a native-volume ratio in $[1/1.10,1.10]$, both at twenty draws.")
    tex.append(f"In the sequential benchmark at 100 ms, the angular model recovered {deadline('angular')['hits']}/128 targets with {float(deadline('angular')['mean_completed']):.2f} completed draws per target. The fitted prior recovered {deadline('prior')['hits']}, pair-shell drift {deadline('radial')['hits']}, released fixed-basis drift {deadline('old-drift')['hits']}, matched iterative flow {deadline('flow16')['hits']} and uFlowCSP {deadline('uflow')['hits']}.")
    val_prior=np.mean([np.load(HERE/f"fits/prior-{s}/val-00000.npz")["energy"].mean() for s in (11,23,37)])
    tex.append(f"The seed-only validation energy score averaged {val_prior:.3f}. The fifteen full training trajectories consumed {cpu/3600:.2f} CPU worker-hours and zero GPU hours. The archive records {audit['total_scored']:,} new candidates, including {audit['exception_draws']} matcher exceptions and {audit['public_api_checks']:,} direct public matcher checks.")
    tex.append(f"Peak GPU allocations in the complete proposal benchmark were {float(cost('angular')['peak_cuda_mib']):.1f} MiB for angular drift, {float(cost('radial')['peak_cuda_mib']):.1f} MiB for pair-shell drift and {float(cost('uflow')['peak_cuda_mib']):.1f} MiB for uFlowCSP. These include the resident model and timed activations. The training descriptor is absent from each physical generator's sampling path.")
    cpuclock={r["method"]:r for r in read("cpu/timing.csv")}
    cpu_comp=("prior","radial","angular","flow16")
    cpu_vals=[f"{float(cpuclock[m]['b1_median_ms']):.2f}" for m in cpu_comp]
    cpu_10k=[f"{float(cpuclock[m]['seconds_per_10000']):.2f}" for m in cpu_comp]
    tex.append("On one Intel Core i7-12700H thread, median warm latencies for the fitted prior, pair-shell drift, angular drift and matched sixteen-step flow were "+", ".join(cpu_vals)+" ms. Their separate batch-24 workloads of exactly 10,000 host-decoded outputs took "+", ".join(cpu_10k)+" s in the same order. The 128 CPU targets each received five batch-one calls per method, with method order balanced within six-target blocks. Concurrent matching jobs shared the CPU during these clocks.")
    balanced={r["method"]:r for r in read("interleaved/recovery.csv") if r["ms"]=="100"}
    counts=[int(balanced[m]["hits"]) for m in ("prior","radial","angular","old-drift","flow16","old-flow16")]
    tex.append(r"We repeated the serial deadlines after balancing method order within six-target blocks (Figure~\ref{fig:v2balanced}). All six services remained resident. Each target--method pair ran for twenty consecutive draws, with shared per-target noise for the physical arms. At 100 ms, the fitted prior, pair-shell drift and angular drift recovered "+f"{counts[0]}, {counts[1]} and {counts[2]} of 128 targets. The earlier fixed-basis generator recovered {counts[3]}, the matched sixteen-step flow {counts[4]}, and the earlier sixteen-step flow {counts[5]}. The evidence package retains both streams, their separate proposals and paired target intervals.")

    body=[]
    for part in tex:
        body.append(part)
        if not part.startswith("\\") and not part.rstrip().endswith((r"\\",r"\\ \hline")):
            body.append("")
    (PAPER/"conditioned.tex").write_text("\n".join(body)+"\n",encoding="utf8")

if __name__=="__main__":main()
