"""Typeset validation, public-test and GPU results directly from case-derived summaries."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]/"paper"
LABEL={"drift":"CrytalDrift","iid":"IID-positive drift","meanflow":"Compact MeanFlow","flow":"Compact flow",
       "flow1":"Flow, 1 Euler step","flow16":"Flow, 16 Euler steps","retrieval":"Retrieval","uflow":"Released uFlowCSP"}


def records(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))


def line(*columns):return " & ".join(str(x) for x in columns)+r" \\"


def training():
    choices=json.loads((HERE/"selection.json").read_text())
    out=[r"\subsection{One-step supervision and independently sampled positive banks}",
r"The original shuffled training design placed the scheduled structure in the first positive slot. Our paired arm kept the identical sequence of composition labels, drew all four positive indices independently with replacement in each block, and generated independent query and negative sets. Hashes of the complete ordered schedules agree for each seed and budget. The distinction matters when a composition has more than one observed geometry. The independent-positive design satisfies the bank-exchangeability hypothesis of the supplement conditional on the labels, parameters and query inputs.",
r"We trained a dedicated one-step control using the MeanFlow identity \cite{meanflow} on the same set network. Two independent uniform times are sorted into $r\leq t$, with probability $1/4$ assigned to the diagonal $r=t$. On the straight log-lattice path from data at $0$ to noise at $1$, let $v_k=k_N-k_D$. On the site torus we shuffle target sites within species, select componentwise wrapped noise--data displacements, subtract their occupied-site mean, and call the resulting tangent $v_f$. The network predicts $u=(u_k,u_f)$, with the site head expressed in units of $\sqrt{12}$. Its target is",
r"\[ \operatorname{sg}\!\left[(v_k,\sqrt{12}\,v_f)-(t-r)\{\partial_k u\,v_k+\partial_f u\,v_f+\partial_t u\}\right]. \]",
r"The derivative follows $k_t=k_N+(t-1)v_k$ and $f_t=f_N+(t-1)v_f\pmod 1$ with $r$ fixed and is evaluated by a forward-mode Jacobian--vector product. Centering $v_f$ translates all occupied clean sites by the same vector at $t=0$, preserving their relative positions. Six interval coefficients feed $(t-r)$ into the shared lattice input features. Training uses the same lattice and masked site squared errors as compact flow. At inference one call at $(r,t)=(0,1)$ subtracts $u_k$ from the lattice noise and $u_f/\sqrt{12}$ from the site noise modulo one. A double-precision finite-difference check agreed with the two JVP outputs to $4.0\times10^{-11}$.",
r"Each fit used width 64, AdamW at $10^{-3}$, clipping at norm 10, 64 positive examples and 16 composition blocks per update. One, three and ten passes correspond to 1,696, 5,088 and 16,960 updates and 108,544, 325,632 and 1,085,440 positive examples per fit. Each drift update evaluated 64 independent negative and 64 live query structures. We chose one pass count per objective by the lowest mean validation geometry score at five draws on 128 fixed material IDs across three seeds. Flow selection used 16 Euler steps. The one-Euler diagnostic uses those same selected weights. The official test set had already been examined in the earlier study.",
r"\begin{table}[!htbp]\centering\small",
r"\caption{Validation-only checkpoint selection. The energy score measures forecast error in standardized log volume and three nearest-neighbor summaries. CPU hours sum process time over three complete ten-pass fits. Each fit used one CPU thread and zero GPU hours.}\label{t:exp-training-fits}",
r"\begin{tabular}{lrrrrrr}\hline Objective & Parameters & Pass 1 & Pass 3 & Pass 10 & Selected & CPU h\\\hline"]
    for m,spec in choices.items():
        fits=[json.loads((HERE/f"fits/{m}-seed{s}/pass10.json").read_text()) for s in (11,23,37)]
        out.append(line(LABEL[m],f'{fits[0]["parameters"]:,}',*[f'{spec["validation_energy"][str(p)]:.3f}' for p in (1,3,10)],
                        spec["pass"],f'{sum(r["process_cpu_seconds"] for r in fits)/3600:.2f}'))
    out.extend([r"\hline\end{tabular}\end{table}",r"\begin{figure}[!htbp]\centering\includegraphics[width=.88\linewidth]{exp-training-curves.pdf}",
r"\caption{Validation geometry score across the fixed training budgets. Thin lines show individual fit seeds and bold lines their arithmetic means. All arms use the same target IDs and five-draw input noises.}\label{fig:exp-training-curves}\end{figure}"])
    if (HERE/"validation-paired.csv").exists() and len(choices)==4:
        pairs=records(HERE/"validation-paired.csv")
        diff=next(r for r in pairs if int(r["passes"])==1 and r["first"]=="iid")
        out.append("Both drift samplers selected the first pass. At that budget the IID-minus-shuffled validation score difference was %.4f (paired 95\\%% interval $[%.4f,%.4f]$). Training for ten passes increased both drift validation scores. We retained the original width for all four objectives. The two velocity objectives selected ten passes."%
                   (float(diff["mean"]),float(diff["lo"]),float(diff["hi"])))
    if (HERE/"test-quality.csv").exists():
        rows=records(HERE/"test-quality.csv")
        cost={r["method"]:r for r in records(HERE/"gpu/runtime.csv")}
        out.extend([r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{3pt}",
r"\caption{Composition-conditioned prediction on all 9,046 official MP-20 test targets, with matched cost measurements. Each raw draw is compared with its own target. RMS averages the lowest accepted error among recovered targets. Compact and retrieval accuracy rows equally weight three seeds. Each released model contributes one checkpoint. The warm batch-one median and fixed 10,000-request wall time use a dedicated GPU timing stream on the same 128-target prefix as Figure~\ref{fig:exp-gpu}. Retrieval runs on the host.}\label{t:exp-conditioned}",
r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrrr}\hline & & & \multicolumn{3}{c}{MR (\%)} & \multicolumn{3}{c}{Normalized RMS} & \multicolumn{2}{c}{Measured cost}\\",
r"Method & $S$ & $kS$ (1/5/20) & $k{=}1$ & 5 & 20 & $k{=}1$ & 5 & 20 & B1 (ms) & min/10,000\\\hline"])
        for m in LABEL:
            per=[r for r in rows if r["method"]==m]
            if len(per)!=3:continue
            step=0 if m=="retrieval" else 16 if m=="flow16" else 1
            out.append(line(LABEL[m],step,"/".join(str(k*step) for k in (1,5,20)),
                    *[f'{100*float(r["mr"]):.2f}' for r in per],*[f'{float(r["rmse"]):.3f}' for r in per],
                    f'{float(cost[m]["b1_median_ms"]):.2f}',f'{float(cost[m]["minutes_for_10000"]):.3f}'))
        external=HERE.parent/"external/csp-s5/summary.json"
        if external.exists():
            results=json.loads(external.read_text())
            keys=[f"9046@{k}" for k in (1,5,20)]
            if any(k in results and results[k]["targets"]==9046 for k in keys):
                timing=json.loads((HERE.parent/"external/timing-s5/runtime.json").read_text())
                out.append(line("CrystalFlow, 5 steps",5,"5/25/100",
                    *[f'{100*results[k]["mr"]:.2f}' if k in results else "---" for k in keys],
                    *[f'{results[k]["rmse"]:.3f}' if k in results else "---" for k in keys],
                    f'{timing["median_b1_ms"]:.2f}',f'{timing["batch_total_seconds"]/60:.3f}'))
        out.append(r"\hline\end{tabular}}\end{table}")
        if (HERE/"test-paired.csv").exists():
            pairs=records(HERE/"test-paired.csv");cols=[]
            for m in ("iid","meanflow","flow1","flow16"):
                r=next((r for r in pairs if int(r["k"])==20 and r["metric"]=="mr" and r["b"]==m),None)
                if r:cols.append("%s %+.2f ($[%.2f,%.2f]$)"%(LABEL[m],100*float(r["mean"]),100*float(r["lo"]),100*float(r["hi"])))
            if cols:out.append(r"At twenty draws, differences in recovery (CrytalDrift minus the named method, percentage points with 95\% paired target-bootstrap intervals) were "+", ".join(cols)+".")
    if (HERE/"geometry.csv").exists():
        geo=records(HERE/"geometry.csv")
        out.extend([r"\begin{table}[!htbp]\centering\small",
r"\caption{Separating lattice and site errors. Shape is the Frobenius norm of the traceless log-stretch difference in the deposited basis. Volume is absolute log volume-per-atom error. Both average 20 draws per test target. Site RMS is a displacement on the unit fractional torus. For the three neighbor columns, we placed each generated motif in its reference lattice and measured mean absolute errors in the minimum, median and 90th percentile site-neighbor distances, in \AA, for five draws on the 128-target prefix.}\label{t:exp-shape}",
r"\begin{tabular}{lrrrrrr}\hline Method & Shape & Volume & Site RMS & Minimum & Median & 90th pct.\\\hline"])
        for m in ("drift","iid","meanflow","flow1","flow16"):
            pick=[r for r in geo if r["arm"].split("-seed")[0]==m]
            if len(pick)!=3:continue
            keys=("mean_traceless_logstretch_norm","mean_absolute_log_volume_per_atom","mean_fractional_site_assignment_rms","mean_absolute_minimum_site_neighbor_A",
                  "mean_absolute_median_site_neighbor_A","mean_absolute_q90_site_neighbor_A")
            out.append(line(LABEL[m],*[f"{np.mean([float(r[k]) for r in pick]):.3f}" for k in keys]))
        out.append(r"\hline\end{tabular}\end{table}")
        out.append(r"The site RMS column gives a feasible species-matched RMS displacement on the unit fractional torus in the deposited basis. We initialize the translation by aligning a fixed atom of the rarest target species to each eligible candidate atom, alternate exact within-species assignments and mean-residual translation updates up to four times, and retain the smallest value. This gives an upper bound on the optimum over origins. It is evaluated on the same 128 targets and five draws as the site-neighbor columns.")
        if len(geo)==15:
            def mean(m,key):
                return np.mean([float(r[key]) for r in geo if r["arm"].split("-seed")[0]==m])
            volume="mean_absolute_log_volume_per_atom";site="mean_fractional_site_assignment_rms"
            out.append("The compact MeanFlow and drifting models had similar fractional-site displacement (%.3f and %.3f), while MeanFlow halved the mean absolute log-volume error (%.3f versus %.3f). The 16-step flow had the smallest fractional-site error (%.3f) among the compact methods. Cell-shape error and site-placement error therefore favor different controls."%
                (mean("meanflow",site),mean("drift",site),mean("meanflow",volume),mean("drift",volume),mean("flow16",site)))
    (PAPER/"exp-training.tex").write_text("\n".join(out)+"\n",encoding="utf8")


def gpu():
    path=HERE/"gpu/runtime.csv"
    if not path.exists():return
    out=[r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{exp-gpu-curves.pdf}",
r"\caption{Raw recovery on 128 hash-selected full-chemistry MP-20 targets. Both panels use the same generated structures. Left, each target receives its own first $k$ candidates. Right, it receives the candidates completed before each measured batch-one deadline. Each episode ends at twenty candidates. All matches are scored retrospectively.}\label{fig:exp-gpu}\end{figure}",
r"We measured serial and 10,000-request workloads on the same GeForce RTX 3060 Laptop GPU using FP32, with TF32 disabled and one host compute thread. Compact models use their validation-selected seed-11 checkpoints. Calls start from host species arrays and end with species, a canonical lattice and fractional and Cartesian coordinates on the host. Retrieval runs on the host. We precomputed eligible retrieval donors and the released uFlowCSP species tokens for these 128 inputs. Five untimed calls precede each measured batch size. The throughput workload cycles the same 128 compositions at batch 24 with a final batch of sixteen. Setup, retrospective matching and relaxation fall outside the timer.",
r"\begin{table}[!htbp]\centering\small\caption{Measured generation costs on the 128-target full-chemistry cohort. NFE counts neural evaluations per structure. Batch-one values are medians over 2,560 serial completions. Throughput uses a complete timed 10,000-request workload for each model.}\label{t:exp-gpu}",
r"\begin{tabular}{lrrrr}\hline Method & NFE & Median B1 (ms) & min/10,000 & Structures/s\\\hline"]
    results=records(path)
    for m in ("drift","iid","meanflow","flow1","flow16","retrieval","uflow"):
        row=next((r for r in results if r["method"]==m),None)
        if row:
            out.append(line(LABEL[m],0 if m=="retrieval" else 16 if m=="flow16" else 1,
                f'{float(row["b1_median_ms"]):.3f}',f'{float(row["minutes_for_10000"]):.3f}',f'{float(row["structures_per_second"]):.1f}'))
    ext=HERE.parent/"external/timing-s5/runtime.json"
    if ext.exists():
        timing=json.loads(ext.read_text())
        out.append(line("CrystalFlow, 5 steps",5,f'{timing["median_b1_ms"]:.3f}',
            f'{timing["batch_total_seconds"]/60:.3f}',f'{timing["per_second"]:.1f}'))
    out.append(r"\hline\end{tabular}\end{table}")
    out.append(r"The GPU was reserved for each timed method. Concurrent CPU-only experiments shared the host, and the batch-one and 10,000-request results describe this machine and launch configuration.")
    paired_path=HERE/"gpu/paired-recovery.csv"
    if paired_path.exists():
        paired=records(paired_path)
        curves=records(HERE/"gpu/recovery-curves.csv")
        delta=next((s for s in paired if s["second"]=="flow16" and s["axis"]=="time" and float(s["budget"])==100),None)
        if delta:
            chosen={s["method"]:s for s in curves if s["axis"]=="time" and float(s["budget"])==100}
            for m in ("drift","flow16","uflow","retrieval"):
                assert m in chosen
            out.append(r"At a 100~ms deadline, CrytalDrift averaged %.2f completed draws per target and recovered %.1f\%%, compared with %.2f draws and %.1f\%% for the 16-step flow. Their paired difference was %+.1f percentage points (95\%% target-bootstrap interval $[%.1f,%.1f]$). Released uFlowCSP recovered %.1f\%% and retrieval recovered %.1f\%% within the same deadline."%
                (float(chosen["drift"]["mean_completed"]),100*float(chosen["drift"]["mr"]),
                 float(chosen["flow16"]["mean_completed"]),100*float(chosen["flow16"]["mr"]),
                 100*float(delta["mr_difference"]),100*float(delta["lo"]),100*float(delta["hi"]),
                 100*float(chosen["uflow"]["mr"]),100*float(chosen["retrieval"]["mr"])))
    (PAPER/"exp-gpu.tex").write_text("\n".join(out)+"\n",encoding="utf8")


def seeds():
    path=HERE/"test-quality-seeds.csv"
    if not path.exists():return
    rows=records(path)
    out=[r"\subsection{Training-seed variation and matcher sensitivity}",
         r"\begin{table}[!htbp]\centering\small",
         r"\caption{Validation-selected models on all 9,046 official targets. The three rate and RMS columns reuse the first one, five and twenty draws of the same ordered stream. The last two columns give twenty-draw recovery under the maximum-site predicate and under a native-volume filter, respectively. The volume filter requires an accepted RMS match and $|\log(V_{\rm gen}/V_{\rm ref})|\leq\log(1.10)$. All recovery columns are percentages.}\label{t:exp-perseed}",
         r"\begin{tabular}{llrrrrrrrr}\hline &&\multicolumn{3}{c}{MR (\%)}&\multicolumn{3}{c}{Normalized RMS} &\multicolumn{2}{c}{MR@20 (\%)}\\",
         r"Method & Seed & 1 & 5 & 20 & 1 & 5 & 20 & Max. site & Volume\\\hline"]
    for m in ("drift","iid","meanflow","flow1","flow16","retrieval","uflow"):
        per=[r for r in rows if r["method"]==m]
        for j,seed in enumerate(sorted(set(r["seed"] for r in per))):
            stream=sorted([r for r in per if r["seed"]==seed],key=lambda r:int(r["k"]))
            assert [int(r["k"]) for r in stream]==[1,5,20]
            out.append(line(LABEL[m] if j==0 else "",seed if m!="uflow" else "release",
                            *[f'{100*float(r["mr"]):.2f}' for r in stream],*[f'{float(r["rmse"]):.3f}' for r in stream],
                            f'{100*float(stream[-1]["max_site"]):.2f}',f'{100*float(stream[-1]["volume_mr"]):.2f}'))
    ext=HERE.parent/"external/csp-s5/score-crosswalk.npz"
    if ext.exists():
        with np.load(ext) as data:
            if data["scored"].all():
                from analysis import array_metrics,estimate
                pool={key:data[key] for key in ("rms","strict_fit","log_volume_ratio")}
                vals=[estimate(array_metrics(pool,k).mean(0)) for k in (1,5,20)]
                out.append(line("CrystalFlow, 5 steps","release",
                                *[f'{100*v[0]:.2f}' for v in vals],*[f'{v[1]:.3f}' for v in vals],
                                f'{100*vals[-1][2]:.2f}',f'{100*vals[-1][3]:.2f}'))
    out.append(r"\hline\end{tabular}\end{table}")
    if (HERE/"verification.json").exists():
        replication=json.loads((HERE/"verification.json").read_text())["archived_one_pass_drift_replication"]
        changed=sum(r["hit_flag_disagreements"] for r in replication.values())
        if changed==1:
            assert replication["drift-seed23"]["changed_candidates"]==[dict(id="mp-1069147",draw=4,archived=False,refit=True)]
            assert sum(r["changed_target_indicators"]["1"]+r["changed_target_indicators"]["20"] for r in replication.values())==0
            assert replication["drift-seed23"]["changed_target_indicators"]["5"]==1
            out.append(r"Repeating the original one-pass drifting fits with one CPU thread changed the accepted status of one candidate among 542,760 relative to the archived two-thread fits. For seed 23, draw four for \texttt{mp-1069147} was accepted in the refit. Its target gained a five-draw match and was recovered by twenty draws in both runs. One- and twenty-draw recovery indicators agreed across all targets and seeds.")
    (PAPER/"exp-seeds.tex").write_text("\n".join(out)+"\n",encoding="utf8")


if __name__=="__main__":training();gpu();seeds()
