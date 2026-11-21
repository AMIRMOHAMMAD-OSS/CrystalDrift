"""Rebuild target-paired summaries, editable figures and the binary benchmark table."""
import argparse
import csv
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[1]/"paper"
METHODS = ("exact","flow1","flow16","retrieval","uflow")
NAMES = dict(exact="CrytalDrift",flow1="Flow, 1 step",flow16="Flow, 16 steps",
             retrieval="Retrieval",uflow="Released uFlowCSP")
STEPS = dict(exact=1,flow1=1,flow16=16,retrieval=0,uflow=1)
COLORS = dict(exact="#b6432e",flow1="#50968d",flow16="#146575",retrieval="#7561a8",uflow="#293c65")
LINES = dict(exact="-",flow1=":",flow16="--",retrieval="-.",uflow="-")
BUDGETS = [.1,.25,.5,1,2,5,10,20,50,100,250,500,1000,2500,5000,10000]
SEEDS = (11,23,37)
BOOT = 10000
counts = np.random.default_rng(20261120).multinomial(128,np.full(128,1/128),size=BOOT).astype(float)


def write_csv(name,rows):
    with (OUT/name).open("w",newline="",encoding="utf8") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def interval(x):
    return np.nanquantile(x,[.025,.975]).tolist()


def stats(masked):
    # Each finite entry is the seed's best RMS on this target; missed cases
    # count in MR, while only hits enter that seed's conditional RMS mean.
    ok = np.isfinite(masked)
    hit_mean = ok.mean(0)
    den = counts@ok.T
    boot_rmse = (counts@np.nan_to_num(masked).T)/np.where(den>0,den,np.nan)
    boot_mr = (counts@hit_mean)/128
    a,b = interval(boot_mr)
    c,d = interval(np.nanmean(boot_rmse,axis=1))
    return dict(mr=hit_mean.mean(),mr_low=a,mr_high=b,
                rmse=np.nanmean(masked,axis=1).mean(),rmse_low=c,rmse_high=d),(boot_mr,np.nanmean(boot_rmse,axis=1))


def best(rms):
    val = np.where(np.isfinite(rms),rms,np.inf).min(axis=-1)
    return np.where(np.isfinite(val),val,np.nan)


def data_arrays(method,kind):
    values,times = [],[]
    seeds=(2026,) if method=="uflow" else SEEDS
    for seed in seeds:
        with np.load(OUT/f"rms-{kind}-{method}-{seed}.npz") as src:
            ids,rms = src["ids"],src["rms"]
            assert rms.shape==(128,20)
            values.append(rms)
        if kind=="episode":
            with np.load(OUT/f"episode-{method}-{seed}.npz") as src:
                np.testing.assert_equal(src["ids"],ids)
                times.append(src["elapsed_ms"])
    return np.array(values),np.array(times),seeds


def summarize(quality_only):
    prefix, by_seed, timed, time_summary, pairs = [],[],[],[],[]
    boot,by_case = {},{}
    curve = {}
    clock_boot,clock_pairs,clock_seeds={},{},[]
    for m in METHODS:
        d,_,seeds=data_arrays(m,"headline")
        for k in range(1,21):
            b=best(d[:,:,:k])
            by_case[m,k]=b
            result,boot[(m,k)] = stats(b)
            prefix.append(dict(method=m,draws=k,nfe_per_draw=STEPS[m],nfe_per_target=k*STEPS[m],**result))
            if k in (1,5,20):
                for s,vs in zip(seeds,b):
                    stat,_=stats(vs[None,:])
                    by_seed.append(dict(method=m,seed=s,draws=k,hits=int(np.isfinite(vs).sum()),**stat))
        if quality_only: continue
        d,t,seeds=data_arrays(m,"episode")
        assert t.shape==d.shape
        duration=np.diff(t,axis=-1,prepend=np.zeros((*t.shape[:2],1)))
        for budget in BUDGETS:
            completed = t<=budget
            ok = np.any(np.isfinite(d)&completed,axis=-1)
            case_rate=ok.mean(axis=0)
            clock_boot[m,budget]=(counts@case_rate)/128
            ci=interval(clock_boot[m,budget])
            timed.append(dict(method=m,budget_ms=budget,mean_completed=completed.sum(-1).mean(),
                              completed_total=int(completed.sum()),mr=ok.mean(),mr_low=ci[0],mr_high=ci[1]))
            for h,seed in enumerate(seeds):
                seed_ci=interval((counts@ok[h].astype(float))/128)
                clock_seeds.append(dict(method=m,seed=seed,budget_ms=budget,
                    median_b1_ms=np.median(duration[h]),p10_b1_ms=np.quantile(duration[h],.1),
                    p90_b1_ms=np.quantile(duration[h],.9),completed=int(completed[h].sum()),
                    target_hits=int(ok[h].sum()),mr=ok[h].mean(),mr_low=seed_ci[0],mr_high=seed_ci[1]))
        curve[m]=np.where(np.isfinite(d),t,np.inf).min(axis=-1).ravel()
        s=seeds[0]
        throughput={}
        path=OUT/f"ten-thousand-{m}-{s}.npz"
        if path.exists():
            with np.load(path) as src:
                elapsed=float(src["elapsed_s"])
                assert sum(src["counts"])==10000
                full=src["batch_ms"][src["counts"]==12]
            throughput=dict(seconds_10000=elapsed,structures_per_second=10000/elapsed,
                batch12_p10_ms=np.quantile(full,.1),batch12_median_ms=np.median(full),
                batch12_p90_ms=np.quantile(full,.9))
        time_summary.append(dict(method=m,median_b1_ms=np.median(duration),p10_b1_ms=np.quantile(duration,.1),
                              p90_b1_ms=np.quantile(duration,.9),seconds_10000=throughput.get("seconds_10000"),
                              structures_per_second=throughput.get("structures_per_second"),
                              batch12_p10_ms=throughput.get("batch12_p10_ms"),
                              batch12_median_ms=throughput.get("batch12_median_ms"),
                              batch12_p90_ms=throughput.get("batch12_p90_ms")))
    for k in (1,5,20):
        for j,m in enumerate(METHODS):
            for other in METHODS[j+1:]:
                a,b=interval(boot[(m,k)][0]-boot[(other,k)][0])
                c,d=interval(boot[(m,k)][1]-boot[(other,k)][1])
                x=next(row["mr"] for row in prefix if row["method"]==m and row["draws"]==k)
                y=next(row["mr"] for row in prefix if row["method"]==other and row["draws"]==k)
                first,second=np.broadcast_arrays(by_case[m,k],by_case[other,k])
                delta=first-second
                common=np.isfinite(delta)
                den=counts@common.T
                boot_delta=(counts@np.nan_to_num(delta).T)/np.where(den>0,den,np.nan)
                e,f=interval(np.nanmean(boot_delta,axis=1))
                pairs.append(dict(first=m,second=other,draws=k,mr_difference=x-y,ci_low=a,ci_high=b,
                    conditional_rmse_difference=np.nanmean(first,axis=1).mean()-np.nanmean(second,axis=1).mean(),
                    conditional_rmse_low=c,conditional_rmse_high=d,
                    mean_common_hits_per_seed=common.sum(1).mean(),
                    common_hit_rmse_difference=np.nanmean(delta,axis=1).mean(),common_hit_rmse_low=e,common_hit_rmse_high=f))
    write_csv("headline-prefix.csv",prefix)
    write_csv("headline-per-seed.csv",by_seed)
    write_csv("headline-paired.csv",pairs)
    if quality_only: return
    write_csv("clock-budgets.csv",timed)
    for budget in BUDGETS:
        for j,m in enumerate(METHODS):
            for other in METHODS[j+1:]:
                a,b=interval(clock_boot[m,budget]-clock_boot[other,budget])
                x=next(row["mr"] for row in timed if row["method"]==m and row["budget_ms"]==budget)
                y=next(row["mr"] for row in timed if row["method"]==other and row["budget_ms"]==budget)
                clock_pairs[m,other,budget]=dict(first=m,second=other,budget_ms=budget,
                    mr_difference=x-y,ci_low=a,ci_high=b)
    write_csv("clock-paired.csv",list(clock_pairs.values()))
    write_csv("clock-per-seed.csv",clock_seeds)
    write_csv("runtime.csv",time_summary)
    plot(prefix,curve)
    write_paper(prefix,pairs,time_summary,timed)


def plot(prefix,curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none","svg.hashsalt":"binary-20261120",
                         "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.25),sharey=True)
    for m in METHODS:
        rows=[r for r in prefix if r["method"]==m]
        axes[0].plot(range(1,21),100*np.array([r["mr"] for r in rows]),LINES[m],
                     color=COLORS[m],label=NAMES[m],lw=2)
        arrivals=np.sort(curve[m])
        xs=np.unique(np.r_[.02,arrivals[(arrivals>.02)&(arrivals<10000)],10000])
        ys=100*np.searchsorted(arrivals,xs,side="right")/len(arrivals)
        axes[1].step(xs,ys,where="post",ls=LINES[m],color=COLORS[m],lw=2)
    axes[0].set(xlim=(1,20),ylim=(-2,102),xlabel="Candidates per target",ylabel="Targets recovered (%)",xticks=[1,5,10,15,20])
    axes[1].set_xscale("log")
    axes[1].set(xlim=(.02,10000),xlabel="Measured serial proposal time per target (ms)",xticks=[.1,1,10,100,1000,10000])
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:g}" if x<1000 else f"{x/1000:g}k"))
    for ax,title in zip(axes,["a   Fixed candidate count","b   Fixed wall time"]):
        ax.set_title(title,loc="left",fontweight="bold",fontsize=10)
        ax.grid(True,axis="y",color="#ddd",linewidth=.65)
        ax.tick_params(length=3)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc="lower center",
               bbox_to_anchor=(.5,0),frameon=False,handlelength=2.5,columnspacing=1.5)
    fig.subplots_adjust(left=.084,right=.985,top=.9,bottom=.32,wspace=.12)
    fig.savefig(PAPER/"revision-binary-accuracy-time.pdf",metadata={"CreationDate":None,"ModDate":None})
    fig.savefig(OUT/"revision-binary-accuracy-time.svg",metadata={"Date":None})
    plt.close(fig)


def write_paper(prefix,pairs,times,budgets):
    def row_for(method,k):
        return next(item for item in prefix if item["method"]==method and item["draws"]==k)
    table=[]
    for m,t in zip(METHODS,times):
        vals=[row_for(m,k) for k in (1,5,20)]
        cost="/".join(str(k*STEPS[m]) for k in (1,5,20))
        sec=t["seconds_10000"]
        sec_fmt=("--" if sec is None else f"{sec:,.1f}".replace(",",r"\,") if sec>=100 else
                 f"{sec:.2f}" if sec>=1 else f"{sec:.3f}")
        latency=t["median_b1_ms"]
        latency_fmt=f"{latency:.1f}" if latency>=100 else f"{latency:.3f}"
        rate=t["structures_per_second"]
        rate_fmt=("--" if rate is None else f"{rate:,.0f}".replace(",",r"\,") if rate>=100 else f"{rate:.2f}")
        table.append(" & ".join([NAMES[m],str(STEPS[m]),cost]+[f'{100*x["mr"]:.1f}' for x in vals]+
                        [f'{x["rmse"]:.3f}' for x in vals]+[latency_fmt,sec_fmt,rate_fmt])+r" \\")
    gap=next(x for x in pairs if x["first"]=="exact" and x["second"]=="uflow" and x["draws"]==20)
    def mr(method,ms):
        return 100*next(x["mr"] for x in budgets if x["method"]==method and x["budget_ms"]==ms)
    raw=r"""\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3.8pt}
\caption{Raw recovery and CPU cost on the same 128 binary MP-20 targets. MR is the percentage of targets with a match. RMS is the mean lowest normalized RMS displacement among matched targets. Both are averaged over three local seeds or the single released checkpoint. $S$ and $kS$ are neural evaluations per draw and per target. The last three columns give median warm batch-one latency (ms), measured wall time for 10,000 structures (seconds), and the resulting throughput (structures per second) at batch 12.}
\label{tab:revision-binary}\smallskip
\begin{tabular}{@{}lrrrrrrrrrrr@{}}\hline
 & \multicolumn{2}{c}{Evaluations} & \multicolumn{3}{c}{MR (\%)} & \multicolumn{3}{c}{Best RMS} & \multicolumn{3}{c}{CPU performance}\\
Method & $S$ & $kS$ (1/5/20) & $k{=}1$ & $5$ & $20$ & $k{=}1$ & $5$ & $20$ & ms/draw & s/10k & struct./s\\\hline
ROWS
\hline\end{tabular}
\end{table}

Each method receives the same released cell composition (two species, two to eight active sites), with no space-group input. We retain the ordered raw proposals and all failures. The released 131-million-parameter uFlowCSP checkpoint trained on the wider MP-20 corpus \cite{uflow}. Its test partition contains all 128 target IDs. We preserve its original five draws and append 15 from an independent, fixed random stream. At 20 proposals uFlowCSP exceeds CrytalDrift by GAP percentage points (95\% paired target-bootstrap interval INTERVAL). The one-Euler flow measures the effect of truncating the compact flow's integration.

We timed the five methods on one Intel Core i7-12700H CPU using three pinned logical cores, three compute threads, FP32 neural inference and a warm service. Every call returns species, a canonical cell, fractional sites and Cartesian sites on the host. Retrieval caches eligible training donors by composition. Model loading, donor-index construction, retrospective matching, relaxation and ranking are outside the proposal timer. The 10,000-structure workload cycles through identical compositions in batches of 12 using local seed 11 or the released checkpoint. Other project jobs ran on separate CPU cores and shared system memory. For the serial curves we time individual completions up to 20 candidates per target and score those exact candidates afterwards. At 5\,ms the observed recovery rates were BUDGET\%. The released checkpoint reached RELEASEONE\% within one second and RELEASEFIVE\% within five seconds.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{revision-binary-accuracy-time.pdf}
\caption{Binary structure recovery by proposal count and wall time. Panel (a) uses prefixes of the saved benchmark draws, including the extended uFlowCSP replay. Panel (b) uses fresh, independently timed serial draws on the same IDs and measures elapsed time from the start of each target. Each method stops after at most 20 proposals. The log time axis spans both compact predictors and the released checkpoint.}
\label{fig:revision-binary-time}
\end{figure}
"""
    s=raw.replace("ROWS","\n".join(table)).replace("GAP",f'{-100*gap["mr_difference"]:.1f}').replace(
        "INTERVAL",f'$[{-100*gap["ci_high"]:.1f},{-100*gap["ci_low"]:.1f}]$')
    s=s.replace("BUDGET",", ".join(f'{mr(m,5):.1f}' for m in METHODS[:-1])+f' and {mr(METHODS[-1],5):.1f}').replace(
        "rates were","rates, in table order, were")
    s=s.replace("RELEASEONE",f'{mr("uflow",1000):.1f}').replace("RELEASEFIVE",f'{mr("uflow",5000):.1f}')
    pending=[NAMES[t["method"]] for t in times if t["seconds_10000"] is None]
    if pending:
        s=s.replace(r"\end{tabular}",r"\end{tabular}\par\smallskip\footnotesize "+
                    ", ".join(pending)+r" full-workload timing in progress.")
    (PAPER/"revision-benchmark.tex").write_text(s,encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quality-only",action="store_true")
    summarize(p.parse_args().quality_only)
