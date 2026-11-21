"""Combine measured per-seed warm generation latency with observed recovery prefixes."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import compare
from frontier import HERE, OUT, SEEDS, MILLISECONDS, collect, percentile_summary


STYLE = {'drift':('Pooled drift','#087369','o','-'), 'exact':('Exact-formula drift*','#653d9b','s','-'),
         'raw':('Raw-coordinate drift','#7ba4a0','P',':'), 'flow1':('Flow, 1 step','#827855','^','--'),
         'flow4':('Flow, 4 steps','#446587','v','--'), 'flow16':('Flow, 16 steps','#24477a','D','--'),
         'retrieval':('Training retrieval','#a34313','h','-')}


def get_timings():
    samples=defaultdict(list)
    for path in sorted((HERE/'speed-results').glob('timings-*.csv')):
        with path.open(encoding='utf8') as stream:
            for r in csv.DictReader(stream):
                if not r.get('wall_ms'):continue
                key=(r['cohort'],r['device'],r['method'],int(r['batch']),int(r['seed']))
                samples[key].append(float(r['wall_ms']))
    if not samples:raise FileNotFoundError('No completed timings under speed-results/')
    if any(not np.all(np.isfinite(v)) or min(v)<=0 for v in samples.values()):
        raise ValueError('Nonpositive or nonfinite timing')
    medians={k:np.median(v) for k,v in samples.items()}
    return medians


def timed_budget(hits,cost,budgets):
    """Return structures x budgets, first averaging seed-specific complete candidate draws."""
    cap=hits.shape[2]
    count=np.minimum(cap,np.floor(budgets[None,:]/cost[:,None]).astype(int))
    cumulative=np.pad(np.logical_or.accumulate(hits,axis=-1),((0,0),(0,0),(1,0)))
    recovery=np.stack([cumulative[i][:,c] for i,c in enumerate(count)])
    return count,recovery.mean(0)


def calculate(collection,medians):
    budget_rows,prefix_rows,first_rows,paired_rows=[],[],[],[]
    budgets=np.asarray(MILLISECONDS)
    arms={}
    boot_cache={}
    for (cohort,criterion,method),(ids,hits) in collection.items():
        profile='drift' if method in ('drift','raw','exact') else method
        batch_conditions=sorted({(dev,bt) for c,dev,m,bt,s in medians if c==cohort and m==profile})
        n=len(ids)
        if n not in boot_cache:
            boot_cache[n]=np.random.default_rng(741123).multinomial(n,np.full(n,1/n),size=10000).astype(float)
        for device,batch in batch_conditions:
            keys=[(cohort,device,profile,batch,seed) for seed in SEEDS]
            if any(k not in medians for k in keys):continue
            cost=np.asarray([medians[k] for k in keys])/batch
            ctx=dict(cohort=cohort,criterion=criterion,method=method,device=device,batch=batch,n_structures=n)
            count,score=timed_budget(hits,cost,budgets)
            arms[cohort,criterion,device,batch,method]=score
            p,lo,hi=percentile_summary(score,boot_cache[n])
            for j,b in enumerate(budgets):
                budget_rows.append({**ctx,'budget_ms':b,'recovery_fraction':p[j],'low':lo[j],'high':hi[j],
                    **{f'candidates_seed{seed}':count[i,j] for i,seed in enumerate(SEEDS)}})
            prefix=np.logical_or.accumulate(hits,axis=2)
            for j in range(hits.shape[2]):
                prefix_rows.append({**ctx,'candidates':j+1,'estimated_ms':(j+1)*cost.mean(),
                    'min_seed_ms':(j+1)*cost.min(),'max_seed_ms':(j+1)*cost.max(),
                    'recovery_fraction':prefix[:,:,j].mean()})
            for i,seed in enumerate(SEEDS):
                seen=hits[i].any(1);first=np.where(seen,hits[i].argmax(1)+1,hits.shape[2])
                first_rows.append({**ctx,'seed':seed,'observed_cap':hits.shape[2],
                    'median_ms_per_candidate':cost[i], 'matched_n':int(seen.sum()),
                    'missed_or_censored_n':int((~seen).sum()),'recovery_fraction':float(seen.mean()),
                    'mean_cost_until_hit_or_cap_ms':cost[i]*first.mean(),
                    'median_estimated_first_hit_ms_among_hits':cost[i]*np.median(first[seen]) if seen.any() else ''})
    for (cohort,criterion,device,batch,method),score in arms.items():
        if method=='exact':continue
        exact=arms.get((cohort,criterion,device,batch,'exact'))
        if exact is None:continue
        n=len(score)
        p,lo,hi=percentile_summary(exact-score,boot_cache[n])
        for j,b in enumerate(budgets):
            paired_rows.append(dict(cohort=cohort,criterion=criterion,contrast='exact minus '+method,device=device,
                batch=batch,budget_ms=b,n_structures=n,difference=p[j],low=lo[j],high=hi[j]))
    if not budget_rows:raise ValueError('No time profiles with all three fitted seeds')
    for filename,rows in [('fixed-time',budget_rows),('prefix-cost',prefix_rows),('time-to-first-summary',first_rows),
                          ('fixed-time-paired',paired_rows)]:
        compare.write_csv(OUT/(filename+'.csv'),rows)
    return prefix_rows


def plot(rows,device):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.68),sharey=True)
    fig.subplots_adjust(left=.092,right=.98,bottom=.33,top=.89,wspace=.13)
    for ax,cohort,title in zip(axes,('ab','multi'),('Two-site AB  ·  189 targets','Binary, 2–8 sites  ·  128 targets')):
        for method in STYLE:
            rr=[r for r in rows if r['device']==device and r['batch']==1 and r['criterion']=='structure' and
                r['cohort']==cohort and r['method']==method]
            if not rr:continue
            title_m,color,marker,line=STYLE[method]
            ax.plot([r['estimated_ms'] for r in rr],[100*r['recovery_fraction'] for r in rr],
                marker=marker,ms=4,color=color,lw=1.3,alpha=.95,ls=line,label=title_m,
                markeredgecolor='white',markeredgewidth=.35)
        ax.set_xscale('log')
        ax.set_xlim(.047,128)
        ax.set_title(title,loc='left',fontsize=10.2,fontweight='semibold')
        ax.grid(axis='y',color='#dce1e2',lw=.6)
        ax.grid(axis='x',color='#ebeeee',lw=.55)
        ax.spines[['right','top']].set_visible(False)
        ax.set_xlabel('Estimated serial generation time (ms)')
    axes[0].set_ylim(-2,102)
    axes[0].set_ylabel('Targets recovered (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.065),columnspacing=1.15,
               frameon=False,fontsize=7.9,handlelength=2.)
    fig.text(.5,.037,'* Exact-formula drift was added after the initial test comparison.',ha='center',
             color='#505b60',fontsize=7.3)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'quality-time-{device}.{ext}',dpi=205,facecolor='white')
    plt.close(fig)


def plot_both(rows):
    devices=sorted({r['device'] for r in rows})
    if devices!=['cpu','cuda']:return
    fig,axes=plt.subplots(2,2,figsize=(8.25,5.5),sharex=True,sharey=True)
    fig.subplots_adjust(left=.094,right=.985,bottom=.205,top=.927,wspace=.13,hspace=.12)
    for j,device in enumerate(devices):
        for ax,cohort,title in zip(axes[j],('ab','multi'),('Two-site AB  ·  189 targets','Binary, 2–8 sites  ·  128 targets')):
            for method,(name,color,marker,line) in STYLE.items():
                values=[r for r in rows if (r['device'],r['batch'],r['criterion'],r['cohort'],r['method'])==
                        (device,1,'structure',cohort,method)]
                if not values:continue
                ax.plot([r['estimated_ms'] for r in values],[100*r['recovery_fraction'] for r in values],label=name,
                    color=color,marker=marker,ms=3.8,ls=line,lw=1.2,markeredgecolor='white',markeredgewidth=.35)
            ax.set(xscale='log',xlim=(.047,128),ylim=(-2,102))
            ax.spines[['top','right']].set_visible(False);ax.grid(color='#e4e8e8',lw=.6)
            if j==0:ax.set_title(title,loc='left',fontsize=10,fontweight='semibold')
            if j==1:ax.set_xlabel('Estimated serial generation time (ms)')
            if cohort=='ab':ax.set_ylabel('Targets recovered (%)')
        axes[j,0].text(.96,.94,'CPU · 1 thread' if j==0 else 'RTX 3060 Laptop GPU',transform=axes[j,0].transAxes,
                       ha='right',va='top',fontsize=7.3,color='#526069')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,.04),
               columnspacing=1.1,frameon=False,fontsize=7.7,handlelength=2.)
    fig.text(.5,.017,'* Exact-formula drift was added after the initial test comparison.',ha='center',color='#505b60',fontsize=7.2)
    for ext in ('pdf','png'):fig.savefig(OUT/f'quality-time.{ext}',dpi=205,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--available',action='store_true')
    args=parser.parse_args()
    collection,pending=collect(args.available)
    medians=get_timings()
    rows=calculate(collection,medians)
    for device in sorted({r['device'] for r in rows}):
        if any(r['device']==device and r['batch']==1 and r['criterion']=='structure' for r in rows):plot(rows,device)
    plot_both(rows)
    compare.write_json(OUT/'time-metadata.json',dict(
        pending_matcher_inputs=pending,timing_files={p.name:compare.digest(p) for p in (HERE/'speed-results').glob('timings-*.csv')},
        fixed_time='Per-seed median measured batch call divided by batch. Prefix lengths are floor(budget/cost), bounded by observed draws. Average over seeds, then targets.',
        prefix_cost='An x-coordinate is prefix length times the mean of the three per-seed median candidate times. Each point shows a directly observed prefix.',
        batch_one='Serial generation of independent draws. Reference matching time, relaxation, model loading, and lookup-index setup excluded.',
        larger_batches='Batch call time divided by size is per-output throughput cost only when the full measured batch is occupied with independent work. Not a single-request latency.',
        censoring='First-hit cost is derived from the first matching observed draw and the timing profile. Failures are censored at the last observed draw.',
        tested_only='No assumption of independent draws or extrapolation beyond 5 matcher draws; secondary environment cap is 20 AB and 5 broad.',
        exact_arm='Added after original test comparison; validation replication compares environment energy only.'))
    print('Wrote fixed-time results and plots for',sorted({r['device'] for r in rows}))


if __name__=='__main__':
    main()

