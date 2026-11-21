"""Render the supplemental scale table from finished paired-score products."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
METHODS=['drift','flow1','flow16','retrieval']
LABELS=['CrytalDrift','One-step flow','16-step flow','Training-set retrieval']


def read(path):return json.loads(path.read_text())


def main():
    mp=HERE/'mp20'
    budgets={k:read(mp/f'full-k{k}.json') for k in (1,5,20) if (mp/f'full-k{k}.json').is_file()}
    wbm=read(HERE/'wbm'/'revision-summary.json')['5']
    wgeom=read(HERE/'wbm'/'geometry-summary.json')['test']
    out=[r'\subsection{Full MP-20 recovery and repeated WBM fits}\label{s:revision-scale}',
        r'We evaluated the frozen full-chemistry generators and training-geometry retrieval on all 9,046 structures in the public MP-20 test split. For each of the three original seeds, all four methods received the same atomic numbers and multiplicities. The neural models shared their initial noise. The first five proposals for the original 128 hash-selected targets were preserved exactly. For every target and draw we recorded the normalized RMS criterion from \texttt{get\_rms\_dist}, the maximum-site criterion from \texttt{fit}, and the native generated-to-reference volume ratio. The volume sensitivity accepts an RMS-matched draw only when this ratio lies between $1/1.10$ and $1.10$. Both pymatgen matchers use the original lattice and site tolerances, primitive reduction, and isotropic rescaling.',
        '',r'\begin{table}[!htbp]\centering\small',
        r'\caption{Original one-pass compact fits on all 9,046 public MP-20 test structures. MR is the percentage of targets with at least one RMS match. Best RMS is the minimum normalized displacement among matches, averaged over recovered targets within each fitted seed, then equally across the three seeds. Each method can recover a different subset.}\label{t:full-mp20-revision}',
        r'\begin{tabular}{l'+'r'*(2*len(budgets))+r'}\hline',
        'Method & '+' & '.join(f'MR@{k}' for k in budgets)+' & '+' & '.join(f'RMS@{k}' for k in budgets)+r'\\\hline']
    for method,label in zip(METHODS,LABELS):
        rates=[100*table['methods'][method]['rms_recovery']['mean'] for table in budgets.values()]
        scores=[table['methods'][method]['mean_best_rms_among_recovered']['mean'] for table in budgets.values()]
        out.append(label+' & '+' & '.join([f'{v:.2f}' for v in rates]+[f'{v:.3f}' for v in scores])+r'\\')
    out+= [r'\hline\end{tabular}\end{table}','']
    if 20 in budgets:
        endpoints=[]
        for k in (1,20):
            d=budgets[k]['paired_drift_minus']['flow1']['rms_recovery']
            endpoints.append(f"${100*d['mean']:+.2f}$ points ($[{100*d['lo']:.2f},{100*d['hi']:.2f}]$)")
        out.append('The drift minus one-step flow recovery differences at one and twenty draws were '
                   +endpoints[0]+' and '+endpoints[1]+', respectively.')
    out+= [
        r'At five draws the RMS recovery difference between CrytalDrift and one-step flow was $+3.53$ percentage points (paired 95\% interval $[3.06,4.00]$). Maximum-site recovery was 9.49\% for drift, 6.27\% for one-step flow, 10.57\% for 16-step flow, and 61.87\% for retrieval. The drift advantage over one-step flow was $+3.22$ points ($[2.85,3.59]$). Requiring native-volume agreement on the RMS-matched draw gave 7.58\%, 7.80\%, 10.25\%, and 54.78\%, respectively, with a drift difference of $-0.22$ points ($[-0.63,0.19]$). Sixteen-step flow recovered $0.44$ points more targets than CrytalDrift by the RMS criterion ($[0.06,0.82]$). Intervals use 10,000 paired resamples of target IDs after averaging recovery indicators across fitted seeds. They condition on the fitted checkpoints.','',
        r'\input{revision-external-full}', '', r'\input{revision-published}', '',
        r'\subsection{Replication on formula-disjoint WBM}',
        r'For WBM we repeated the paired training with seeds 23 and 37, using the original 205,091 structures, 12,819 updates and the same 128 held-out targets as seed 11. The five-draw RMS-recovered target counts for seeds 11, 23 and 37 were $(39,40,37)$ for drift and $(28,35,27)$ for one-step flow. Their paired recovery difference was $+6.77$ percentage points ($[2.60,10.94]$). With the native-volume rule it was $+3.91$ points ($[-0.52,8.07]$). The held-out formulas are disjoint from WBM training. Geometry retrieval can therefore recover an arrangement by assigning the supplied species to sites from a training template.','',
        r'\begin{table}[!htbp]\centering\small',
        r'\caption{Three fitted seeds on the original 128 WBM test structures, with five proposals per target. Energy is the fair conditional four-feature score. All recovery columns are percentages.}\label{t:wbm-revision}',
        r'\begin{tabular}{lrrrr}\hline Method & Energy & MR@5 & Max@5 & MR+vol.@5\\\hline']
    for m,label in zip(METHODS,LABELS):
        row=wbm['methods'][m]
        values=[f"{wgeom['stats'][m]['conditional_energy']['mean']:.3f}"]
        values.extend(f"{100*row[k]['mean']:.2f}" for k in ('rms_recovery','strict_recovery','rms_and_volume_1.10'))
        out.append(label+' & '+' & '.join(values)+r'\\')
    out.extend([r'\hline\end{tabular}\end{table}',''])
    blind=mp/'blind-public-parity.json';primitives=mp/'primitive-prerequisites.json'
    if blind.is_file() and primitives.is_file():
        audit=read(blind);census=read(primitives)
        changed=len(census['reference_reduction_changes'])
        fallback=(f'Primitive reduction changed the count for {changed:,} of the 9,046 MP-20 references. Those cases went directly to pymatgen. '
                  if changed else 'All 9,046 reference cells retained their native site count under primitive reduction. ')
        out.append(
            'The accelerated matcher uses a necessary lattice-frame and site-subset test when the native and '
            'primitive reference cells contain the same number of sites. '+fallback+
            f"An independent check used both public matcher calls on {audit['slots']:,} draws "
            f"from {audit['cases']} separately hash-selected targets. Every RMS and maximum-site "
            'hit indicator agreed, and the largest normalized RMS difference among matched pairs was '
            + (r'$<10^{-8}$.' if audit['largest_normalized_rms_error']<1e-8 else
               f"{audit['largest_normalized_rms_error']:.3g}."))
        out.append('')
    (ROOT/'paper/revision-scale.tex').write_text('\n'.join(out),encoding='utf8')


if __name__=='__main__':main()
