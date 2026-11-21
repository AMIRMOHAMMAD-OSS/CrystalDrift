"""Draw the manuscript comparison from cached, structure-level bootstrap results."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OUT = Path(__file__).resolve().parent / 'comparison'
LABELS = {'drift':'Drift, pooled references', 'exact':'Drift, exact formula*', 'raw':'Raw-coordinate drift', 'flow1':'Flow, 1 step',
          'flow4':'Flow, 4 steps', 'flow16':'Flow, 16 steps', 'retrieval':'Training retrieval'}
COLORS = {'drift':'#09665f', 'exact':'#553a91', 'raw':'#64938d', 'flow1':'#647183',
          'flow4':'#647183', 'flow16':'#647183', 'retrieval':'#aa5719'}
ORDER = {'ab':('drift','exact','raw','flow1','flow4','flow16','retrieval'),
         'multi':('drift','exact','flow1','flow16','retrieval')}
PANELS = [('energy','Conditional energy score',1.),
          ('short_contact_or_failure','Short-contact rate (%)',100.)]


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(8.1,5.2),sharey=True)
    fig.subplots_adjust(left=.28,right=.975,top=.91,bottom=.17,wspace=.21)
    for cohort, methods in ORDER.items():
        with (OUT/f'{cohort}-groups.csv').open(encoding='utf-8') as f:
            rows = {(r['method'],r['metric']):r for r in csv.DictReader(f) if r['group']=='all'}
        with (OUT/f'{cohort}-per-seed.csv').open(encoding='utf-8') as f:
            sd = list(csv.DictReader(f))
        start=1.1 if cohort=='ab' else 9.45
        for j,method in enumerate(methods):
            y=start+j
            for ax,(metric,heading,scale) in zip(axes,PANELS):
                row=rows[method,metric]
                average,low,high=(float(row[c])*scale for c in ('value','low','high'))
                ax.plot([low,high],[y,y],linewidth=2.3,color=COLORS[method],solid_capstyle='round',zorder=3)
                ax.scatter(average,y,s=25,color=COLORS[method],edgecolor='white',linewidth=.55,zorder=4)
                for seed_row in sd:
                    if seed_row['method']==method:
                        ax.scatter(float(seed_row[metric])*scale,y+.20,marker='|',s=16,
                                   linewidths=.7,color=COLORS[method],alpha=.8,zorder=2)
    ypos=[1.1+i for i in range(7)]+[9.45+i for i in range(5)]
    for ax,(_,title,_) in zip(axes,PANELS):
        ax.set_title(title,loc='left',fontsize=10.2,fontweight='semibold',pad=8)
        ax.set_yticks(ypos, [LABELS[m] for methods in ORDER.values() for m in methods])
        ax.set_ylim(14.05,-.42)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(axis='y',length=0,pad=7)
        ax.tick_params(axis='x',length=3,color='#8b949b',labelsize=8.3)
        ax.grid(axis='x',color='#e2e6e8',linewidth=.65,zorder=0)
        for spine in ('top','right','left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#a7adb0')
        ax.axhline(7.88,color='#c5cdd0',lw=.65)
    axes[0].set_xlim(left=-.06)
    axes[1].set_xlim(left=-.12)
    for y,txt in ((.06,'Two-site AB'),(8.42,'Binary, 2–8 sites')):
        axes[0].text(-.035,y,txt,ha='right',va='center',transform=axes[0].get_yaxis_transform(),
                     fontweight='semibold',color='#202830',fontsize=9.4,clip_on=False)
    fig.text(.08,.053,'Dots show means, bars show 95% intervals across test structures, and ticks show individual seeds.',
             ha='left',va='bottom',fontsize=7.5,color='#47515a')
    fig.text(.08,.022,'* Exact-formula drift was added after the original test comparison and checked on the separate validation split.',
             ha='left',va='bottom',fontsize=7.5,color='#47515a')
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'crystal-comparison.{ext}',dpi=190,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
