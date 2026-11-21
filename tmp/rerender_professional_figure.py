from pathlib import Path
source=Path('tmp/draw_crystal_professional_figure_final.py')
s=source.read_text(encoding='utf-8-sig')
a=s.index('fig.savefig(out,format=')
s=s[:a]+"""matplotlib.rcParams['pdf.fonttype']=42
fig.savefig(out.with_suffix('.pdf'),format='pdf',facecolor='white',metadata={'Title':'Periodic drift and Wyckoff-constrained crystal generation'})
fig.savefig(out.with_suffix('.png'),dpi=3600/fig.get_figwidth(),facecolor='white')
fig.savefig(root/'tmp/Figure1_Periodic_Drift_Preview.png',dpi=1450/fig.get_figwidth(),facecolor='white')
plt.close(fig)
"""
exec(compile(s,str(source),'exec'))
