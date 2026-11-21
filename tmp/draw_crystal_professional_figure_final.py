from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgb
plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'stix','svg.fonttype':'none','axes.unicode_minus':False})
root=Path.cwd()
out=root/'CrystalDrift_Figure1_Design'/'Figure1_Periodic_Drift.svg'
W,H=1460,949
fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0),aspect='equal')
ax.set_axis_off()
ink='#1B232C'; navy='#183963'; blue='#336CC0'; teal='#137A7D'; orange='#C55737'; purple='#8065A3'; gray='#687482'; lite='#CCD4D9'; atomgray='#64747D'
def t(x,y,s,size=20,col=ink,weight=None,ha='left',z=9):
    return ax.text(x,y,s,fontsize=size*.72,color=col,fontweight=weight,ha=ha,va='center',zorder=z,clip_on=False)
def eq(x,y,s,size=25,col=ink,ha='center'):
    return t(x,y,'$'+s+'$',size,col,ha=ha)
def line(x,y,u,v,c=ink,w=1.4,ls='-',z=2):
    ax.plot([x,u],[y,v],color=c,lw=w*.72,ls=ls,solid_capstyle='round',zorder=z)
def arr(x,y,u,v,c=blue,w=2.4,ms=13,z=5):
    ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=ms,linewidth=w*.72,color=c,shrinkA=0,shrinkB=0,zorder=z))
def poly(points,edge,fill='none',w=2.2,z=3):
    ax.add_patch(Polygon(points,closed=True,facecolor=fill,edgecolor=edge,lw=.72*w,joinstyle='round',zorder=z))
def atom(x,y,r,c=atomgray,z=5):
    base=np.array(to_rgb(c))
    ax.add_patch(Circle((x,y),r,fc=base*.69,ec='white',lw=.6,zorder=z))
    for k in range(10):
        a=k/10
        ax.add_patch(Circle((x-r*(.03+.24*a),y-r*(.04+.29*a)),r*(.91-.77*a),fc=base*(.88+.10*a)+(1-base)*(.035+.48*a*a),ec='none',zorder=z+.02+k*.004))
def ring(x,y,r,c=gray,z=6):
    ax.add_patch(Circle((x,y),r,fc='white',ec=c,lw=1.15,zorder=z))
def box(x,y,w,h,s,size=26):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',fc='white',ec=navy,lw=1.65,zorder=1))
    eq(x+w/2,y+h/2,s,size,navy)
def head(x,y,k,s):
    t(x,y,k,33,navy,'bold'); t(x+42,y,s,29,ink,'semibold')
def dot(x,y,r,c,z=4):ax.add_patch(Circle((x,y),r,fc=c,ec='white',lw=.75,zorder=z))
def diamond(x,y,r,c):
    poly([(x-r,y),(x,y-r),(x+r,y),(x,y+r)],c,'white',2.2,5)
# A. Two bases, and a physical perturbation as a control.
head(30,39,'a','One crystal, two lattice bases')
o=np.array([119.,262.]); step=82.
for i in range(-1,4):
    x=o[0]+i*step
    if 25<x<365: line(x,95,x,310,lite,1.05,(0,(2,4)))
for j in range(-1,4):
    y=o[1]-j*step
    if 90<y<315:line(33,y,359,y,lite,1.05,(0,(2,4)))
def grid(i,j): return o+np.array([step*i,-step*j])
sq=[grid(0,0),grid(1,0),grid(1,1),grid(0,1)]
sh=[grid(0,0),grid(1,0),grid(2,1),grid(1,1)]
poly(sq,teal,'#EDF7F6',3,2.1);poly(sh,orange,(1,.84,.76,.32),3,2.2)
for i in range(-1,4):
 for j in range(-1,4):
    for di,dj,c,rr in [(0,0,atomgray,7.8),(.5,.5,'#AC5855',5.8)]:
      x,y=grid(i+di,j+dj)
      if 28<x<360 and 89<y<315:atom(x,y,rr,c)
eq(159,283,r'\mathbf{a}',23,teal);eq(103,219,r'\mathbf{b}',24,teal)
eq(202,126,r'\mathbf{a}+\mathbf{b}',23,orange)
line(201,141,175,205,orange,1)
line(45,341,70,341,teal,3); t(78,341,'Original cell',18)
line(194,341,218,341,orange,3);t(226,341,'New basis',18)
# Two matched affinity diagrams, following the raw-to-invariant visual comparison.
def micro(x,y,skew,col):
    ss=14 if skew else 16.1
    dx=13 if skew else 0
    pts=[(x,y+ss/2),(x+ss,y+ss/2),(x+ss+dx,y-ss/2),(x+dx,y-ss/2)]
    poly(pts,col,'white',1.7,3)
for row,y,method,values in [(0,149,'Raw lattice entries',('1.00','0.21')),(1,271,'Periodic features',('0','> 0'))]:
    t(384,y-57,method,21,weight='bold')
    t(701,y-58,'Distance',15,gray,ha='right')
    for j,(dy,lab,clr) in enumerate([(-20,'Rebased',orange),(22,'Expanded',purple)]):
        strong=(row==0 and j==1) or (row==1 and j==0)
        arr(397,y,485,y+dy,blue if strong else '#AEB9C6',4.3 if strong else 1.15,13 if strong else 7)
        micro(493,y+dy,j==0,clr)
        t(534,y+dy,lab,17)
        t(697,y+dy,values[j],20,navy if strong else gray,'bold' if strong else 'normal',ha='right')
    dot(397,y,6,navy)
line(384,201,699,201,lite,1)
t(384,343,'Thicker arrow = higher kernel affinity.',17,gray)
box(35,380,662,58,r"L'=UL,\qquad F'=FU^{-1}\ \mathrm{mod}\ 1,\qquad U\in\mathrm{GL}(3,\mathbb{Z})",24)
# B. An image across the boundary is a physical neighbor.
head(759,39,'b','Building an invariant crystal descriptor')
left,top,s=782,111,149
poly([(left,top),(left+s,top),(left+s,top+s),(left,top+s)],'#919EA8','#F8FAFB',1.65,1)
for i in [0,1]:line(left+s,top+i*s,left+s+82,top+i*s,lite,1.35,(0,(3,4)),1)
i=np.array([897.,204.]); jj=np.array([806.,167.]); j1=jj+[s,0]; k=np.array([858.,143.])
ax.add_patch(Circle(i,80,fc='none',ec='#C4CFD9',lw=.9,linestyle=(0,(2.5,3.5)),zorder=1))
line(*i,*j1,blue,3);line(*i,*k,blue,3)
for pos,c,rr in [(jj,'#AC5855',8.5),(j1,'#AC5855',8.5),(k,atomgray,9.5),(i,atomgray,10.5)]:atom(*pos,rr,c,4)
# Angular interval in screen coordinates between vectors to k and j1.
ax.add_patch(Arc(i,53,53,theta1=237,theta2=328,ec=orange,lw=1.4,zorder=4))
eq(902,168,r'\theta',21,orange);eq(889,223,'i',21);eq(802,144,'j',21)
eq(977,147,r'j+\mathbf{a}',22,blue);eq(845,124,'k',21)
t(786,300,'Periodic neighbors',18,gray)
t(1036,102,'Species-resolved geometry',21,weight='bold')
eq(1036,145,r'd_{ij\mathbf{n}}=\|\mathbf{r}_{ij\mathbf{n}}\|,\quad \cos{\theta_{jik}}',23,ha='left')
t(1036,186,'Pair shells and angular powers',19)
eq(1036,217,r'\ell=0,1,2',22,navy,ha='left')
t(1036,263,'Optional descriptor channels',20,weight='bold')
t(1036,293,'Inter-orbit contacts or',19)
t(1036,321,'reciprocal-space intensities',19)
eq(888,347,r'\mathbf{r}_{ij\mathbf{n}}=(f_j-f_i+\mathbf{n})L',22,ha='center')
box(766,380,658,58,r'\Phi(gX)=\Phi(X),\qquad K_T(gX,hY)=K_T(X,Y)',25)
# C. The training-only field, drawn in a schematic feature plane.
head(30,501,'c','Drifting in periodic feature space')
t(40,556,'One training composition',19,gray)
# Schematic bank points: 2 references for each sign. Equal-weight cluster centroids lie along drawn direction.
for x,y in [(96,720),(71,752)]:diamond(x,y,8,orange)
for x,y in [(239,612),(270,629)]:dot(x,y,9,teal)
t(46,787,'Generated references',17,orange)
t(223,579,'Observed references',17,teal,ha='center')
qx,qy=161,694
arr(qx,qy,225,651,blue,3.8,16);dot(qx,qy,9,'#223E60')
eq(222,680,'V_i',25,blue);eq(qx-18,qy-14,'x_i',24,navy)
t(43,826,'Schematic projection of feature space',16,gray)
t(338,557,'Kernel affinity',20,weight='bold')
eq(338,597,r'K_T(X,Y)=e^{-\|\Phi(X)-\Phi(Y)\|_2/T}',23,ha='left')
t(338,652,'Attraction and repulsion',20,weight='bold')
eq(338,694,r'D_i^T=m_i^+m_i^-(\mu_i^+-\mu_i^-)',24,ha='left')
t(338,737,'Weighted centroids of observed (+)',17,gray)
t(338,763,'and generated (-) references.',17,gray)
t(338,805,'Normalize each field and average over T.',17)
t(338,832,'sg holds the training target fixed.',16,gray)
box(35,858,662,64,r'\mathcal{L}(\theta)=\frac{1}{2M}\sum_i\|x_i-\mathrm{sg}(x_i+V_i)\|_2^2',27)
# D. Explicit one-step inference on a full general inversion orbit in 3D.
head(759,501,'d','One-step generation on Wyckoff orbits')
eq(766,554,r'AB_2',25,navy,ha='left')
arr(821,554,864,554,blue,2.3,10)
t(880,554,'Occupied pattern from the training catalog',19)
eq(765,594,r'H=P\overline{1}',23,navy,ha='left')
atom(894,594,7.3,'#AC5855',5);t(909,594,'A',19,weight='bold');eq(930,594,r'1h\ \mathrm{(fixed)}',21,ha='left')
atom(1083,594,7.3,'#4974A9',5);t(1099,594,'B',19,weight='bold');eq(1121,594,r'2i:\ \{q,-q\}\ \mathrm{mod}\ 1',22,ha='left')
def cell(origin,avec,bvec,cvec,coords,after=False):
 base=np.asarray(origin); mat=np.array([avec,bvec,cvec],float)
 def v(fr):return base+np.array(fr)@mat
 for fr,j in [((0,0,0),0),((0,0,0),1),((0,0,0),2),((1,0,0),1),((1,0,0),2),((0,1,0),0),((0,1,0),2),((0,0,1),0),((0,0,1),1),((1,1,0),2),((1,0,1),1),((0,1,1),0)]:
    p=v(fr);e=list(fr);e[j]=1;end=v(e); hidden=fr[1]==1 and fr[2]==0
    line(*p,*end,'#AAB7C3' if hidden else '#60788D',1.3 if hidden else 1.8,(0,(3,3)) if hidden else '-',2)
 one,two=v(coords),v(1-np.array(coords))
 line(*one,*two,'#9BB0BF',1.5,(0,(2,3)),3)
 cc=v([.5,.5,.5]);atom(*cc,7.5,'#AC5855',6.5)
 for j,pos in enumerate((one,two)):
    if after:
      q0=np.array([.21,.26,.31]);old=v(q0 if j==0 else 1-q0)
      ring(*old,5.5,'#9AA8B5',4)
      direction=(pos-old)/np.linalg.norm(pos-old);end=pos-8*direction
      arr(*(old+2*direction),*end,orange,2.3,9,5)
    atom(*pos,8.3,'#4974A9',6.6)
 return one,two
p1,p2=cell((806,764),(137,0),(-29,-36),(16,-106),[.21,.26,.31])
eq(p1[0]-15,p1[1]+17,'q',21,navy);eq(p2[0]+17,p2[1]-8,'-q',21,navy)
p1,p2=cell((1223,767),(147,-3),(-38,-29),(24,-112),[.19,.58,.42],True)
eq(p1[0]-17,p1[1]-13,"q'",21,navy);eq(p2[0]+21,p2[1]+17,"-q'",21,navy)
arr(966,707,1166,707,blue,2.6,13);ax.add_patch(FancyBboxPatch((1001,684),134,46,boxstyle='round,pad=0,rounding_size=9',fc='white',ec=navy,lw=1.45,zorder=7));eq(1068,708,r'G_\theta',27,navy)
t(1068,748,'One network call',16,gray,ha='center')
eq(866,800,r'(L_0,q)',23,ha='center');eq(1292,800,r"(L',q')",23,ha='center')
t(1083,828,'One free position determines both B sites.',18,ha='center')
box(766,858,658,64,r"G_\theta(gX_0)=gG_\theta(X_0),\qquad H\subseteq\mathrm{Sym}\!\left(G_\theta(X_0)\right)",23)
fig.savefig(out,format='svg',facecolor='white',metadata={'Title':'Periodic drift and Wyckoff-constrained crystal generation','Description':'Four-panel methods figure. Periodic representations define a training-only feature-space drift. A one-step orbit decoder transports an inversion-related pair and the lattice.'})
plt.close(fig)
print(out.resolve(),out.stat().st_size)

