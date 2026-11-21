"""Draw the first multielement restricted-screen pass and miss in the frozen relaxation panel."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle,Polygon
import numpy as np
from data import ELEMENTS,unpack_gram

HERE=Path(__file__).resolve().parent
D=HERE.parent/"denovo";S=D/"screen"
# Colors are shared across examples. Radii are plotting radii in angstroms.
STYLE={16:("#d7911a",.31),38:("#138779",.41),82:("#7259a6",.39),
       24:("#316da4",.34),25:("#b74461",.36),31:("#4a9178",.33)}
ORIGIN=np.array([.10317,.21843,.31791])
CORNERS=np.array(list(itertools.product((0,1),repeat=3)),float)
EDGES=[(i,j) for i,j in itertools.combinations(range(8),2) if abs(CORNERS[i]-CORNERS[j]).sum()==1]


def camera(cells,sites):
    # One orientation per raw/relaxed pair. Choose it solely to separate projected centers.
    best=None
    for el in (12,20,28,36):
        for az in range(7,360,12):
            a,b=np.radians([az,el])
            direction=np.array([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)])
            side=np.array([-np.sin(a),np.cos(a),0]);up=np.cross(direction,side)
            m=np.stack([side,up,direction],axis=-1)
            scores=[]
            for l,f in zip(cells,sites):
                p=(f@l)@m
                v=(CORNERS@l)@m
                ij=np.triu_indices(len(f),1)
                separ=np.linalg.norm(p[ij[0],:2]-p[ij[1],:2],axis=1)
                shape=np.ptp(v[:,:2],axis=0)
                scores.append(np.quantile(np.clip(separ,0,1.15),.05)+.35*min(shape)/max(shape))
            candidate=(min(scores),az,el,m)
            if best is None or candidate[0]>best[0]:best=candidate
    score,az,el,m=best
    # Align the long image-plane axis with the wider direction of the figure panel.
    footprint=np.concatenate([(CORNERS-.5)@l@m[:,:2] for l in cells])
    _,axes=np.linalg.eigh(footprint.T@footprint)
    long_axis=axes[:,-1]
    if long_axis[0]<0:long_axis=-long_axis
    roll=np.column_stack((long_axis,[-long_axis[1],long_axis[0]]))
    m[:,:2]=m[:,:2]@roll
    return score,az,el,m


def contacts(l,f,z):
    # Use periodic neighborhoods to determine contact thresholds. Show links only between displayed sites.
    bounds=5*np.linalg.norm(np.linalg.inv(l),axis=0)
    nearest=np.full(len(z),np.inf);entries=[]
    for i in range(len(z)):
        for j in range(len(z)):
            if z[i]==z[j]:continue
            d=f[j]-f[i]
            spans=[range(int(np.ceil(-t-a)),int(np.floor(t-a))+1) for t,a in zip(bounds,d)]
            for shift in itertools.product(*spans):
                frac=d+shift;distance=np.linalg.norm(frac@l)
                if 1e-8<distance<5:
                    nearest[i]=min(nearest[i],distance)
                    entries.append((i,j,shift,frac,distance))
    out=[]
    for i,j,t,frac,distance in entries:
        if distance>1.18*min(nearest[i],nearest[j])+1e-7:continue
        if any(t) or j<i:continue
        out.append(np.stack([f[i],f[i]+frac])@l)
    return out


def rgba_mix(c,target,amount):
    return (1-amount)*np.array(colors.to_rgb(c))+amount*np.array(target)


def sphere(ax,x,y,r,color,depth):
    ax.add_patch(Circle((x,y),r,facecolor=rgba_mix(color,(0,0,0),.30),
                        edgecolor="#283339",linewidth=.25,zorder=depth))
    # Shift nested opaque vector circles toward a single light source.
    for i in range(14):
        t=(i+1)/15
        tint=rgba_mix(color,(1,1,1),.04+.56*t**2)
        ax.add_patch(Circle((x-.24*r*t,y+.30*r*t),r*(.97-.73*t),
                            facecolor=tint,edgecolor="none",zorder=depth+.02+i*.001))


def geometry(ax,l,f,z,m,center,limits,scale_bar):
    v=(CORNERS@l-center)@m;p=(f@l-center)@m
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_aspect("equal");ax.axis("off")
    faces=[]
    for axis in range(3):
        for end in (0,1):
            ids=np.flatnonzero(CORNERS[:,axis]==end)[[0,1,3,2]]
            faces.append((v[ids,2].mean(),ids))
    for depth,indices in sorted(faces,key=lambda q:q[0])[:3]:
        ax.add_patch(Polygon(v[indices,:2],closed=True,facecolor="#e9f3f2",edgecolor="none",alpha=.60,zorder=-1))
    plane=CORNERS[np.flatnonzero(CORNERS[:,0]==0)[[0,1,3,2]]].copy();plane[:,0]=.5
    plane=(plane@l-center)@m
    ax.add_patch(Polygon(plane[:,:2],closed=True,facecolor="none",edgecolor="#c3cecf",linestyle=":",linewidth=.62,zorder=0))
    for i,j in EDGES:
        u=v[[i,j]];back=u[:,2].mean()<0
        ax.plot(u[:,0],u[:,1],ls=(0,(2,2)) if back else "-",lw=.70 if back else .94,
                color="#aab5b8" if back else "#74848a",zorder=0)
    links=contacts(l,f,z)
    for line in links:
        u=(line-center)@m;ax.plot(u[:,0],u[:,1],lw=1.00,color="#a2afb3",zorder=.5,solid_capstyle="round")
    for j in np.argsort(p[:,2]):
        c,r=STYLE[int(z[j])]
        sphere(ax,*p[j,:2],r,c,1+float(p[j,2]-p[:,2].min())*.3)
    if scale_bar:
        x=limits[0,0]+.26;y=limits[1,0]+.16
        ax.plot([x,x+2],[y,y],color="#48545a",lw=1.2)
        for xx in (x,x+2):ax.plot([xx,xx],[y-.035,y+.035],color="#48545a",lw=1)
        ax.text(x+1,y+.07,"2 Å",ha="center",va="bottom",fontsize=6.6,color="#48545a")
    return len(links)


def dimensions(l):
    g=l@l.T;a=np.sqrt(g.diagonal());corr=g/np.outer(a,a)
    b=np.degrees(np.arccos(np.clip(corr[[1,0,0],[2,2,1]],-1,1)))
    return a,b


def fmt_formula(z):
    # Use conventional cation-first reading for the two reported examples. Preserve cell stoichiometry for other arms.
    order=(38,82,25,24,31,16)
    if not set(z)<=set(order):order=sorted(set(z))
    return "".join(ELEMENTS[q]+(f"$_{{{int((z==q).sum())}}}$" if (z==q).sum()>1 else "") for q in order if q in z)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arm",default="contacts-small");ap.add_argument("--stem",default="crystal-gallery");settings=ap.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","pdf.fonttype":42,"font.size":9,
                        "axes.spines.top":False,"axes.spines.right":False})
    arm=settings.arm
    rows={r["id"]:r for r in csv.DictReader((S/"per-candidate.csv").open()) if r["arm"]==arm}
    panel=json.loads((S/"panel.json").read_text())["indices_zero_based"]
    with np.load(D/f"pools/{arm}.npz") as d:
        idxs=[i for i in panel if len(np.unique(d["z"][i][d["z"][i]>0]))>=2]
        chosen=[next(i for i in idxs if rows[f"{arm}-{i+1:05}"]["energetic_pass"]==success)
                for success in ("True","False")]
        metadata=[(str(d["chemistry_source"][i]),str(d["donor"][i]),str(d["cell_donor"][i]),int(d["group"][i])) for i in chosen]
        others=sorted({int(z) for i in chosen for z in d["z"][i] if z>0 and z not in STYLE})
        palette=("#b2912c","#6778a8","#c67a49","#458dab","#72557d","#557050","#939e45")
        for j,q in enumerate(others):STYLE[q]=(palette[j%len(palette)],.34)
    fig=plt.figure(figsize=(7.65,6.45),facecolor="white");record=[]
    for rr,(i,(composition,donor,cdonor,sg)) in enumerate(zip(chosen,metadata)):
        hi=.977-.497*rr;label=f"{arm}-{i+1:05}";row=rows[label]
        x=json.loads((S/f"inputs/{label}.json").read_text());r=json.loads((S/f"records/{label}.json").read_text())
        z=np.array(x["z"]);fr=[(np.asarray(p)+ORIGIN)%1 for p in (x["f"],r["final_f"])]
        L=[np.linalg.cholesky(unpack_gram(k)) for k in (x["k"],r["final_k"])]
        # Adjacent complete native cells reveal periodic coordination while preserving every drawn measurement.
        repeat=np.array([2,1,1])
        shown_L=[np.diag(repeat)@l for l in L]
        shown_f=[np.concatenate([f/repeat,(f+np.array([1,0,0]))/repeat]) for f in fr]
        shown_z=np.tile(z,2)
        best,az,el,m=camera(shown_L,shown_f)
        centers=[np.ones(3)*.5@l for l in shown_L]
        envelope=np.concatenate([(CORNERS@l-center)@m for l,center in zip(shown_L,centers)])[:,:2]
        ext=np.max(abs(envelope),axis=0)+.68;limits=np.array([[-v,v] for v in ext])
        limits[1,0]-=.60
        formula=fmt_formula(z);passed=(rr==0)
        accent="#17675e" if passed else "#984733"
        fig.text(.025,hi,"AB"[rr],fontsize=10.3,weight="bold",color=accent,va="top")
        fig.text(.055,hi,formula,fontsize=14.5,weight="bold",color="#243238",va="top")
        fig.text(.975,hi-.002,("Pass" if passed else "Above threshold")+"  ·  "+
                 f"{1000*float(row['above_augmented_hull']):.1f} meV/atom",ha="right",va="top",
                 color=accent,fontsize=11,weight="semibold")
        fig.text(.025,hi-.048,f"Draw {i+1:,}     Template donor {donor}     Cell donor {cdonor}     Group {sg}",
                 color="#425059",fontsize=8.25)
        for j,q in enumerate(sorted(set(z))):
            fx=.027+.075*j;fy=hi-.076;fig.add_artist(Line2D([fx],[fy],marker="o",ms=6.2,color=STYLE[q][0],
                                        markeredgecolor="white",markeredgewidth=.45,transform=fig.transFigure))
            fig.text(fx+.013,fy,ELEMENTS[q],ha="left",va="center",fontsize=8.4,color="#263238")
        notes=[]
        for c,(l,f) in enumerate(zip(L,fr)):
            xmid=.284+.447*c
            fig.text(xmid,hi-.102,"As generated" if c==0 else "After CHGNet relaxation",
                     ha="center",va="top",fontsize=9.5,weight="semibold",color="#384851")
            ax=fig.add_axes([xmid-.214,hi-.392,.428,.265])
            nlinks=geometry(ax,shown_L[c],shown_f[c],shown_z,m,centers[c],limits,scale_bar=(c==0))
            lengths,angles=dimensions(l)
            fmt=lambda vals,form:", ".join(format(float(v),form) for v in vals)
            fig.text(xmid,hi-.403,"a, b, c   "+fmt(lengths,".2f")+" Å",fontsize=8.15,ha="center",va="top",color="#34454b")
            fig.text(xmid,hi-.432,"α, β, γ   "+fmt(angles,".1f")+"°",fontsize=7.9,ha="center",va="top",color="#4a585d")
            notes.append(dict(cell_lengths_A=lengths.tolist(),cell_angles_degree=angles.tolist(),displayed_contact_lines=nlinks))
        if rr==0: fig.add_artist(Line2D([.025,.975],[.505,.505],transform=fig.transFigure,lw=.65,color="#d6dfe0"))
        record.append(dict(id=label,draw_one_based=i+1,composition_source=composition,template_donor=donor,
            cell_donor=cdonor,requested_group=sg,species=z.tolist(),energy_eV_per_atom=float(row["above_augmented_hull"]),
            energy_pass=passed,converged=row["converged"],training_match=row["training_match"],reference_count=int(row["reference_count"]),
            camera_azimuth_deg=az,camera_elevation_deg=el,displays=notes))
    for ext in ("pdf","svg"):fig.savefig(HERE/f"{settings.stem}.{ext}",bbox_inches="tight",pad_inches=.07)
    (HERE/f"{settings.stem}.json").write_text(json.dumps(dict(arm=arm,examples=record,
        rule="The first proposal with at least two chemical elements in each energetic_pass class (True, then False), in the already locked random relaxation-panel hash order. Illustrations conditioned on the restricted-hull screen, not prevalence estimates.",
        coordinates="Two adjacent native cells along a are displayed. Native a,b,c and angles are printed beneath. All sites are wrapped after a common translation in both states. The pair uses the same orthographic camera and angstrom scale.",
        contacts="Illustrative segments connect displayed unlike atoms within 5 angstroms and 1.18 times the smaller of their periodic nearest-unlike-atom distances. They do not establish formal bonds."),
        indent=2)+"\n",encoding="utf8")
    print(record)

if __name__=="__main__":main()
