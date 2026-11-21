"""Reduce the fixed one-traversal descriptor and affinity screen using identical development seeds."""
import csv
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from figures import HERE,V3,TEAL,ORANGE,BLUE,GRAY,save
from write import table
sys.path.insert(0,str(V3/"evaluation"))
from analysis import ci

def signals(path):
    with np.load(path) as a:
        g=np.isfinite(a["rms"])
        n=g&(np.abs(a["log_volume_ratio"])<=np.log(1.1))
        return a["ids"].astype(str),g[:,:5].any(1),n[:,:5].any(1)

def main():
    F=V3/"models/fits"
    ids,prior,pn=signals(F/"prior/dev-scores.npz")
    aid,b,bn=signals(F/"small-angular-laplace-11/full-dev-scores.npz")
    np.testing.assert_array_equal(aid,ids)
    arms=[("Pair","pair","laplace"),("Center radial","center0","laplace"),("Angular","angular","laplace"),
          ("Orbit variance","variance","laplace"),("Inter-orbit contact","contacts","laplace"),
          ("Reciprocal","reciprocal","laplace"),("Angular","angular","gaussian"),("Inter-orbit contact","contacts","gaussian")]
    qq=[]
    for name,arm,affinity in arms:
        p=F/f"small-{arm}-{affinity}-11/full-dev-scores.npz"
        tid,h,nt=signals(p)
        np.testing.assert_array_equal(ids,tid)
        row={"name":name,"descriptor":arm,"affinity":affinity,"n":len(ids),"hits":int(h.sum()),
             "native":int(nt.sum()),"prior_hits":int(prior.sum()),"prior_native":int(pn.sum())}
        for k,x in (("prior",prior),("angular",b)):
            d=h.astype(float)-x;low,high=ci(d)
            row.update({k+"_pp":100*d.mean(),k+"_low":100*low,k+"_high":100*high,k+"_gained":int((h&~x).sum()),k+"_lost":int((x&~h).sum())})
        qq.append(row)
    with (HERE/"kernel-screen.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=qq[0]);w.writeheader();w.writerows(qq)
    fig,ax=plt.subplots(figsize=(6.15,3.3))
    for i,q in enumerate(qq):
        y=q["prior_pp"];lo=q["prior_low"];hi=q["prior_high"]
        ax.errorbar(y,i,xerr=[[y-lo],[hi-y]],fmt="o" if q["affinity"]=="laplace" else "s",
            color=TEAL if q["affinity"]=="laplace" else ORANGE,capsize=2.5,ms=4)
    ax.axvline(0,color=GRAY,lw=.8);ax.set_yticks(range(len(qq)),[q["name"]+" · "+q["affinity"].capitalize() for q in qq])
    ax.invert_yaxis();ax.set_xlabel("Five-draw gain over the identical prior (percentage points)");ax.grid(axis="x",alpha=.13)
    fig.tight_layout();save(fig,"kernel-screen")
    data=[[q["name"],q["affinity"],q["hits"],q["native"],f"{q['prior_pp']:+.2f}",f"{q['angular_pp']:+.2f}",
           f"{q['angular_gained']}/{q['angular_lost']}"] for q in qq]
    tex=table("llrrrrr",["Descriptor","Affinity","Hits","Native","vs.\\ prior","vs.\\ angular","Gains/losses"],data,
      "Fixed full-traversal screen on 96 development targets at five draws. The last three columns give paired percentage-point changes and the numbers gained/lost against ordinary angular drift with Laplace affinity. Each model has 218,497 parameters. The same fixed physical seed bank and generator initialization enter every otherwise matched comparison.")
    (HERE/"kernel-results.tex").write_text(tex+"\n"+r"\begin{figure}[htbp]\centering\includegraphics[width=.9\linewidth]{kernel-screen.pdf}\caption{Paired development increments at the end of one full traversal. Each interval resamples the 96 target identities with both arms fixed.}\end{figure}"+"\n")
if __name__=="__main__":main()
