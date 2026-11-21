"""Compare measured recovery across instantiated capacities and same-device proposal latencies."""
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figures import HERE,V3,TEAL,BLUE,ORANGE,VIOLET,GRAY,save
ROOT=V3.parents[1]
def read(p):return json.loads(p.read_text())
def main():
    rep=read(V3/"evaluation/reports/test.json")["quality"]
    models=[
        ("sitegraph-tiny","Site graph",6723,GRAY,"s"),
        ("angular-small-long","Angular drift",218497,TEAL,"o"),("angular-medium-long","Angular drift",2098297,TEAL,"o"),
        ("contacts-medium-long","Contact drift",2098297,ORANGE,"+"),("meanflow-medium-long","Orbit MeanFlow",2098297,VIOLET,"x"),
        ("uflow-small-11-long","Retrained uFlowCSP",233868,BLUE,"s"),("uflow-medium-11-long","Retrained uFlowCSP",2135052,BLUE,"s"),
        ("released-uflow","Released uFlowCSP",131215116,BLUE,"*")]
    for m in ("angular-large-11","reference-scale-11"):
        key="all/5/"+m
        if key in rep:
            group="large-angular-laplace-11" if m.startswith("angular-") else next(p.parent.name for p in (V3/"models/fits").glob("reference-*-11/complete.json"))
            par=read(V3/"models/fits"/group/"complete.json")["parameters"]
            name,color,style=("Angular drift",TEAL,"o") if "angular" in group else ("Contact drift",ORANGE,"+")
            models.append((m,name,par,color,style))
    clock=V3/"evaluation/timing"
    prices={v["method"]:float(v["first_median"]) for v in csv.DictReader((clock/"latency.csv").open())}
    clock_mr={v["method"]:100*float(v["mr"]) for v in csv.DictReader((clock/"draws.csv").open()) if int(v["k"])==5}
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.15))
    legend={};records=[]
    for m,label,par,color,mark in models:
        key="all/5/"+m
        if key not in rep:continue
        mr=100*rep[key]["mr"]
        size=65 if mark=="*" else 36
        style={"edgecolors":color,"facecolors":"none"} if mark in ("s","o") else {"color":color}
        axes[0].scatter(par,mr,s=size,marker=mark,linewidths=1.3,zorder=3,**style)
        if m in prices:axes[1].scatter(prices[m],clock_mr[m],s=size,color=color,marker=mark,linewidths=1.3,zorder=3)
        legend[label]=Line2D([],[],ls="",marker=mark,color=color,markerfacecolor="none" if mark in ("s","o") else color)
        records.append(dict(method=m,parameters=par,official_mr5=mr,clock_mr5=clock_mr.get(m),gpu_first_median_ms=prices.get(m)))
    axes[0].set(xscale="log",xlabel="Registered parameters",ylabel="Five-draw recovery (%)",title="Official MP-20 · 9,046 targets")
    axes[1].set(xlabel="Median first proposal (ms, GPU)",title="Timed panel · 64 targets",xticks=[8,10,12,14,16,18])
    for ax in axes:ax.grid(alpha=.15);ax.set_ylim(0,100)
    fig.legend(list(legend.values()),list(legend),ncol=3,frameon=False,loc="lower center",fontsize=7)
    fig.tight_layout(rect=(0,.18,1,1));save(fig,"capacity-cost")
    with (HERE/"capacity-cost.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=records[0]);w.writeheader();w.writerows(records)
if __name__=="__main__":main()
