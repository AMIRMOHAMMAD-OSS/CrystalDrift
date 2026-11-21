"""Quantify cell geometry, coupled site motion and the independent de novo workloads."""
from pathlib import Path
import numpy as np
from compose import V3,EVAL,DEN,read,draw,table,text

def main():
    official=read(EVAL/"reports/test.json")
    methods=[("prior","Symbolic seed"),("angular-small-11","Angular 0.2M"),("contacts-small-11","Contacts 0.2M"),
             ("reciprocal-small-11","Reciprocal 0.2M"),("angular-medium-long","Angular 2M"),
             ("contacts-medium-long","Contacts 2M"),("meanflow-medium-long","Orbit MeanFlow 2M")]
    rr=[]
    for tag,name in methods:
        p=EVAL/"diagnostics"/("test-"+tag+".npz")
        with np.load(p) as d:
            x=d["draws"];ids=d["ids"].astype(str)
            assert list(ids)==official["ids"][:len(ids)] and len(ids)==96
            assert len(x)<=20*len(ids) and len(np.unique(x[["id","draw"]]))==len(x)
            rr.append([name,len(x),f"{draw(official,tag,20)['median_log_v']:+.3f}",f"{np.median(x['shape']):.3f}",
                       f"{np.median(x['site_median']):.3f}",f"{np.quantile(x['site_max'],.95):.3f}"])
    with np.load(EVAL/'diagnostics/test-contacts-medium-long.npz') as a,np.load(EVAL/'diagnostics/test-meanflow-medium-long.npz') as b:
        field_volume=[float(np.median(d['draws']['delta_volume'])) for d in (a,b)]
        field_motion=[float(np.median(d['draws']['site_median'])) for d in (a,b)]
    out=[r"\section{What changes in the predicted cell}",
         table("lrrrrr",["Generator","Draws",r"$\log(V/V_*)$","Shape error","Site median","Site tail"],rr,
             "The log-volume column takes the median across the full official 20-draw pools. The remaining geometric summaries use the first 96 registered test IDs with their ordered 20 draws. Shape error is dimensionless and site displacements are in angstroms. The last column is the 95th percentile of each draw's largest site displacement."),
         r"For the cell-shape diagnostic, Niggli-reduce the lattice, form its Gram matrix and divide by the cube root of its determinant. The error is the Euclidean distance between the sorted log eigenvalues of the proposed and reference unit-determinant matrices. This spectral statistic can coincide for different packings. For site motion we keep the paired seed's site ordering, wrap each fractional component of the update into the centered unit interval and convert to Cartesian displacement in the seed lattice. For a skew cell this chart convention can choose a different image from the shortest physical displacement. The site-median column takes the median within each draw, then the median across draws.",
         f"On these same 1,920 ordered proposals, the median log-volume change from the paired seed was {field_volume[0]:+.3f} for 2M contact drift and {field_volume[1]:+.3f} for same-backbone MeanFlow. Their median site-motion summaries were {field_motion[0]:.3f} and {field_motion[1]:.3f}~\\AA, respectively. These paired changes describe the network's action even for target proposals which never match.",
         r"\subsection{Observed costs for the empirical-composition pools}"]
    pools=[("prior","Symbolic seed"),("angular-small","Angular 0.2M"),("contacts-small","Contacts 0.2M"),
           ("angular-medium-long","Angular 2M"),("contacts-medium-long","Contacts 2M"),
           ("released-angular","Released angular"),("released-uflow","Released uFlowCSP")]
    rr=[]
    for tag,name in pools:
        q=read(DEN/"results"/(tag+".json"));r=read(DEN/"pools"/(tag+".json"))
        assert q["n"]==10000
        rr.append([name,q["with_fingerprints"],f"{q['w1_elements']:.3f}",r.get("batch","--"),
                   "GPU" if tag=="released-uflow" else "CPU",f"{r['seconds']:.1f}"])
    prior_seconds=read(DEN/'pools/prior.json')['seconds'];contact_seconds=read(DEN/'pools/contacts-medium-long.json')['seconds'];native_seconds=read(DEN/'pools/released-uflow.json')['seconds']
    out.append(f"Generating the 10,000 composition-matched candidates took {prior_seconds:.1f} seconds for the symbolic prior and {contact_seconds:.1f} seconds for the 2M contact map on one CPU thread. The separate released uFlowCSP GPU job took {native_seconds:.1f} seconds. The same-device cost comparison uses the synchronized workload in Table~\\ref{{tab:main}}.")
    out.append(table("lrrrrr",["Generator","Fingerprints",r"$W_1(n_{\rm elem})$","Batch","Device","Seconds"],rr,
        "Each raw pool used the same 10,000 sampled training compositions. Fingerprints counts the structures with usable structure and composition fingerprints for evaluation. The Wasserstein estimate samples within each method's eligible pool. Seconds give the recorded generation jobs on their respective devices and exclude retrospective matching and CHGNet relaxation. Table 1 gives the separate interleaved, shared-device benchmark."))
    text("physical-diagnostics.tex","\n\n".join(out))

if __name__=="__main__":
    main()
