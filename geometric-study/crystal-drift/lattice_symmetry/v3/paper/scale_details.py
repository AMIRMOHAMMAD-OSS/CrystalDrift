"""Document measured capacity, registered exposure and archived numerical sensitivities."""
from pathlib import Path
import json
import csv
import numpy as np
from write import HERE,V3,ROOT,table
from binary_history import binary_comparison, wbm_replication

def read(p):return json.loads(p.read_text(encoding="utf8"))
def main():
    fits=V3/"models/fits"
    desc={"tiny-angular-laplace-11-published":"Released graph, refit",
          "small-angular-laplace-11-published":"Widened released graph",
          "small-angular-laplace-11-local":"Orbit model, local messages",
          "small-angular-laplace-11-systemaux":"Training-only system label",
          "small-angular-laplace-11":"Orbit angular 0.2M","medium-angular-laplace-11":"Orbit angular 2M",
          "large-angular-laplace-11":"Orbit angular 15M","small-radial4-laplace-11":"Four radial shells",
          "small-angular-laplace-11-8pass":"Orbit angular 0.2M, long","medium-angular-laplace-11-8pass":"Orbit angular 2M, long",
          "medium-contacts-laplace-11-8pass":"Orbit contact 2M, long",
          "medium-meanflow-laplace-11-commonorder-8pass":"Orbit MeanFlow 2M, long",
          "reference-contacts-laplace-11":"Reference-size contacts","reference-angular-laplace-11":"Reference-size angular"}
    data=[]
    for label,name in desc.items():
        p=fits/label
        if not (p/"complete.json").is_file():continue
        r=read(p/"complete.json");c=read(p/"confirmation.json") if (p/"confirmation.json").is_file() else {}
        data.append([name,f"{r['parameters']:,}",f"{r['updates']:,}",f"{r.get('anchor_exposures',r['distinct_anchors']):,}",
                     f"{r['seconds']/60:.1f}",f"{r['peak_allocated_MiB']:.0f}",c.get("hits","--")])
    for tag,name,pars in (("small-11-long","Native uFlow 0.2M, long",233868),
                          ("medium-11-long","Native uFlow 2M, long",2135052)):
        p=V3/"uflow/fits"/tag;r=read(p/"complete.json")
        with np.load(V3/f"uflow/scores/{tag}/frozen/confirm/outcomes.npz") as d:hit=int(np.isfinite(d["rms"][:,:5]).any(1).sum())
        data.append([name,f"{pars:,}",f"{r['updates']:,}",f"{r['rows']:,}",f"{r['seconds']/60:.1f}",f"{r['peak_bytes']/2**20:.0f}",hit])
    txt=[r"\section{Scale, exposure and numerical boundaries}",
      table("lrrrrrr",["Generator","Parameters","Updates","Anchors","Minutes","Peak MiB","Confirm."],data,
        "Actual completed fits on one RTX 3060 Laptop GPU. Minutes are observed fit-receipt wall times. First-traversal rows and eight-traversal rows refer to their own independently labeled budgets. Concurrent host scoring and bank preparation can affect these training times. Confirmation is the selected checkpoint's five-draw count out of 96. Peak memory is allocated device tensor memory."),
      r"A direct small-model replay exposed a boundary of the matcher. For material \texttt{mp-8584}, draw 2, the public matcher accepted the archived CUDA coordinates with normalized RMS displacement 0.445342 and returned no match for a regenerated CPU realization. The maximum perturbations over the 480 development proposals were $3.82\times10^{-6}$~\AA{} in the cell and $1.20\times10^{-7}$ in fractional coordinates. One flag changed and all native-volume-qualified target counts agreed. A CUDA replay using a cached-input shortcut also changed this decision. The production validation path retained its original seed construction and its archived arrays. The release identifies the device on which each output was produced.",
      r"The first orbit MeanFlow experiment used the same anchor set as drift but also drew event times from the within-shard shuffle generator. Only 328 of 26,419 ordered positions agreed with angular drift. The strict control draws times from a separate seed stream. Its complete ordered anchor sequence agrees with drift at every position. This distinction affects comparisons of training trajectories even when total exposure and distinct IDs agree.",
      r"\paragraph{Earlier datasets and controls.} The preceding fixed-basis study includes a 128-target binary MP-20 cohort, a separate full-test replay, Wang--Botti--Marques structures from the Matbench Discovery release \cite{wbm,matbench}, and an Alexandria allocation from the Alex-MP-20 release \cite{mattergen} with reduced-formula exclusions. For WBM, a new reduced-formula split of 256,259 relaxed cells gives 205,091 training, 25,117 validation and 26,051 test entries. Seeds 11, 23 and 37 were evaluated on the archived 128-target WBM test cohort. Those archived experiments use the models and splits of that release. The periodic models developed here have new protected donors and primitive-orbit eligibility. Their official MP-20 denominator is 9,046. The source bundle identifies the checksums and independent reconstruction commands for both earlier releases."]
    rep=read(V3/"evaluation/reports/test.json")
    capacity_rows=[]
    for tag,name in (("sitegraph-tiny","Released graph, refit"),("sitegraph-small","Widened released graph"),
                     ("angular-small-local","Local orbit backbone"),("angular-small-11","Global orbit, 0.2M"),
                     ("angular-small-systemaux","Global orbit, system supervision"),("radial4-small","Global orbit, four radial shells"),
                     ("angular-medium-11","Global orbit, 2M"),("angular-large-11","Global orbit, 15M"),
                     ("reference-scale-11","Reference-scale orbit model")):
        key='all/5/'+tag
        if key not in rep['quality']:continue
        q=rep['quality'][key];d=rep['paired'][key+'--prior'];q20=rep['quality']['all/20/'+tag]
        capacity_rows.append([name,q['hits_per_seed'][0],f"{d['pp']:+.2f}",f"[{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]",q['native_hits'][0],q20['hits_per_seed'][0]])
    if capacity_rows:txt.insert(2,table('lrrrrr',['Generator','Hits@5',r'$\Delta$ pp','95\\% interval','Native@5','Hits@20'],capacity_rows,
        'Frozen architectural controls on all 9,046 MP-20 targets. Each difference subtracts the same supplied symbolic starting structures. System supervision is a seven-way training-only loss. Counts are paired by target and proposal prefix.'))
    if (fits/'small-angular-laplace-11-systemaux/complete.json').is_file():
        txt.append(r"\paragraph{Training-only crystal-system label.} In the small angular control an auxiliary linear head predicts one of seven crystal systems from the multiplicity-weighted mean orbit embedding. We add $0.1$ times the cross-entropy for the two active generated draws with the labels of their registered same-formula positive partners. The $7\times 114$ weights and seven biases add 805 training-only parameters. Generation discards the classifier and receives the same formula and symbolic seed as ordinary angular drift.")
    with (V3/"models/fixed-training-cost.csv").open() as fp:
        budget={(r['tier'],r['method'],int(r['training_job_s'])):r for r in csv.DictReader(fp)}
    by_cost=[]
    for tier in ('small','medium'):
        for method,label in [('angular','Angular drift'),('contacts','Contact drift'),('meanflow','Orbit MeanFlow'),('uflow','Native uFlowCSP')]:
            vals=[budget.get((tier,method,cap)) for cap in (120,240,960)]
            by_cost.append([label+(' 0.2M' if tier=='small' else ' 2M'),*[v['validation_hits'] if v else '--' for v in vals]])
    txt.insert(2,table('lrrr',['Model','120 s','240 s','960 s'],by_cost,
        'Best five-draw development recovery out of 96 targets among saved checkpoints completed by each elapsed training budget. A dash means no registered checkpoint was available by that deadline. Time is measured on the same RTX 3060 Laptop GPU and includes the fit up to that checkpoint. It excludes the separate reference-bank construction and the cost of retrospective matching. Repeated counts can denote the same selected weights.'))
    txt.append(wbm_replication())
    txt.append(binary_comparison())
    txt.append(r"The compact binary fits used 3,630 training cells. One Euler step diagnoses the discretization of the compact flow at the one-call budget. Table~\ref{tab:historical-binary} uses the ordered material identifiers in \texttt{revision/binary/target-crosswalk.csv} and reconstructs each endpoint from its 128-by-20 matrix of public matcher displacements. This cohort and these fixed-basis weights are separate from the full-chemistry orbit fits.")
    (HERE/"scale-details.tex").write_text("\n\n".join(txt)+"\n")
if __name__=="__main__":main()
