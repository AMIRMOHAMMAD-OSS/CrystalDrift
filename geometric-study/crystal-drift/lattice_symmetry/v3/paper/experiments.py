"""Write paired training, transfer and physical-screen sections from archived records."""
import csv
import json
from pathlib import Path
from compose import HERE,V3,DEN,EVAL,read,rows,draw,pair,pct,r3,ci_text,figure,table,text
from write import devrows

def training():
    shown=devrows()
    looks=[]
    for title,folder in (("Angular 0.2M","small-angular-laplace-11-8pass"),
                         ("Angular 2M","medium-angular-laplace-11-8pass"),
                         ("Contact 2M","medium-contacts-laplace-11-8pass"),
                         ("Orbit MeanFlow 2M","medium-meanflow-laplace-11-commonorder-8pass")):
        p=V3/"models/fits"/folder
        if (p/"selection.json").is_file():
            s=read(p/"selection.json");c=read(p/"confirmation.json")
            looks.append([title,*[s["dev"][key]["hits"] for key in ("half","full","epoch4","epoch8")],
                          Path(s["chosen"]).stem,c["hits"]])
    for tier in ("small","medium"):
        p=V3/"uflow/fits"/(tier+"-11-long")
        if (p/"selection.json").is_file():
            import numpy as np
            s=read(p/"selection.json")
            with np.load(V3/f"uflow/scores/{tier}-11-long/frozen/confirm/outcomes.npz") as z:
                conf=int(np.isfinite(z["rms"][:,:5]).any(1).sum())
            looks.append(["Native uFlow "+("0.2M" if tier=="small" else "2M"),
                *[s["results"][key] for key in ("ema-00416","ema-00832","ema-03328","ema-06656")],
                "eight",conf])
    # The controlled screen uses ordinary positive banks and seed 11. Longer runs have a
    # separate table and never compete with the first-traversal descriptor comparison.
    main=[r"\paragraph{Equal training exposure.} Each protected primitive training ID contributes an anchor once per traversal. Positive and negative banks each contain two draws. The scheduled anchor supplies the first positive, and the second comes from the same composition. These banks are often sparse. Among the scheduled rows, 86.74\% have only one distinct observed structure at that exact composition. Training seed construction excludes all structures with the anchor's reduced chemistry from the donor search.",
          r"The full first traversal of the small orbit model processes 26,419 distinct anchor structures and 274,066 anchor atoms in 1,652 optimizer updates. The global orbit networks contain 218,497 and 2,098,297 registered parameters. An embedding can count as trainable even when its element or orbit type is absent from the data. The evidence identifies each inactive input row and verifies its zero gradient. Replacing the field with a MeanFlow loss keeps these widths, the seed distribution and the decoder fixed. We report the original within-shard order and a second run with matching anchor order and a separate random stream for event times.",
          r"The native uFlowCSP small and medium networks have 233,868 and 2,135,052 registered parameters. Their active counts after removing the disabled fine space-group branch are 219,084 and 2,098,092. The authors' full architecture has 131,215,116 parameters. The initial native training catalog has 26,614 rows. A second native fit uses exactly the 26,419 orbit-eligible rows and their order. The common anchors do not make the native objective, training-only symmetry labels or Gaussian input distribution identical to the orbit experiment.",
          figure("learning-curves","Development recovery at the saved training exposures for the orbit generator. Each point counts targets recovered at five draws. The same disjoint confirmation panel scores the selected checkpoint of each run.","fig:learn"),
          figure("uflow-learning","Native uFlowCSP development recovery as capacity and exposure increase. These fits start from the authors' 100-token Gaussian prior. Each marker represents an actual EMA checkpoint.","fig:ufit")]
    if len(looks)==6:
        main.insert(3,table("lrrrrlr",["Generator","0.5","1","4","8","Chosen","Confirm."],looks,
            "Development hits among 96 targets at four fixed training exposures, in traversals. The chosen checkpoint is the earliest development maximum. The independent confirmation panel also has 96 targets."))
        main.insert(4,r"All four orbit models selected the half-traversal checkpoint. Both reduced native uFlowCSP models selected eight traversals. More training improved native uFlowCSP development recovery; the longer orbit trajectories did not improve the earlier maximum. The matched parameter table reports this validation decision explicitly. A common allocated training budget and an identical deployed exposure are different comparisons.")
    if (HERE/"capacity-cost.pdf").is_file():
        main.append(figure("capacity-cost","Recovery against instantiated parameter count on the official split and against measured batch-one GPU latency on the timed subset. Markers describe the trained models and their native priors. The released 131M comparator has its original training history.","fig:capacity"))
    scale=read(EVAL/"reports/test.json")['quality']
    if 'all/5/angular-large-11' in scale:
        large=scale['all/5/angular-large-11']
        main.append(f"The 15,011,747-parameter angular network used the same 26,419 anchors and 1,652 updates in its full traversal as the smaller angular networks. Both checkpoint looks recovered 62 of 96 development targets. The selected half-traversal weights recovered 50 of 96 on confirmation and {large['hits_per_seed'][0]:,} of 9,046 at five draws on the public test set.")
    if 'all/5/reference-scale-11' in scale:
        ref=scale['all/5/reference-scale-11'];r=read(V3/'models/fits/reference-contacts-laplace-11/complete.json')
        medium=scale['all/5/contacts-medium-long']
        elapsed=read(V3/'models/reference-wall-clock.json')['elapsed_s']
        main.append(f"The {r['parameters']:,}-parameter contact model trained on the same anchors at microbatch two. A traversal required {r['updates']:,} optimizer updates. The fit clock recorded {r['seconds']/60:.1f} minutes on the laptop GPU. Elapsed time from the capacity check to the final optimizer checkpoint was {elapsed/60:.1f} minutes, including a restart, validation generation and file writes. At five draws the selected 136M and 2M checkpoints each recovered {ref['hits_per_seed'][0]:,} targets. With native-volume agreement, the counts were {ref['native_hits'][0]:,} and {medium['native_hits'][0]:,}, respectively. The difference in optimizer-step counts limits a capacity-only interpretation.")
    text("training-results.tex","\n\n".join(main))
    detailed=[r"\section{Capacity, exposure and validation details}",
        table("lrrrrr",["Generator","Registered","Look","Dev.","Confirm.","Native"],shown,
          "Development and confirmation recovery at five draws out of 96 targets in each panel. Native is the confirmation count satisfying the same-draw volume rule. Checkpoints and per-target flags accompany every row. The longer capacity series is shown separately."),
        r"Fresh geometry and negative draws are regenerated deterministically for each later traversal. All methods in a comparison index the same saved bank stream. The Gaussian and distance-based affinities change both weights and the detached displacement direction. A kernel comparison at common exposure uses the same optimizer schedule, reference pairs, query seeds, lattice features and evaluation seeds. The registered two-look screen and the fixed full-traversal reanalysis account separately for early checkpoint selection."]
    text("validation-detail.tex","\n\n".join(detailed))

def transfer():
    f=read(EVAL/"reports/family.json");s=read(EVAL/"reports/shared.json")
    a=read(EVAL/"reports/arrangement.json");c=read(EVAL/"reports/arrangement-control.json")
    names=[("Symbolic prior","prior"),("Small angular","angular-small-11"),("Small contact","contacts-small-11"),
           ("Small reciprocal","reciprocal-small-11"),("Medium contact","contacts-medium-11"),("Orbit MeanFlow","meanflow-medium-11")]
    rr=[]
    for name,m in names:
        rr.append([name,*[draw(r,m)["hits_per_seed"][0] for r in (f,s,a,c)]])
    t=table("lrrrr",["Generator","Family (228)","Shared (96)","Uncovered (25)","Covered (96)"],rr,
        "Five-draw recovery on four explicitly identified cohorts. The first two are protected transfer panels. The last two are exploratory validation strata defined by an independent anonymous-structure search. All listed models use the first two-look selection and seed 11.","tab:transfer")
    prose=[r"\subsection{Withheld families and continuous arrangements}",t,
        f"With all members of a structural family withheld, the small contact update changed five-draw recovery by {ci_text(pair(f,'contacts-small-11'))}. For shared symbolic families with held-out chemistry the corresponding gain was {ci_text(pair(s,'contacts-small-11'))}. Family-level resampling preserves correlations between materials from the same group of occupations. The anonymous occupied-pattern vocabulary cannot exactly retrieve any of the 228 withheld-family targets. Their scale-enabled matches therefore need a separate crystallographic check.",
        r"Among 512 additional validation targets searched for anonymous analogues within an occupied pattern, 25 lacked a match at the registered tolerance. After this search we froze those IDs with 96 covered controls. At five draws all nine frozen orbit rows recovered the same one of the 25 uncovered targets. The absence of discordances leaves a one-sided 95\% upper bound of 11.3\% on the probability of a discordant target under independent sampling. In the covered controls, the small contact map recovered 41 of 96 and the prior 34. The contrast within each group is paired. The search that defined these two strata prevents interpreting their raw recovery difference as a causal effect of template availability."]
    if 'all/20/contacts-small-3seeds' in f['quality']:
        k20=draw(f,'contacts-small-3seeds',20);zero=draw(f,'prior',20);change=pair(f,'contacts-small-3seeds',k=20)
        prose.append(f"At 20 draws the three small contact fits recovered {', '.join(str(i) for i in k20['hits_per_seed'])} of 228 excluded-family targets, compared with {zero['hits_per_seed'][0]} for the identical prior. The seed-averaged difference was {ci_text(change)}. Extra draws improve the chance of meeting the matcher even when the exact occupied pattern lies outside the available catalog.")
    if (V3/"geometry/audits/supplied-confirm/result.json").is_file():
        prose.append(r"For the seed-11 small contact model, none of the 49 scale-enabled hits on withheld families had the reference space-group type under independent detection at either 0.01 or 0.10~\AA{} in the best accepted proposal. At 20 draws, two targets had at least one accepted proposal with the correct detected group under the primary tolerance, both before and after refinement. None of the 21 newly recovered targets had such a proposal. Returning the correct crystal family and satisfying the published scale-enabled matcher are distinct endpoints on this challenge.")
        prose.append(r"Supplying the target's element-labelled occupied pattern tests a different input regime. On the 96 frozen confirmation targets, the independently rerun medium contact pipeline recovered 51 at five draws when inferring the pattern and 70 when supplied with it. With the supplied pattern alone, its zero-call prior recovered 66. The 19-target difference from supplying the pattern was larger than the four-target gain from refining its continuous geometry. The supplement records the chart failure and the numerical sensitivity of the inferred rerun.")
    text("transfer-results.tex","\n\n".join(prose))

def physical():
    source=[r for r in rows(DEN/"screen/by-arm.csv") if r["group"]=="all"]
    rr=[]
    for r in source:
        name=r["arm"].replace("-",r"\,")
        rr.append([name,r["n"],r["complete_hulls"],r["converged"],r["thermally_screened"],r["sun_lower"],f"{float(r['candidate_core_seconds']):.1f}"])
    t=table("lrrrrrr",["Raw pipeline","Draws","Hull","Conv.","Energy","Joint","CPU s"],rr,
        "Paired CHGNet relaxation experiment on eight prospectively hash-selected draws per arm. Hull counts complete restricted-hull calculations, Conv. counts optimizer convergence, Energy counts passing the restricted energy screen and Joint counts all physical, novelty and uniqueness requirements. CPU seconds sum single-core candidate relaxation times. Competing phases are shared across arms.","tab:hull")
    prose=[r"\subsection{Relaxation against a finite phase library}",t,
         r"We used 119 distinct retained competing phases across the eight selected compositions. Each candidate and phase received the same potential, cell relaxation, optimizer and force criterion. The joint endpoint additionally requires the covalent-contact screen, novelty against the original training catalog and within-panel uniqueness. A zero count out of eight permits an underlying yield as high as 31.2\% at a one-sided 95\% binomial limit. We treat these as restricted, model-potential outcomes. Experimental synthesizability and DFT stability require independent evidence.",r"\input{crystal-gallery-extended}"]
    text("physical-results.tex","\n\n".join(prose))

if __name__=="__main__":
    training();transfer();physical()
