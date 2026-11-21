"""Build matched seed and initializer comparisons, retaining each individually trained weight."""
from pathlib import Path
import argparse
import csv
from compose import EVAL, V3, read, table, pct, text
from write import HERE

ARMS=(
    ("Angular 0.2M","angular-small","prior"),
    ("Contacts 0.2M","contacts-small","prior"),
    ("Angular 2M","angular-medium","prior"),
    ("Contacts 2M","contacts-medium","prior"),
    ("Native uFlow 0.2M","uflow-small","uflow-prior-11"),
    ("Native uFlow 2M","uflow-medium","uflow-prior-11"),
)

def main(working):
    lines=[]; details={}
    for panel in ("test","family","shared"):
        rep=read(EVAL/"reports"/(panel+".json"))
        for label,base,prior in ARMS:
            for k in (1,5,20):
                stem=f"all/{k}/";agg=base+"-3seeds"
                if stem+agg not in rep["quality"]:
                    if working:continue
                    raise ValueError(f"Missing three-seed reduction {panel}/{k}/{base}")
                q=rep["quality"][stem+agg]; zero=rep["quality"][stem+prior]
                paired=rep["paired"][stem+agg+"--"+prior]
                for i,s in enumerate((11,23,37)):
                    individual=rep["quality"][stem+base+"-"+str(s)]
                    assert q["hits_per_seed"][i]==individual["hits_per_seed"][0]
                    lines.append(dict(panel=panel,method=label,seed=s,k=k,targets=q["n"],hits=q["hits_per_seed"][i],
                        zero_call_hits=zero["hits_per_seed"][0],native=individual["native_hits"][0],zero_call_native=zero["native_hits"][0],
                        scale_pp=paired["seed_pp"][i],rms=individual["rms_matched"],strict_mr=individual["strict"],
                        target_bootstrap_across_fixed_seeds_pp=paired["pp"],bootstrap_low_pp=paired["ci95_pp"][0],
                        bootstrap_high_pp=paired["ci95_pp"][1],score_file=rep["source"][base+"-"+str(s)][0]))
                details[panel,k,base]=(q,zero,paired)
    if lines:
        with (V3/"release/SEED-RESULTS.csv").open("w",newline="",encoding="utf8") as fp:
            writer=csv.DictWriter(fp,fieldnames=list(lines[0]));writer.writeheader();writer.writerows(lines)
    out=[r"\section{Training-seed variability and the native priors}",
        r"Each model in the replication experiment traverses its complete eligible training set. Development MR@5 selects between the half- and full-traversal checkpoints independently in each fit. For each target the three fitted weights use the same ordered initial draws. We first average paired win-minus-loss indicators over the three weights for that target, then resample targets or protected families. The interval describes target sampling for these three fitted models. The deposited per-seed results retain variation among the fits."]
    for panel,title,k in (("test","Official MP-20",5),("family","Withheld families",20)):
        data=[]
        for label,base,prior in ARMS:
            key=(panel,k,base)
            if key not in details:continue
            q,zero,d=details[key];lo,hi=d["ci95_pp"]
            data.append([label,zero["hits_per_seed"][0],*q["hits_per_seed"],f"{d['pp']:+.2f}",f"[{lo:+.2f}, {hi:+.2f}]",f"{d['native_pp']:+.2f}"])
        if data:
            out.append(table("lrrrrrrr",["Method","Prior","11","23","37",r"$\Delta$ pp","95\\% interval",r"$\Delta$ native"],data,
                f"{title} at {k} draws. Counts for training seeds 11, 23 and 37 use the same ordered formulas and paired initial noise. The drift rows use an inferred symbolic orbit template. Retrained native uFlowCSP rows use the Gaussian law of the released architecture. The last column applies the same-draw physical-volume requirement."))
    official=read(EVAL/"reports/test.json"); paired_file=V3/"uflow/published-paired.json"
    from compose import ci_text
    nd=[]
    for tier in ('small','medium'):
        key=f'all/5/contacts-{tier}-3seeds--angular-{tier}-3seeds'
        if key in official['paired']:nd.append(ci_text(official['paired'][key]))
    if len(nd)==2:
        out.append(f"Matching the training seed within each contact--angular pair gives a five-draw difference of {nd[0]} at 0.2M parameters and {nd[1]} at 2M. Both contrasts share the starting proposals and target IDs.")
    if 'all/5/contacts-medium-3seeds' in official['quality']:
        q=official['quality']['all/5/contacts-medium-3seeds'];z=official['quality']['all/5/prior']
        scale=', '.join(f'{i:,}' for i in q['hits_per_seed']);native=', '.join(f'{i:,}' for i in q['native_hits'])
        out.append(f"For 2M contact drift, scale-enabled five-draw hit counts across fits were {scale}. The corresponding native-volume counts were {native}, versus {z['native_hits'][0]:,} for the common prior. Agreement of scale-enabled rates across independent training runs obscures much larger variation in native-volume recovery.")
    if not working and not paired_file.is_file():raise ValueError('Missing released-checkpoint native prior replay')
    published=read(paired_file)['metrics'] if paired_file.is_file() else {}
    native_rows=[]
    for k in (1,5,20):
        for tag,label in [('uflow-small-11-long','Retrained small'),('uflow-medium-11-long','Retrained medium')]:
            stem=f'all/{k}/';q=official['quality'][stem+tag];zero=official['quality'][stem+'uflow-prior-11']
            d=official['paired'][stem+tag+'--uflow-prior-11'];lo,hi=d['ci95_pp']
            native_rows.append([label,k,zero['hits_per_seed'][0],q['hits_per_seed'][0],f"{d['pp']:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',q['native_hits'][0]])
        if str(k) in published:
            q=published[str(k)];m=q['scale'];lo,hi=m['ci95_pp']
            native_rows.append(['Released full size',k,m['zero_hits'],m['released_hits'],f"{m['delta_pp']:+.2f}",f'[{lo:+.2f}, {hi:+.2f}]',q['native']['released_hits']])
    out.append(table('lrrrrrr',['Model','$k$','Prior','Trained',r'$\Delta$ pp','95\\% interval','Native hits'],native_rows,
        'Gaussian starts and their one-call outputs on the same 9,046 official targets. The retrained rows use their selected eight-traversal weights. Each prior returns the actual starting tensor for the corresponding learned proposal. The published weights use their original replay noise streams. The paired intervals resample targets.'))
    text("replication-results.tex","\n\n".join(out))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--working",action="store_true");main(p.parse_args().working)
