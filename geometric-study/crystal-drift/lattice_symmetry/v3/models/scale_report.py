"""Write the capacity, supervised-control and measured cost ledger from raw receipts."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    rows=[]
    labels=['tiny-angular-laplace-11-published','small-angular-laplace-11-published',
       'small-angular-laplace-11-local','small-angular-laplace-11','medium-angular-laplace-11','medium-contacts-laplace-11',
       'large-angular-laplace-11','small-angular-laplace-11-systemaux']
    gate=HERE/'reference-capacity.json'
    if gate.exists():
        d=json.loads(gate.read_text());labels.append(f"reference-{d['arm']}-laplace-11")
    for label in labels:
        root=HERE/'fits'/label;complete=root/'complete.json';selection=root/'selection.json';confirm=root/'confirmation.json'
        if not all(f.is_file() for f in [complete,selection,confirm]):continue
        c,s,f=(json.loads(file.read_text()) for file in [complete,selection,confirm])
        pick=Path(s['chosen']).stem;supervision=c.get('system_auxiliary',{})
        end_to_end=None
        if label.startswith('reference-'):
            receipt=HERE/'reference-wall-clock.json'
            if not receipt.exists():
                start=gate.stat().st_mtime;last=max(complete.stat().st_mtime,(root/'resume-location.json').stat().st_mtime)
                evidence=dict(gate_complete_unix=start,final_receipt_unix=last,elapsed_s=last-start,
                              boundary='From the completed physical capacity gate through final fit/optimizer receipt, including restart and I/O.')
                receipt.write_text(json.dumps(evidence,indent=2))
            end_to_end=json.loads(receipt.read_text())['elapsed_s']
        rows.append(dict(model=label,supervision='train-only crystal-system auxiliary' if supervision else 'drift',
            parameters=c['parameters'],training_only_parameters=supervision.get('training_only_parameters',0),
            steps=c['updates'],eligible_anchors=c['distinct_anchors'],anchor_exposures=c.get('anchor_exposures',c['query_exposures']//2),batch=c['batch'],
            training_job_s=c['seconds'],full_schedule_s=end_to_end,allocated_peak_MiB=c['peak_allocated_MiB'],selected=pick,
            development_hits=s['dev'][pick]['hits'],development_native=s['dev'][pick]['native'],
            confirmation_hits=f['hits'],confirmation_native=f['native'],digest=s['sha256']))
    with (HERE/'capacity-ledger.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Measured capacity and exposure','',
          'Each row uses 96 development targets to select the checkpoint and a disjoint 96-target confirmation panel. The identical prior recovered 59 development and 48 confirmation targets at five draws. Native volume requires a match within a factor of 1.1 of the target volume on that same proposal. Training time includes checkpoint writes and validation geometry generation and excludes CPU structure matching.','',
          '| Fit | Deployed weights | Extra training weights | Updates | First-pass seconds | Peak CUDA MiB | Selected | Development | Confirmation | Native confirmation |',
          '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        text.append('| {model} | {parameters:,} | {training_only_parameters:,} | {steps:,} | {training_job_s:.1f} | {allocated_peak_MiB:.1f} | {selected} | {development_hits} | {confirmation_hits} | {confirmation_native} |'.format(**r))
    if gate.exists():
        d=json.loads(gate.read_text())
        text+=['','## Comparator-sized profile','',f"The {d['parameter_count']:,}-parameter {d['arm']} model completed {d['profile_updates']} measured optimizer updates on the local RTX 3060 Laptop GPU. After one warm-up update, the mean was {d['warm_mean_update_s']:.3f} s and p95 was {d['warm_p95_update_s']:.3f} s. The measured peak was {d['measured_peak_allocated_MiB']:.1f} MiB. At microbatch {d['microbatch']} a traversal of 26,419 anchors takes {d['projected_updates']:,} updates, compared with 1,652 at microbatch 16. Extrapolating the warm mean gives {d['projected_single_traversal_s']/3600:.2f} GPU hours before checkpoint writes and selection.", '', 'The 15M row extends the fixed-angular, microbatch-16 series. The comparator-sized contact row uses the promoted descriptor and about eight times as many optimizer updates per traversal. The medium contact row supplies its descriptor-matched reference at matched data exposure. The auxiliary row changes supervision at the original small angular capacity and inference interface.']
        clock=HERE/'reference-wall-clock.json'
        if clock.exists():text+=['',f"The comparator-sized full fit occupied {json.loads(clock.read_text())['elapsed_s']/3600:.2f} wall hours from its positive capacity gate to its final fit and optimizer receipt, including a checkpoint restart, I/O and validation structure generation."]
    (HERE/'CAPACITY.md').write_text('\n'.join(text)+'\n')
    print('Produced capacity ledger for',len(rows),'frozen fits.')
if __name__=='__main__':main()
