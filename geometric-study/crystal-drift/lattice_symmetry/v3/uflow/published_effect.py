"""Paired published EMA and exactly replayed pre-update latent, including original cell checks."""
import json
import numpy as np
from results import HERE,ROOT,ci,read,reorder,by_draw

def main():
    old=ROOT/'revision/full/external/mp20'
    score=HERE/'scores/published-prior/frozen/test/outcomes.npz'
    prior=read(score)
    with np.load(score) as d:zero_errors=int(np.count_nonzero(d['error']))
    historic=json.loads((old/'complete-manifest.json').read_text())
    trained=reorder(read(old/'complete-crosswalk-k20.npz'),prior['ids'])
    with np.load(old/'targets.npz') as f:np.testing.assert_array_equal(prior['ids'],f['ids'])
    earlier_volumes=[]
    for p in sorted(old.glob('draw*-cases*.npz')):
        with np.load(p) as f:
            earlier_volumes.extend(np.exp(f['k'][:,:3].sum(1)).tolist())
    assert len(earlier_volumes)==180920 and len(prior['ids'])==9046
    results={}
    for k in (1,5,20):
        p=by_draw(prior,k);m=by_draw(trained,k);row={'k':k}
        for name,left,right in [('scale',m[0],p[0]),('native',m[2],p[2]),
                  ('max_site',trained['strict_fit'][:,:k].any(1),prior['strict_fit'][:,:k].any(1))]:
            d=100*(left.astype(float)-right)
            row[name]=dict(released_hits=int(left.sum()),zero_hits=int(right.sum()),gained=int((left&~right).sum()),
                lost=int((right&~left).sum()),delta_pp=float(d.mean()),ci95_pp=ci(d,prior['ids']))
        overlap=m[0]&p[0]
        row['normalized_rms']=dict(released=float(m[1][m[0]].mean()),zero=float(p[1][p[0]].mean()),
              both=int(overlap.sum()),common_released=float(m[1][overlap].mean()),common_zero=float(p[1][overlap].mean()))
        results[str(k)]=row
    result=dict(targets=9046,ordered_candidates=180920,model_calls_each={'released':1,'initialization':0},
        zero_candidate_errors=zero_errors,archived_learned_exceptions=historic['audit']['exceptions'],
        archived_learned_minimum_volume=float(min(earlier_volumes)),
        archived_learned_cells_below_point_one=int(sum(v<=.1 for v in earlier_volumes)),metrics=results)
    (HERE/'published-paired.json').write_text(json.dumps(result,indent=2)+'\n')
    print(results,flush=True)

if __name__=='__main__':main()
