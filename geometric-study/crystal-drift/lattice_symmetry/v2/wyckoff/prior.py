"""Train-only conditional metric prior. Site charts use independent uniform coordinates."""
import collections
from pathlib import Path
import numpy as np
from charts import HERE,chart,compatible_metric,expand,generic_draw
from labels import read

def occupation(template):
    return tuple(sorted((int(z),l) for z,l in template))

def composition(group,template):
    return np.bincount([z for z,l in template],
         weights=[len(chart(group,l)[0]) for z,l in template],minlength=119)

def key(group,template):
    return group,tuple(sorted(l for _,l in template))

def design(z):
    n=z.sum()
    return np.r_[1.,np.log(n),z[1:]/n]

class Prior:
    def __init__(self,rows):
        self.group=collections.defaultdict(list)
        self.pattern=collections.defaultdict(list)
        self.cells=[];X=[];y=[]
        for j,row in enumerate(rows):
            g=row["group"];z=composition(g,row["template"]);L=np.array(row["cell"])
            X.append(design(z));y.append(np.log(abs(np.linalg.det(L))/z.sum()));self.cells.append(L)
            self.group[g].append(j);self.pattern[key(g,row["template"])].append(j)
        X=np.asarray(X);y=np.asarray(y);self.cells=np.asarray(self.cells)
        penalty=np.eye(X.shape[1])*.2;penalty[0,0]=0.
        self.coef=np.linalg.solve(X.T@X+penalty,X.T@y)
        self.residual=y-X@self.coef;self.all=np.arange(len(X))

    def draw(self,group,template,rng):
        z=composition(group,template);idxs=self.pattern.get(key(group,template)) or self.group.get(group)
        idx=rng.choice(self.all if idxs is None else idxs)
        C=self.cells[idx];G=compatible_metric(group,C@C.T)
        logv=np.clip(design(z)@self.coef+self.residual[idx],np.log(3),np.log(160))
        vol=z.sum()*np.exp(logv)
        L=np.linalg.cholesky(G)
        L*=np.cbrt(vol/abs(np.linalg.det(L)))
        u=generic_draw(group,template,rng);f,z=expand(group,template,u)
        return L,f,z

def records():
    rows=list(read("train"))
    assert len(rows)==27136 and len({r["id"] for r in rows})==27136, len(rows)
    return [r for r in rows if "error" not in r]
