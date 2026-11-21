"""Materialize exact equivalences before evaluating other seed and capacity arms."""
from finalists import specs
from register import freeze,model,V3
from reuse import equivalent
if __name__=="__main__":
    for group in ("strict","long","replicated","kernels"):
        frozen=[]
        for tag,folder in specs(group):
            if not (V3/"models/fits"/folder/"selection.json").is_file():continue
            row=model(tag,folder);freeze("weight-"+tag,[row]);frozen.append(row)
            for panel in ("family","shared","test"):equivalent(panel,[row])
        if len(frozen)==len(specs(group)):freeze(group+"-frozen",frozen)

