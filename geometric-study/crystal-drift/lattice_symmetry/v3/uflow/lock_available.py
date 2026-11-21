"""Freeze every currently complete development sweep, with no access to test or confirmation arrays."""
from evaluate import HERE,evaluate

if __name__=="__main__":
    for done in sorted((HERE/"fits").glob("*/complete.json")):
        fit=done.parent
        if "smoke" in fit.name or (fit/"selection.json").exists():continue
        ready=list((HERE/"predictions"/fit.name).glob("ema-*/dev/ready.json"))
        if len(ready)==2:
            evaluate(fit.name,"dev",1,None)
