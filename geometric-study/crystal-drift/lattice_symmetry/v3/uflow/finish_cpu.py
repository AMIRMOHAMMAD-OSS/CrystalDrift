"""Finish the six frozen confirmations, then the five complete native-law comparisons."""
from evaluate import HERE,evaluate
from score_queue import drain

if __name__=="__main__":
    for tier in ("small","medium"):
        for tag in (f"{tier}-11-common",f"{tier}-23",f"{tier}-37"):
            if not (HERE/"scores"/tag/"frozen/confirm/outcomes.npz").is_file():
                evaluate(tag,"frozen",1,["confirm"])
    drain(["small-11","medium-11","small-11-long","medium-11-long","prior-11"],
          ["confirm","family","shared","test"],2)
