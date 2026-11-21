"""Release read caches for sealed backup volumes and lower only the study's CPU scoring priority."""
from pathlib import Path
import os
import sys

def study_jobs():
    names=("evaluation/farm.py","evaluation/paired.py","evaluation/shards.py","evaluation/follow_scores.py",
           "uflow/score_queue.py","evaluation/wave.py","models/prepare.py","evaluation/series.py")
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd=(p/"cmdline").read_bytes().decode(errors="replace").replace("\0"," ")
            cwd=(p/"cwd").resolve()
            if "geometric-study/crystal-drift" not in str(cwd) and "geometric-study/crystal-drift" not in cmd:continue
            if not (p/"comm").read_text().startswith("python"):continue
            if any(x in cmd for x in names):yield int(p.name)
        except (OSError,RuntimeError):pass

if __name__=="__main__":
    for pid in study_jobs():
        try:os.setpriority(os.PRIO_PROCESS,pid,max(8,os.getpriority(os.PRIO_PROCESS,pid)))
        except (PermissionError,ProcessLookupError):pass
    total=0
    for p in Path(sys.argv[1]).glob("pack-*.zip"):
        with p.open("rb") as f:
            os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        total+=p.stat().st_size
    print("Sealed backup bytes released from the Linux page cache",total,flush=True)
