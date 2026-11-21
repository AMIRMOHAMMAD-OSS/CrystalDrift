"""Sample Windows host CPU utilization and RAM alongside measurable new CUDA episodes."""
import argparse
import csv
import time
from pathlib import Path
import psutil
HERE=Path(__file__).resolve().parent

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--seconds",type=int,default=7200);p.add_argument("--interval",type=int,default=10)
    args=p.parse_args();path=HERE/"gpu/host-measurements.csv"
    start=time.time()
    with path.open("a",newline="") as stream:
        rows=csv.writer(stream)
        if stream.tell()==0:rows.writerow(("epoch_seconds","windows_cpu_utilization_pct","available_ram_bytes","free_commit_bytes"))
        while time.time()-start<args.seconds and not ((HERE/"gpu/power0-settings.json").is_file() and (HERE/"interleaved/settings.json").is_file()):
            cpu=psutil.cpu_percent(interval=args.interval)
            rows.writerow((time.time(),cpu,psutil.virtual_memory().available,psutil.swap_memory().free));stream.flush()
