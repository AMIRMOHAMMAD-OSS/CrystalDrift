"""Suspend or resume only native matching and package replay processes for isolated clocks."""
import argparse
import json
import os
from pathlib import Path
import signal
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parents[1]
root=HERE.parents[2]
def main(mode):
    eligible=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            argv=(path/'cmdline').read_bytes().split(b'\0');cwd=(path/'cwd').resolve()
            if len(argv)<2 or b'python' not in argv[0] or cwd!=root:continue
            if not any(x.endswith((b'lattice_symmetry/v3/uflow/finish_cpu.py',b'lattice_symmetry/v3/release/replay.py',b'lattice_symmetry/v3/release/build_evidence.py')) for x in argv):continue
            eligible.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    assert len(eligible)<=6,eligible
    for pid in eligible:
        try:os.kill(pid,signal.SIGSTOP if mode=='pause' else signal.SIGCONT)
        except ProcessLookupError:pass
    stamp={'mode':mode,'pids':eligible,'utc':datetime.now(timezone.utc).isoformat()}
    with (HERE/'diagnostics/isolated-clock-interval.jsonl').open('a') as f:f.write(json.dumps(stamp)+'\n')
    print(stamp,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pause','resume']);main(p.parse_args().mode)
