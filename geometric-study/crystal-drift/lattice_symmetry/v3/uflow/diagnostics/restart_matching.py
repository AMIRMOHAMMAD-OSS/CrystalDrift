"""Gracefully interrupt only this resumable score pipeline after matcher-code changes."""
import os
import signal
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
eligible={}
replicas='--replicas' in sys.argv
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        argv=(path/'cmdline').read_bytes().split(b'\0')
        cwd=(path/'cwd').resolve()
        script=b'lattice_symmetry/v3/uflow/score_queue.py' if replicas else b'lattice_symmetry/v3/uflow/finish_cpu.py'
        if len(argv)<3 or b'python' not in argv[0] or not any(a.endswith(script) for a in argv):continue
        if replicas and not {b'small-23',b'medium-23',b'small-37',b'medium-37'}.issubset(argv):continue
        if cwd!=HERE.parents[2]:continue
        eligible[int(path.name)]=argv[0].decode()
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
assert 1<=len(eligible)<=5,eligible
print(eligible,flush=True)
for pid in sorted(eligible):
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
