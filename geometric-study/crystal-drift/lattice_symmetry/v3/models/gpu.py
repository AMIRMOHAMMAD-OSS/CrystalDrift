"""Local WSL launch wrapper; appends an optional Numba site without shadowing Torch."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
handoff=Path(__file__).resolve().with_name('takeover.once')
if __name__=='__main__' and handoff.exists():
    handoff.rename(handoff.with_name('takeover.completed'))
    raise SystemExit('Registered queue handoff completed before starting the next fit.')
sys.path.insert(0,str(root))
if os.environ.get('CRYTALDRIFT_NUMBA_SITE'):
    sys.path.append(os.environ['CRYTALDRIFT_NUMBA_SITE'])
import runpy
runpy.run_module('lattice_symmetry.v3.models.train',run_name='__main__')

