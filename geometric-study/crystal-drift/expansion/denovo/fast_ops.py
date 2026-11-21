"""Accelerate the pinned CrystalNN geometric formulas by compiling their original numeric loops.

The generated module takes its formulas from the installed, version-checked pymatgen
source. The surrounding Voronoi, tabulated chemical radii, weights and site feature
layout remain those of the released evaluator. verify_fast.py checks parity independently.
"""

import ast
from functools import lru_cache
from importlib.metadata import version
import inspect
from pathlib import Path
import re
import sys
import textwrap

import numpy as np

HERE = Path(__file__).resolve().parent


def prepare():
    from pymatgen.analysis.local_env import LocalStructOrderParams

    assert version("pymatgen") == "2024.4.13"
    src = inspect.getsource(LocalStructOrderParams.get_order_parameters)
    block = src.split("        if self._geomops:", 1)[1].split(
        "        # Then, deal with the new-style", 1
    )[0]
    block = block[block.index("            ipi = 1 / pi") :]
    block = textwrap.dedent(block)
    # Each call in CrystalNNFingerprint corresponds to a single OP with given parameters.
    params = sorted(set(re.findall(r"self\._params\[idx\]\[([^\]]+)\]", block)))
    for i, key in enumerate(params):
        block = block.replace(f"self._params[idx][{key}]", f"params[{i}]")
    tree = ast.parse(block)
    species = sorted(
        {
            s.value
            for s in ast.walk(tree)
            if isinstance(s, ast.Constant) and isinstance(s.value, str)
        }
    )

    class Convert(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(species.index(node.value)), node)
            if node.value is None:
                return ast.copy_location(ast.Constant(0.0), node)
            return node

    tree = Convert().visit(tree)
    block = ast.unparse(tree)
    block = block.replace("for idx, typ in enumerate(self._types):", "for idx in range(1):")
    block = block.replace("for (idx, typ) in enumerate(self._types):", "for idx in range(1):")
    block = block.replace("range(len(self._types))", "range(1)").replace(
        "self._comp_azi", "comp_azi"
    )
    block = block.replace("np.inner(", "np.dot(")
    block = block.replace("qsp_theta[idx][j].append(0.0)", "pass").replace(
        "norms[idx][j].append(0)", "pass"
    )
    block = block.replace(
        "max(ops[idx], *qsp_theta[idx][j])", "max(ops[idx], np.max(qsp_theta[idx][j]))"
    )
    block = block.replace("float(sum([x**2 for x in dist]))", "np.sum(dist**2)")
    preamble = "from math import *\nimport numpy as np\nfrom numba import njit\n"
    preamble += (
        "PARAMS=" + repr([ast.literal_eval(s) for s in params]) + "\nTYPES=" + repr(species) + "\n"
    )
    preamble += """@njit(cache=True)
def gramschmidt(v,u):
    return v-(np.dot(v,u)/np.dot(u,u))*u

@njit(cache=True)
def geom(v,typ,params,comp_azi):
    n_neighbors=len(v)
    dist=np.empty(n_neighbors)
    rij_norm=np.empty_like(v)
    for i in range(n_neighbors):
        dist[i]=np.sqrt(np.dot(v[i],v[i]))
        rij_norm[i]=v[i]/dist[i]
    left_of_unity=1.-1.e-12
    very_small=1.e-12
    fac_bcc=1./exp(-0.5)
    gaussthetak=np.zeros(1)
    ops=np.zeros(1)
    qsp_theta=np.zeros((1,n_neighbors,n_neighbors-1))
    norms=np.zeros((1,n_neighbors,n_neighbors-1))
"""
    generated = preamble + textwrap.indent(block, "    ") + "\n    return ops[0]\n"
    path = HERE / "_ops_compiled.py"
    if not path.exists() or path.read_text(encoding="utf8") != generated:
        path.write_text(generated, encoding="utf8")


@lru_cache(maxsize=1)
def install():
    from pymatgen.analysis import local_env as le
    import _ops_compiled as kern

    original = le.LocalStructOrderParams.get_order_parameters

    def dispatch(self, structure, n, indices_neighs=None, tol=0.0, target_spec=None):
        if (
            len(self._types) == 1
            and self._geomops
            and not self._geomops2
            and indices_neighs is not None
        ):
            p = getattr(self, "_kernel_params", None)
            if p is None:
                kind = self._types[0]
                if kind not in kern.TYPES:
                    return original(self, structure, n, indices_neighs, tol, target_spec)
                p = (
                    kern.TYPES.index(kind),
                    np.array([self._params[0].get(key, 0.0) for key in kern.PARAMS], float),
                )
                self._kernel_params = p
            v = np.array([structure[i].coords for i in indices_neighs]) - structure[n].coords
            return [kern.geom(v, p[0], p[1], self._comp_azi)]
        return original(self, structure, n, indices_neighs, tol, target_spec)

    le.LocalStructOrderParams.get_order_parameters = dispatch
    return original


if __name__ == "__main__":
    prepare()
