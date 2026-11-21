"""Compare the compiled and released CrystalNN descriptors on blind positions."""

import argparse, json, warnings, time
import numpy as np
from score import HERE, arrays, make, fp_tools
from fast_ops import install
from fast_faces import install_faces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    p.add_argument(
        "--canonical",
        action="store_true",
        help="Rotate stock inputs to the released length/angle convention",
    )
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(91567)
    labels = []
    crystals = []
    for arm in (
        "reference",
        "learned-drift",
        "empirical-drift",
        "learned-flow16",
        "learned-uflow",
        "crystalflow-joint",
    ):
        idx = rng.choice(len(arrays(arm)["z"]), size=args.n, replace=False)
        for i in idx:
            if arrays(arm)["z"][i].any():
                crystals.append(make(arm, i))
                labels.append((arm, int(i)))
    fp = fp_tools()[0]

    def evaluate(items):
        output = []
        for c in items:
            try:
                output.append(np.array([fp.featurize(c, i) for i in range(len(c))]))
            except Exception as e:
                output.append(type(e).__name__)
        return output

    comparison = crystals
    if args.canonical:
        from pymatgen.core import Structure, Lattice

        comparison = [
            Structure(Lattice.from_parameters(*c.lattice.parameters), c.species, c.frac_coords)
            for c in crystals
        ]
    t = time.monotonic()
    stock = evaluate(comparison)
    t1 = time.monotonic() - t
    install()
    install_faces()
    fp.featurize(crystals[0], 0)
    t = time.monotonic()
    fast = evaluate(crystals)
    t2 = time.monotonic() - t
    errors = []
    for p, q in zip(stock, fast):
        if isinstance(p, str) or isinstance(q, str):
            assert isinstance(p, str) and p == q
            errors.append(None)
        else:
            errors.append(float(np.nanmax(np.abs(p - q))))
    result = dict(
        per_structure=list(zip(labels, errors)),
        stock_s=t1,
        compiled_s=t2,
        max_error=max(x for x in errors if x is not None),
    )
    path = HERE / ("fp-canonical-parity.json" if args.canonical else "fp-parity.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("parity", result["max_error"], "stock/compiled", t1, t2)
    assert result["max_error"] < 1e-5


if __name__ == "__main__":
    main()
