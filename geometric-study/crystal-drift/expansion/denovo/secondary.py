"""Score a separately sampled 100-step CrystalFlow pool without changing the five primary arms."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARM = "crystalflow-joint-local"


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=ROOT, check=True)


def geometry_bank_ready():
    try:
        with np.load(HERE / f"pools/{ARM}.npz") as b:
            return (
                b["z"].shape == (10000, 20)
                and len(b["ids"]) == 10000
                and len(b["valid_output"]) == 10000
            )
    except (FileNotFoundError, zipfile.BadZipFile, EOFError, ValueError):
        return False


def timing_ready():
    try:
        record = json.loads((HERE / f"pools/{ARM}.json").read_text())
        return len(record["b1_ms"]) == 128 and record["elapsed_s"] > 0
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the independently generated raw archive to finish writing",
    )
    p.add_argument("--fingerprint-workers", type=int, default=5)
    args = p.parse_args()
    while not geometry_bank_ready():
        if not args.wait:
            raise SystemExit("The independent 10,000-draw archive is incomplete.")
        time.sleep(20)
    with ThreadPoolExecutor(max_workers=2) as executor:
        fingerprints = executor.submit(
            run,
            "score.py",
            "--kind",
            "fp",
            "--compiled",
            "--chunk",
            20,
            "--workers",
            args.fingerprint_workers,
            "--arms",
            ARM,
        )
        novelty = executor.submit(run, "novelty.py", "--workers", 2, ARM)
        run("score.py", "--kind", "raw", "--chunk", 100, "--workers", 2, "--arms", ARM)
        run("report.py", "--arms", ARM)
        run("property_graphs.py", "--workers", 2, ARM)
        run("property_eval.py", ARM)
        fingerprints.result()
        novelty.result()
    if args.wait:
        while not timing_ready():
            time.sleep(15)
    run("report.py", "--arms", ARM)
    panel = json.loads((HERE / f"panel/{ARM}.json").read_text())
    if "w1_formation_energy" not in panel:
        # The 1,000 valid rows can change if a raw-valid structure fails featurization.
        run("property_graphs.py", "--workers", 2, ARM)
        run("property_eval.py", ARM)
        run("report.py", "--arms", ARM)
    run("write_paper.py")
    run("validate.py")


if __name__ == "__main__":
    main()
