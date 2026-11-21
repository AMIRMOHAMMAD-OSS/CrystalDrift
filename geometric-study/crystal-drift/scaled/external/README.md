# Released uFlowCSP on the multielement test cohorts

Each cohort uses the 128 held-out IDs with the smallest SHA-256 digests, in exactly the order returned by the scaled study's selection function. The external model receives the elements and their exact cell counts. The reference coordinates, cell and space group are withheld. It draws five independent prior samples per condition in one ordered run with seed 2026. All 640 outputs in each cohort enter scoring before any filtering, ranking or relaxation. [Shared selection](../experiment.py)

The original authors' Cartesian network and MeanFlow sampler run with 100 padded tokens, one model evaluation per candidate, the null symmetry label and the released EMA parameters. The sampler orders elements by electronegativity. Decoding averages the active lattice tokens, converts the Cartesian sites into fractional coordinates and restores ascending atomic-number order. Stored predictions have arrays of shape 128 × 5 × 6 for log-Gram coefficients and 128 × 5 × 20 × 3 for fractional coordinates. Supplied element counts and material IDs accompany them. The source commit, package versions, hashes, thread count and inference duration appear in each cohort's replay manifest. Earlier two-site and binary-cohort replays use this same checkpoint and pinned source. [Pinned authors' code](https://github.com/usccolumbia/uFlowCSP/tree/e87af8483e6d45b0217b4ede430862e992d2b1d0) · [Published Figshare checkpoint](https://doi.org/10.6084/m9.figshare.33974422.v1) · [Checkpoint provenance and earlier runs](../../uflowcsp-results/README.md)

All selected MP-20 IDs are in the authors' published test-ID list and none occurs in their train-ID list. WBM uses a distinct identifier namespace. A disjoint identifier crosswalk across those datasets cannot establish disjoint crystal structures, reduced formulas or prototypes. The authors' release training script is for MP-20. The WBM results measure transfer to these selected WBM records. [Pinned training script](https://github.com/usccolumbia/uFlowCSP/blob/e87af8483e6d45b0217b4ede430862e992d2b1d0/scripts/train.sh)

## Results

| Test cohort | Recovery on draw 1 | Recovery through draw 5 | Contact ≥0.5 Å | Fair environment score on native training scales |
|---|---:|---:|---:|---:|
| Full MP-20 | 63/128 (49.22%) | 86/128 (67.19%) | 610/640 | 0.9812 |
| WBM | 59/128 (46.09%) | 94/128 (73.44%) | 619/640 | 0.9419 |

The 95% case-bootstrap intervals for five-draw external recovery were 59.38–75.00% on MP-20 and 65.63–81.25% on WBM. Every candidate was numerically evaluable and the matcher reported no exceptions. The same WBM predictions scored 0.8005 using MP-20 training scales for the transfer analysis. Inference used two CPU threads while local experiments were also running. These elapsed times do not provide a controlled latency comparison. [MP-20 paired estimates](mp20/paired.json) · [WBM native comparison](wbm/paired.json) · [MP-20-to-WBM comparison](wbm/paired-mp20-to-wbm.json)

## Reproduction

From this directory, supply the verified EMA file produced by the shared extractor. Install the same dependency versions recorded in the replay manifests. The optional argument `--site` accepts one or more isolated dependency directories.

~~~sh
python replay.py --weights /path/to/ema.safetensors --cohort mp20 wbm --threads 2 --batch 12
python assess.py --cohort mp20 wbm
~~~

The common structure matcher writes the normalized RMS of every matched candidate and a null value for misses. The physical evaluator uses the same four environment quantities, empirical training scales and finite-ensemble correction as the local scaled models. Paired 95% percentile intervals resample 128 held-out IDs 10,000 times after averaging the available local seed fits for each target. The external arm has one released checkpoint and one inference seed. WBM's paired comparisons to MP-20-trained local models use the MP-20 training feature scales for both arms. The files record the included seeds and the direction of each difference. [Matcher](../match.py) · [Environment score](../evaluate.py) · [MP-20 per-case matcher results](mp20/uflow-seed2026-test-matching.json) · [WBM per-case matcher results](wbm/uflow-seed2026-test-matching.json)
