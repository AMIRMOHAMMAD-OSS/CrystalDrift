# Binary crystal fingerprint ablation

The twelve fits are four descriptors times seeds 11, 23, 37. The original 70-entry fingerprint was independently refit at all three seeds and every parameter in each archived checkpoint was reproduced exactly. Only the function mapping a query or reference crystal into features changed. The 58-entry arm drops the 12 real/imaginary triple-product coordinates. The 71-entry signed arm appends the real part of the axial O-type–O-type–Si-type bispectrum (using the lower-Z element twice for arbitrary compositions), divided by √6. The 71-entry redundant arm copies the first real cross-power component, with its original normalization, times 1.987171. This coefficient equates the mean added squared distance on 4,096 fixed training-only, within-composition real/generated pairs. It introduces no new information. [Fixed training loop](../../train.py) · [Independent check and calibration](checks.json)

Every fit uses 1,200 AdamW updates, batch 64, 16 exact-formula blocks of four, 3,630 official training structures and identical sampled random indices/noises at a given seed. Five predictions per record are drawn as the prefix of the original 20-candidate RNG streams. All 1,231 validation and 1,229 test rows receive energy scores. The standard matcher uses the 128 lowest SHA-256 ID digests per split. The recorded RMS arrays represent every candidate on these selected targets. The contact criterion is a minimum periodic distance of 0.5 Å. All recorded candidates were evaluable, with zero matcher exceptions. Paired case bootstrap intervals fix the trained models and average each target over the three seeds before resampling 10,000 times. The study previously inspected these official splits for other comparisons. Four feature choices and their training budgets remained fixed. The lowest validation energy identified the pair-only arm. [Validation results](val-summary.json) · [Test results](test-summary.json)

Each directory under *runs* contains weights, configurations and SHA-256 records, predictions, per-ID CSVs, and NPZ arrays with target and generated physical features, evaluation masks, matcher indices and per-candidate normalized RMS. Plotting source writes the editable SVG and vector PDF into the paper directory.

~~~sh
python checks.py
python run.py
python score.py
python figure.py
~~~

Run in this directory using NumPy, SciPy, PyTorch, matplotlib and the pinned pymatgen 2024.4.13. The score script accepts an optional isolated dependency path through its site argument. The pre-existing binary data and archived weights reside two directories above.
