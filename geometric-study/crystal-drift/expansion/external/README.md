# CrystalFlow, the iterative CSP and joint de novo comparator

CrystalFlow has separate MP-20 weights for composition-conditioned prediction and unconditional joint generation. The published conditional network consumes atomic numbers in the requested cell, latent fractional coordinates and six polar lattice coordinates. It uses a fully connected periodic graph and a time embedding. The released unconditional model also predicts atom types. Its Table 3 deposition supplies an independent pool of 10,000 ordered raw structures. [Paper and supplementary benchmark](https://arxiv.org/html/2412.11693v2) · [Released code](https://github.com/ixsluo/CrystalFlow/tree/9c25ff0245d787efd87c9d1d797a8968443608cc) · [Original checkpoints](https://github.com/ixsluo/CrystalFlow/releases/tag/v1.0.0-alpha.1)

## Reproduce the unconditional comparison

Run the following from the project root. The importer verifies the published MD5, retains the original draw numbers and sorts atomic numbers together with their fractional sites. File entry 4731 has a header and no geometry. It remains an empty composition and an explicit missing geometry in the 10,000-row pool. The raw POSCAR entries were used. The separate CONTCAR section contains the structures after the authors' relaxation. Cite the source under CC BY 4.0. [Luo et al., Table 3 deposited structures](https://doi.org/10.6084/m9.figshare.29978608.v3)

~~~sh
mkdir -p /tmp/cry-flow
curl -fL https://ndownloader.figshare.com/files/57393610 -o /tmp/cry-flow/Tab3.txt
python expansion/external/import_joint.py /tmp/cry-flow/Tab3.txt
~~~

## Reproduce the conditional comparison

The deposited CSP checkpoint has SHA-256 b08366eb04d47698da7ea09dbccb1c4a1304a4c3f506fdc1d8c294910ade6dc2. The checkpoint is available unchanged in the authors' release or as the single file identified below. The reader extracts decoder tensors from the verified PyTorch storage and treats the training metadata as inert data. The conditional decoder loads its full state strictly. The code fetcher checks each Python file against its Git blob ID at the recorded commit. Linux, NumPy, requests, and CUDA-enabled PyTorch suffice for the sampler. The common matcher environment is documented in revision/full/README.md. [Single-file archived checkpoint](https://doi.org/10.6084/m9.figshare.34336776)

~~~sh
mkdir -p /tmp/cry-flow
curl -fL https://ndownloader.figshare.com/files/69920037 -o /tmp/cry-flow/conditional.ckpt
python expansion/external/fetch_source.py /tmp/cry-flow/source
python expansion/external/weights.py /tmp/cry-flow/conditional.ckpt /tmp/cry-flow
python expansion/external/check_split.py /tmp/cry-flow/MP20-CrystalFlow-test.csv
python expansion/external/check_reference.py --weights /tmp/cry-flow --source /tmp/cry-flow/source
python expansion/external/check_sampler.py --weights /tmp/cry-flow --source /tmp/cry-flow/source
python expansion/external/crystalflow.py --weights /tmp/cry-flow --source /tmp/cry-flow/source --steps 5 --batch 24 --tf32
python expansion/external/score.py expansion/external/csp-s5 --workers 4
python expansion/external/seal.py expansion/external/csp-s5
python expansion/external/timing.py --weights /tmp/cry-flow --source /tmp/cry-flow/source --steps 5
python expansion/external/score_timing.py expansion/external/timing-s5
python expansion/external/crystalflow.py --weights /tmp/cry-flow --source /tmp/cry-flow/source --steps 100 --batch 24 --tf32 --draws 1 --targets 128
python expansion/external/score.py expansion/external/csp-s100 --workers 1
python expansion/external/step_sensitivity.py
python expansion/external/crystalflow.py --weights /tmp/cry-flow --source /tmp/cry-flow/source --steps 5 --coord-slope 5 --batch 24 --tf32 --draws 1 --targets 128
python expansion/external/crystalflow.py --weights /tmp/cry-flow --source /tmp/cry-flow/source --steps 100 --coord-slope 0 --batch 24 --tf32 --draws 1 --targets 128
python expansion/external/score.py expansion/external/csp-s5-coord5 --workers 1
python expansion/external/score.py expansion/external/csp-s100-coord0 --workers 1
python expansion/external/factorial.py
python expansion/external/paper.py
~~~

The sampler evaluates the pinned CSPNet class definitions. It uses the authors' uniform fractional noise and Gaussian polar lattice initialization with 0.1 standard deviation and a +1 isotropic offset. The Euler update evaluates the velocity at j/S for j from 1 to S. Coordinates are wrapped modulo one after each update. The unguided sampler supplies no crystal-system or target-geometry information. Repeated evaluations of the same reciprocal and fractional embeddings within one decoder call reuse the identical computed tensors. PyTorch index addition and counts implement the graph scatter mean. The first batch checks the exact fully connected edge sequence against the native generator and independently checks the log-Gram output against the authors' matrix exponential.

The local five-step operating point uses the command parser's default unit site-velocity multiplier. The authors' documented 100-step CSP command also passes `--anneal_coords --anneal_slope 5`, multiplying the site velocity by 1+5t while leaving the lattice velocity unchanged. The 100-step sensitivity uses that configuration on paired initial noise. The two settings differ in both integration resolution and site-velocity schedule. The original full-split 100-step publication values are identified separately. Each tile holds up to 256 target IDs for one draw and retains native coordinates and raw volume. The casewise scorer reuses the public-API-audited pymatgen 2024.4.13 protocol of the compact and uFlowCSP arms. The full-split score matrix retains every ID/draw pair. [Authors' evaluation commands](https://github.com/ixsluo/CrystalFlow/tree/9c25ff0245d787efd87c9d1d797a8968443608cc#csp-generation)

For draw index j and zero-based first row i of a generation minibatch, the PyTorch seed is 2062100 + 10000 j + i. Initial fractional coordinates use the CPU generator before transfer to CUDA. Initial lattices use the CUDA generator. Both follow the released sampler. The material IDs use the existing SHA-256 ordering of the MP-20 cohort. Ordered draws remain independent of target matching, physical screening and resource budgets.

The full-split generation pool used FP32 tensors with TF32 matrix multiplication enabled. The separate GPU timing episodes used full IEEE FP32 with TF32 disabled in every arm. The timing program synchronizes CUDA after returning the canonical cell, ordered elements, fractional sites and Cartesian sites to host memory. Five runs per batch shape precede 128 twenty-draw serial episodes and a distinct 10,000-structure workload at batch size 24 with a final batch of 16. The elapsed-time curve is scored from the saved timed serial proposals. Matching, relaxation and checkpoint loading are outside the proposal timer.

## Time the official unconditional checkpoint

The published joint structures and a freshly timed pool are distinct experiments. The unconditional generator samples each site count from the authors' MP-20 size prior, initializes a continuous 28-channel periodic-table encoding at every site, and integrates type, lattice and fractional-site velocities jointly. The authors' released unconditional command uses 100 Euler evaluations and unit velocity multipliers. The timer includes count and noise sampling, sorted species and sites, native decoded cells and Cartesian coordinates returned to host memory. The single-sample workload uses three warm-up draws and 128 measured draws. The 10,000-draw workload uses batch size 24 with a final batch of 16. The scoped archive reader verifies the authors' stored ZIP CRC before exposing weights. [Joint and conditional checkpoints](https://github.com/ixsluo/CrystalFlow/releases/tag/v1.0.0-alpha.1)

The timed joint run invokes the pinned CSPFlow.sample method, including the original graph construction, per-step type decoding and trajectory creation. A native PyTorch scatter mean implements the graph reduction. A candidate optimization that reused edge embeddings produced identical species and coordinates against the 100-step CPU source test, but two GPU executions differed by as much as 0.00103 in a five-step lattice-chart check. The production joint benchmark therefore retains the uncached source sampler.

An additional fixed-noise CUDA audit compared two original and two portable evaluations at each precision. With deterministic reductions enabled, the original and portable implementations agreed exactly on fractional sites and species for both models and both IEEE FP32 and TF32. Native lattice entries agreed within 0.00000060 Å. With ordinary IEEE FP32 reductions, an original-to-original repeat differed by up to 0.00175 in fractional coordinates on the two joint test cells. TF32 also changed the numerical trajectory relative to IEEE FP32. The archived raw arrays retain the exact outcomes used for scoring. The full-test CSP pool used TF32. The timing pools used IEEE FP32.

~~~sh
python expansion/external/fetch_joint.py /tmp/cry-flow
python expansion/external/weights.py /tmp/cry-flow/joint.ckpt /tmp/cry-flow/joint --joint-manifest /tmp/cry-flow/joint-source.json
python expansion/external/check_sampler.py --weights /tmp/cry-flow/joint --source /tmp/cry-flow/source
python expansion/external/joint_timing.py --weights /tmp/cry-flow/joint --source /tmp/cry-flow/source
python expansion/external/verify_joint_local.py --source /tmp/cry-flow/source
python expansion/external/gpu_numerics.py --source /tmp/cry-flow/source --csp /tmp/cry-flow --joint /tmp/cry-flow/joint
~~~
