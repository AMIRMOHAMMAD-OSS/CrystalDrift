"""Create the main and supplementary protocol-C LaTeX from saved target-level reductions."""
import json
import csv
import argparse
from collections import Counter
from pathlib import Path
from charts import HERE,chart

def main(preview=False):
    s=json.loads((HERE/"summary.json").read_text());arms=s["arms"]
    P=HERE.parent/"paper";ids=["prior","angular-A","angular-C","orbit-C","DiffCSPpp-S20","DiffCSPpp-S999-test-128x1"]
    names=["Template prior","A angular, template seeds","C angular","C angular + orbit variance","DiffCSP++ (respaced)","DiffCSP++ (native)"]
    after=arms["angular-C"]["retention"]["relaxed"]["0.01"]
    baseline=arms["DiffCSPpp-S20"]
    def line(a,name):
        v=arms[a]; m=v["metadata"];r=v["retention"]["raw"]["0.01"]
        perf=[v["recovery"].get(str(k)) for k in (1,5,20)]
        mr=[f"{100*u['match_rate']:.1f}" if u else "--" for u in perf]
        rm=[f"{u['matched_rms']:.3f}" if u and u["matched_rms"] is not None else "--" for u in perf]
        last=v["retention"].get("relaxed",{}).get("0.01",{})
        return " & ".join([name,f"{m['nfe']:,}",*mr,*rm,f"{100*r['valid']/r['n']:.1f}",str(last.get("contains","--"))])+r"\\"
    ds=arms["angular-C"]
    pair=s["paired"]["prior-20"];c=arms["angular-C"]["recovery"]["20"]
    q=arms["orbit-C"]["recovery"]["20"]
    sample=json.loads((HERE/"example.json").read_text())
    text=r"""\section{Generation with a supplied occupied Wyckoff template}\label{sec:wyckoff}
The second task supplies a Hall setting and the multiset of element-labelled occupied Wyckoff letters. A letter can occur more than once. These inputs determine conventional-cell site counts, but provide neither the cell volume nor any free coordinate. We use the first 128 official test identifiers in their SHA-256 ordering. Their expanded conventional cells contain up to 60 sites. For each occupied orbit, we independently sample its free coordinates, expand the tabulated coset images and draw a compatible metric from a train-only prior. We evaluate a frozen formula-trained angular map on these seeds and fit a second angular map on expanded conventional crystals conditional on their supplied template. These are protocol C results. Protocol A supplies only the unit-cell composition.

\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.4pt}
\caption{Prediction from the same supplied Wyckoff templates on 128 test targets. MR (\%) and mean best normalized RMS use ordered raw draws. RMS averages the matched targets. Valid (\%) is the structural-validity fraction among all draws. Retained is the number of the first 16 targets whose first draw retains the requested subgroup after unconstrained CHGNet relaxation at 0.01\,\AA. The native DiffCSP++ row evaluates one draw.}\label{tab:wyckoff}
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrr}\hline
 & & \multicolumn{3}{c}{MR} & \multicolumn{3}{c}{RMS}& & \\
Method & NFE & 1 & 5 & 20 & 1 & 5 & 20 & Valid & Retained\\\hline
"""+"\n".join(line(a,n) for a,n in zip(ids,names))+r"""
\hline\end{tabular}}\end{table}
"""
    text+=(f"The C-trained angular map recovered {c['hits']} targets at twenty draws, compared with {arms['prior']['recovery']['20']['hits']} "
           f"for its train-fitted metric prior and random Wyckoff coordinates used alone. "
           f"The paired recovery difference was {-100*pair['MR_difference']:.2f} percentage points "
           f"(95\\% target-bootstrap interval $[{-100*pair['ci'][1]:.2f},{-100*pair['ci'][0]:.2f}]$). "
           f"Adding the variance across repeated orbits recovered the same {q['hits']} targets. "
           f"Their mean matched RMS values were {c['matched_rms']:.3f} and {q['matched_rms']:.3f}.\n")
    text+=r"""DiffCSP++ supplies an independent predictor using the same group, letters and species \cite{diffcspplus}. We replayed its released MP-20 weights with 20 predictor--corrector iterations (40 neural calls) at all three draw budgets and repeated the first-draw comparison with the complete 999-iteration sampler (1,998 calls). The abbreviated sampler preserves the authors' coordinate projections and indexes their learned noise predictions at 20 equally spaced original time indices. It uses the corresponding noise marginals and respaced lattice posterior variance. SGFM also constrains each trajectory by supplied orbit information \cite{sgfm}.

"""
    if not preview or (HERE/"full/summary.json").is_file():
        full=json.loads((HERE/"full/summary.json").read_text())
        text+=(f"Across all {full['cohort']:,} official test records, one draw from the template prior alone recovered "
           f"{full['hits'][0]:,} targets ({100*full['hits'][0]/full['cohort']:.2f}\\%), and one C-angular proposal recovered "
           f"{full['hits'][1]:,} ({100*full['hits'][1]/full['cohort']:.2f}\\%). "
           f"The seven rejected template round trips remain in the denominator. The paired angular-minus-prior difference was "
           f"{100*full['paired_angular_minus_prior']:.2f} points ($[{100*full['paired_95'][0]:.2f},{100*full['paired_95'][1]:.2f}]$).\n")
    raw=ds["retention"]["raw"]["0.01"]
    text+=(f"At 0.01\\,\\AA, the explicit site-orbit and metric tests accepted all {raw['n']:,} C-angular draws. "
           f"Independent symmetry finding contained the requested operations on {raw['contains']:,} draws and assigned "
           f"the exact requested group type on {raw['exact']:,}. The detected operation set equalled the requested set in its supplied setting "
           f"on {raw['exactops']:,} draws. After relaxation, {after['contains']}/16 retained the requested subgroup, "
           f"{after['exactops']}/16 recovered its operation set and {after['exact']}/16 returned the requested group type. "
           "The raw guarantee concerns the operations shared by a template-compatible seed. Additional symmetry can occur, "
           "and unconstrained relaxation can alter the group.\n")
    text+=r"""\begin{figure}[!htbp]\centering\includegraphics[width=.94\linewidth]{wyckoff-retention.pdf}
\caption{Requested subgroup containment, exact finite operation set in the supplied setting, and exact international group type at 0.01\,\AA. Error bars give 95\% bootstrap intervals resampling whole target IDs. The left panel expands the range from 99.6 to 100\% and includes twenty draws on each of 128 targets. The right panel follows the first draw from the first sixteen targets per method. An optimizer step cap retains its last observable geometry.}\label{fig:wyckoff-retention}
\end{figure}
"""
    (P/"wyckoff.tex").write_text(text,encoding="utf8")
    test=json.loads((HERE/"template-tests.json").read_text())
    supplemental=r"""\section{Wyckoff charts, equivariance and evaluation}\label{app:wyckoff}
\paragraph{Setting and orbit decoder.}
For a conventional cell we use one specified Hall setting per group, tabulated by PyXtal 1.0.9 \cite{pyxtal}. If the row coordinate of an orbit representative is $s(u)=a+uB$, its affine chart has $\operatorname{rank}B\in\{0,1,2,3\}$. Coordinates are reduced modulo one. The table supplies the maps $s_j(u)=a_j+u B_j$ produced by acting on $s$ with one representative of each coset of its site stabilizer. For a right action, $sg=sh$ holds precisely when $gh^{-1}$ is in the stabilizer. Each independent occupied orbit is drawn once, with a common chemical element on all images. We reject coincident orbit sites or collisions between occupied orbits to fractional tolerance $10^{-7}$ and redraw the parameters of the new orbit up to 32 times. The program reports a failed proposal if this bound is reached. Its random continuous prior gives exceptional boundaries zero probability in the free directions. Fixed zero-degree charts and conventional-cell centering translations are included explicitly. A specialization or additional symmetry can still occur in the generated geometry.

For a row point operation $R$, compatibility requires $RGR^\top=G$. Given an SPD proposal $H$, the average $\overline G=|\mathcal G|^{-1}\sum_{R,t}RHR^\top$ remains SPD and is invariant because multiplication permutes the operation list. We rescale to the sampled volume and take a Cholesky factor. A species-preserving fractional operation of a compatible seed induces an orthogonal physical frame action. The paired equivariance theorem applied to the corresponding site permutation and its fractional translation shows that its operations remain a subgroup of the output stabilizer. This implication includes inversion and improper rotations. It does not imply that the detected group is exactly the requested group, and an arbitrary continuous unconditioned seed has no prescribed nontrivial stabilizer.

\paragraph{Label extraction and training.}
We run spglib 2.5.0 at 0.01\,\AA\ on the actual MP-20 input structure, imposing the PyXtal Hall setting on a second call where necessary \cite{spglib}. The native and conventional atom maps are joined through their primitive site indices. We recover a representative of each species-labelled crystallographic orbit by solving its affine chart over the possible wraps and verify the expanded coset set with a minimum-distance assignment. Verification includes conventional-to-native species ratios, volume per site, the compatible metric, and the public primitive-cell StructureMatcher. We retain the material ID and the reason for every rejection.
"""
    ls=s["labels"]
    supplemental+=(f"The train, validation and test partitions retained {ls['train']['accepted']:,}/{ls['train']['total']:,}, "
                   f"{ls['val']['accepted']:,}/{ls['val']['total']:,} and {ls['test']['accepted']:,}/{ls['test']['total']:,} "
                   "round trips, respectively. All 39 rejections failed the final StructureMatcher check. "
                   "All 128 nominated test targets passed. Changing the symmetry-finding tolerance from 0.01 to 0.001 and 0.1\\,\\AA\ changed "
                   "the group type on three and zero nominated test records, respectively. The corresponding validation counts were six and five. "
                   "The release gives orbit counts by crystal system and free-coordinate dimension.\n")
    supplemental+=r"""The common metric prior samples the shape of a training cell with the same group and Wyckoff-letter multiset when available, falls back first to group and then to the training archive, and applies the group average. A ridge regression on training element fractions and log site count models log volume per atom, with an empirical donor residual. Uniform free parameters supply the site coordinates. The conditional angular objective samples four training rows with replacement per update. For each row's exact labelled template it independently draws three data references, three generated negatives and three live queries. The positive-sampling archive contains 26,821 distinct templates, of which 26,591 have only one record. The supervised template establishes orbit multiplicity and site dimension; repeated sampling in a singleton bank supplies no additional geometries. Both C objectives use the same 6,723-parameter physical generator, shared train-only scaling, AdamW at $10^{-3}$, the same initialization and training draws, and 512 updates. We select among updates 0, 64, 256 and 512 by validation MR at five draws on 64 fixed IDs, breaking ties by lower matched RMS. The global and orbit-augmented objectives selected 256 and 512 updates, respectively.

\paragraph{Orbit statistic and independent checks.}
For each center and order $\ell=0,1,2$ we sum the real harmonics of the weighted neighbor vectors and contract over their components. We average this scalar separately over each occupied orbit. Group orbits by their element and their multiplicity. Within each class compute the variance of these means, then average over the occupied orbits and concatenate the three standardized entries with the physical fingerprint. A simultaneous transport of sites and orbit membership preserves each per-orbit mean. Reordering the orbits permutes terms inside the symmetric reduction. A change of setting transports these same orbit sets and preserves their multiplicities under a unimodular transformation. This statistic avoids an ordering by origin-dependent Wyckoff letters. Cell enlargement or merging occupations changes its domain. A categorical template label is constant within a template bank. The extra block supplies continuous information whenever equivalent-size orbits of the same species have distinct environments.
"""
    supplemental+=(f"All {test['positions']:,} tabulated charts in 230 settings reproduced the orbit of a generic point under independently "
        "supplied spglib Hall operations. Maximum torus-coordinate discrepancy was $3.4\\times10^{-16}$. "
        "Thirty-nine hand-chosen and random templates tested fixed inversion centers, mirror planes, free lines, a screw, a glide, "
        "centering and repeated same-species orbits under a restoring checkpoint, a forward call, rebasing and an optimizer update. "
        "The largest paired cell or site difference was $3.6\\times10^{-15}$ and the added statistics agreed within $2.1\\times10^{-16}$.\n")
    supplemental+=r"""\paragraph{Detection and relaxation.}
For each tolerance we compare every requested operation with the independently detected spglib operations in the supplied setting, pairing equal fractional point matrices and requiring Cartesian translation discrepancies below the tolerance. We also compare the requested and detected international group numbers. As a separate direct check we expand each orbit from its first generated site, compare the two finite sets in both directions and test the isometry defect of each cell action. For $Q=L^{-1}RL$, we use $\max_j|\sigma_j(Q)-1|$ times the longest supplied lattice vector as a metric residual in \AA. The motif residual uses Cartesian distances at the assigned fractional wraps, and the check uses the larger residual. On every audited raw and relaxed cell, $0.1\max_j\|(L^{-1})_{:j}\|<0.044$, so every image vector shorter than a tested tolerance lies inside these wraps. For resolved occupancy we also require the shortest intersite or nonzero image distance to exceed the tolerance, so sites that coalesce there cannot count as separate occupants. We transport the initial origin by the mean wrapped site displacement when analysing the unconstrained final cell. All candidates retain their original indices. The explicit parameterized subgroup and the numerical recovery by symmetry finding can therefore be checked independently. With CHGNet 0.3.0 we relax sites and cell at zero applied pressure, using 100 FIRE steps and a generalized force threshold of 0.1 eV/\AA. Screened groups and raw groups have different sample sizes. No competing phase energies enter this retention experiment.

\begin{figure}[!htbp]\centering\includegraphics[width=.76\linewidth]{wyckoff-example.pdf}
\caption{The first multispecies nontrivial-group C-angular draw in the preselected relaxation cohort, for the template of MP-20 record mp-20959 (group 127). The two views share a physical scale. Raw and optimized coordinates and forces are archived. The raw nearest distance was 2.35\,\AA\ and the final distance was 2.63\,\AA.}\label{fig:wyckoff-example}
\end{figure}
"""
    timing=json.loads((HERE/"template-timing.json").read_text())["methods"]
    supplemental+=r"""\begin{table}[!htbp]\centering\footnotesize
\caption{C-trained angular predictions under three tolerances. Counts refer to all 2,560 raw draws from 128 templates, and 16 final CHGNet geometries from the first draw of the first 16 templates. The orbit column checks the expected occupancy within each supplied orbit under its own requested operations, including repeated occupations.}\label{tab:wyckoff-tolerance}
\begin{tabular}{lrrrrrrrr}\hline
 & \multicolumn{4}{c}{Raw (2,560)} & \multicolumn{4}{c}{After relaxation (16)}\\
Tolerance (\AA)&Subgroup&Set&Type&Orbit&Subgroup&Set&Type&Orbit\\\hline
"""
    for tol in (".001",".01",".1"):
        key=str(float(tol));rr=ds["retention"]["raw"][key];post=ds["retention"]["relaxed"][key]
        vals=[x[v] for x in [rr,post] for v in ("contains","exactops","exact","occupancy")]
        supplemental+=tol+" & "+" & ".join(f"{v:,}" for v in vals)+r"\\"+"\n"
    supplemental+=r"""\hline\end{tabular}\end{table}
"""
    audit=json.loads((HERE/"operation-order-audit.json").read_text())
    diff={t:{stage:sum(v[stage]["counts"][t][1] for v in audit.values()) for stage in ("raw","relaxed")} for t in ("0.001","0.01","0.1")}
    denom={stage:sum(v[stage]["total"] for v in audit.values()) for stage in ("raw","relaxed")}
    supplemental+=("We match requested translations injectively to detected translations for each equal rotation. "
        "Containment together with equal operation counts, including centering translations, gives equality of these finite sets modulo lattice translations at the stated tolerance. "
        f"At 0.001, 0.01 and 0.1\\,\\AA, finite-set equality and international type equality differed on "
        f"{diff['0.001']['raw']}, {diff['0.01']['raw']} and {diff['0.1']['raw']} of {denom['raw']:,} raw candidates. "
        f"Their corresponding disagreement counts after relaxation were {diff['0.001']['relaxed']}, {diff['0.01']['relaxed']} and {diff['0.1']['relaxed']} of {denom['relaxed']}.\n")
    data=list(csv.DictReader((HERE/"strata.csv").open()))
    series=[(r["stratum"],int(r["n"])//20) for r in data if r["method"]=="angular-C" and r["stage"]=="raw" and r["category"]=="system"]
    targets=json.loads((HERE/"test-cohort.json").read_text())
    dofs=Counter(int(chart(r["group"],l)[2].sum()) for r in targets for _,l in r["template"])
    supplemental+=("The external-comparator cohort contained "+", ".join(f"{n} {name.lower()}" for name,n in series)
                   +" targets. The occupied orbits included "+", ".join(f"{dofs[d]} with {d}" for d in range(4))
                   +" free coordinates. The archived stratification counts every template under each degree it contains.\n")
    failures=[(r["stratum"],int(r["n"])-int(r["subgroup"])) for r in data if r["method"]=="angular-C" and r["stage"]=="raw" and r["category"]=="system" and int(r["n"])>int(r["subgroup"])]
    supplemental+="Independent detection missed the requested subgroup at 0.01\\,\\AA\ on "+ " and ".join(f"{n} {name.lower()} draw"+("s" if n!=1 else "") for name,n in failures)+".\n"
    external=arms["DiffCSPpp-S20"]["metadata"];native=arms["DiffCSPpp-S999-test-128x1"]["metadata"]
    supplemental+=(f"On one Intel Core i7-12700H CPU thread, median batch-one latency across the first 16 test templates was "
        f"{timing['prior']['median_ms']:.2f} ms for the prior, {timing['angular-C']['median_ms']:.2f} ms for C angular, "
        f"and {timing['DiffCSPpp-S20']['median_ms']:.1f} ms for the 40-call DiffCSP++ sampler. "
        f"At batch 16, their measured throughputs were {timing['prior']['throughput']:.1f}, "
        f"{timing['angular-C']['throughput']:.1f} and {timing['DiffCSPpp-S20']['throughput']:.1f} cells/s, respectively. "
        "The clock covers supplied-template decoding and all model calls through returned host arrays after two warm-ups. "
        "Other CPU jobs shared the host. Independently, generating the archived 2,560 respaced DiffCSP++ candidates took "
        f"{external['seconds']:.1f} s on the RTX 3060 Laptop GPU. The 128 native-schedule draws took {native['seconds']:.1f} s.\n")
    (P/"wyckoff-supplement.tex").write_text(supplemental,encoding="utf8")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--preview",action="store_true");main(p.parse_args().preview)
