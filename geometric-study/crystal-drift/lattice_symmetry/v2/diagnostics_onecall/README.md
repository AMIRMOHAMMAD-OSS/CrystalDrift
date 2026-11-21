# One-call volume calibration

This post-release experiment starts from the frozen angular checkpoint chosen at 256 updates. It fits one composition-to-volume readout from all 27,136 official MP-20 training structures. Each predicted cell is isotropically rescaled. Its shape and fractional atomic sites are retained. The readout evaluates a fixed formula using fitted coefficients and element counts. The generation network is evaluated once.

Let \(x_a\) be the element fractions and \(N\) the supplied unit-cell atom count. Start with the released composition ridge prediction of log volume per atom. Fit its training residual by ridge regression on \(2\sqrt{x_ax_b}\) for unordered distinct element pairs, penalty 1. Add \(\log N\) to obtain a predicted cell log-volume \(\mu\). For a generated lattice \(L\) assign

\[
\log \tilde V = 0.75\mu + 0.25\log|\det L|,\qquad
\tilde L = \exp\!\big[(\log \tilde V-\log|\det L|)/3\big]L .
\]

The coefficient 0.25 and the penalty were fixed before scoring the pilots. Because the scale factor depends only on the cell volume and composition, it commutes with Cartesian frame and unimodular basis changes, periodic wrapping, and relabeling. The readout can narrow the volume distribution for compositions with several true polymorphs. Its structural-recovery evidence concerns formula-conditioned MP-20 data.

The initial direct check re-ran the pinned pymatgen comparator on the 128 validation targets used in the released model selection, with the same five ordered draws for all three fitted seeds. Native-volume-qualified recovery increased from 9.38% to 15.36%. The paired target-level bootstrap interval for the gain is 3.13–9.38 percentage points. All 1,920 individual scale-normalized outcomes were unchanged. Scaled five-draw recovery at the saved 256, 1,024 and 3,392 updates was 17.71%, 15.89% and 15.36%, respectively. The validation population is small and had already been used in the release’s checkpoint selection.

For all 9,046 official test targets across three seeds and 20 draws each, the main retrospective calculation holds the archived scale-normalized match flags fixed and recalculates the release’s native-volume qualifier, \(|\log(V_{\rm pred}/V_{\rm ref})|\leq\log(1.1)\). Qualified recovery increased from 21.24% to 27.94% (paired 95% interval for the gain 6.33–7.10 percentage points). Mean absolute log-volume error decreased from 0.164 to 0.067. The optional rematch command independently re-runs the pinned matcher on 256 reproducibly sampled test targets for three seeds and all 20 draws. Its results are written to direct_test_check.json when all three seed passes complete.

From the crystal-drift directory:

~~~sh
python lattice_symmetry/v2/diagnostics_onecall/volume_pilot.py fit
"$MATCHER_PYTHON" lattice_symmetry/v2/diagnostics_onecall/volume_pilot.py validation
python lattice_symmetry/v2/diagnostics_onecall/volume_pilot.py archive
"$MATCHER_PYTHON" lattice_symmetry/v2/diagnostics_onecall/volume_pilot.py rematch
~~~

The live inference entry point is load_calibrated_service(seed=11, device="cpu") in the same module. Call the returned service with the same zero-padded atomic-number array and NumPy random generator as the released load_service.
