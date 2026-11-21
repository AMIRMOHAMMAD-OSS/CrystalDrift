"""Check invariances, the two-site inverse, and a controlled periodic seam."""
import json
from pathlib import Path

import torch

from model import WAVES, CrystalGenerator, chemistry, crystal_features, drifting_field


def multisite_collision():
    """Two different eight-site binary motifs with the same implemented 70 features."""
    subsets = ((0, 1, 4, 6), (0, 1, 3, 7))
    z = torch.tensor([[8] * 4 + [14] * 4] * 2)
    diagonal_gram = 100 * torch.tensor([1., 2.**.5, 3.**.5])
    k = torch.zeros(2, 6)
    k[:, :3] = diagonal_gram.log() / 2
    f = torch.zeros(2, 8, 3)
    gap_sets, directed_differences = [], []
    for j, subset in enumerate(subsets):
        s = torch.tensor(subset, dtype=torch.long)
        f[j, :, 0] = (s / 12.).repeat(2)
        f[j, 4:, 1] = 1 / 3
        gaps = torch.remainder(s.roll(-1) - s, 12)
        gap_sets.append(gaps.sort().values.tolist())
        directed_differences.append(torch.bincount((s[:, None] - s).remainder(12).ravel(), minlength=12))
    assert torch.equal(*directed_differences)
    assert gap_sets == [[1, 2, 3, 6], [1, 2, 4, 5]]
    features = crystal_features(k, f, z, 'multi')
    assert features.shape == (2, 70)
    max_error = (features[0] - features[1]).abs().max().item()
    assert max_error < 5e-14
    return dict(integer_x_sets=[list(s) for s in subsets], x_denominator=12,
                fractional_y_by_species=[0., 1/3], atoms_per_species=[4, 4],
                atomic_numbers=[8, 14], diagonal_gram_A2=diagonal_gram.tolist(),
                directed_difference_counts_mod_12=directed_differences[0].tolist(),
                sorted_cyclic_gaps_mod_12=gap_sets, feature_length=features.shape[1],
                max_absolute_full_feature_difference=max_error)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(617)
    torch.set_default_dtype(torch.float64)
    z = torch.tensor([[8,8,22,0],[12,30,30,30],[11,17,0,0]])
    k = torch.randn(3, 6)
    f = torch.rand(3,4,3)
    translation = torch.rand(3,1,3)
    perm = torch.tensor([2,1,3,0])
    zp = z[:,perm]
    fp = (f[:,perm] + translation).remainder(1.)
    features = crystal_features(k, f, z, "multi")
    err = (features - crystal_features(k, fp, zp, "multi")).abs().max().item()
    assert err < 2e-13 and torch.equal(chemistry(z, k.dtype), chemistry(zp,k.dtype))
    model = CrystalGenerator().eval()
    cell, sites = model(z,k,f)
    cell_p, sites_p = model(zp,k,fp)
    d = (sites_p - sites[:,perm] - translation + 0.5).remainder(1) - 0.5
    gen_err = max((cell - cell_p).abs().max().item(), d[zp>0].abs().max().item())
    assert gen_err < 2e-13
    z2 = torch.tensor([[8,22],[11,17],[12,30]])
    x = f[:,:2,:].clone()
    phase = crystal_features(k, x, z2, "ab")[:,6:]
    h = len(WAVES)
    angle = torch.atan2(phase[:,h:h+3],phase[:,:3])/(2*torch.pi)
    inverse_err = ((angle - (x[:,1]-x[:,0]) + 0.5).remainder(1)-0.5).abs().max().item()
    assert inverse_err < 2e-13
    refs = torch.randn_like(features)
    q = torch.randn_like(features)
    c = chemistry(z, k.dtype)
    swapped = drifting_field(q,features,refs,c,c)+drifting_field(q,refs,features,c,c)
    anti = swapped.abs().max().item()
    assert anti < 2e-12
    x0, x1 = torch.zeros(1,2,3),torch.zeros(1,2,3)
    x0[0,1,0],x1[0,1,0] = 0.99, 0.01
    def feature_distance(rep,a,b):
        va = crystal_features(k[:1],a,z2[:1],rep)
        vb = crystal_features(k[:1],b,z2[:1],rep)
        return (va-vb).norm().item()
    # Equal 0.02 relative displacements across and away from the chart seam.
    y0,y1 = x0.clone(),x1.clone()
    y0[:,1,0],y1[:,1,0] = 0.49,0.51
    seam = {r:[feature_distance(r,x0,x1),feature_distance(r,y0,y1)] for r in ("raw","ab")}
    assert abs(seam["ab"][0]-seam["ab"][1])<1e-12
    assert abs(seam["raw"][0]/seam["raw"][1]-49)<1e-10
    report = dict(descriptor_origin_and_permutation_error=err, generator_equivariance_error=gen_err,
                  recovered_AB_fractional_difference_error=inverse_err, reference_swap_error=anti,
                  equal_periodic_displacement_feature_distances=seam,
                  distinct_eight_site_motifs=multisite_collision())
    Path(__file__).with_name("geometry-checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
