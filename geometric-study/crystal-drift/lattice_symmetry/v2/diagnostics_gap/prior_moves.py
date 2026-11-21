"""Compare the first site and cell predictions to their identical input seeds."""
import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
V2 = OUT.parent
ROOT = V2.parents[1]


def first_prediction(method, name):
    path = V2 / f'csp/test/{method}-seed11/{name}.npy'
    return np.load(path, mmap_mode='r')[:, 0]


def main():
    # csp/predict.py uses SeedSequence([410932, seed, target-block start]) for every arm.
    # The untransformed prior archive therefore stores the seeds taken by the angular generator.
    f0, f1 = [first_prediction(arm, 'f') for arm in ['prior', 'angular']]
    k0, k1 = [first_prediction(arm, 'k') for arm in ['prior', 'angular']]
    with np.load(ROOT / 'revision/full/mp20/cohort.npz') as cohort:
        z = cohort['z']
    n = (z > 0).sum(1)
    stretch = np.zeros((len(z), 3, 3))
    stretch[:, range(3), range(3)] = k0[:, :3]
    i, j = np.triu_indices(3, 1)
    stretch[:, i, j] = stretch[:, j, i] = k0[:, 3:] / np.sqrt(2)
    w, q = np.linalg.eigh(stretch)
    cell = (q * np.exp(w)[:, None]) @ q.swapaxes(1, 2)
    delta = f1 - f0
    delta = delta - delta.round()
    # Length of the wrapped chart displacement in the input physical cell.
    # In highly oblique charts, this representative need not be the shortest translation.
    lengths = np.linalg.norm(delta @ cell, axis=-1)
    volume_change = np.expm1(k1[:, :3].sum(1) - k0[:, :3].sum(1))
    with (OUT / 'prior_moves.csv').open('w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['sites','targets','site_count','median_displacement_A','p90_displacement_A','median_relative_volume_change'])
        for key, ix in [('all', n > 0), ('1-4', n <= 4), ('5-8', (n >= 5)&(n <= 8)),
                        ('9-12', (n >= 9)&(n <= 12)), ('13-20', n >= 13)]:
            d = lengths[ix][z[ix] > 0]
            writer.writerow([key, int(ix.sum()), len(d), np.median(d), np.quantile(d, .9), np.median(volume_change[ix])])


if __name__ == '__main__':
    main()
