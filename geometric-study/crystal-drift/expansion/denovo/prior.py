"""Fit train-only empirical and autoregressive discrete composition priors for MP-20."""

from collections import Counter, defaultdict
import numpy as np


def split_formula(row):
    z, n = np.unique(row[row > 0], return_counts=True)
    return tuple(map(int, z)), tuple(map(int, n))


class CompositionPrior:
    """Sample a multiplicity pattern and then chemically ordered species with backoff."""

    def __init__(self, rows, shrinkage=4.0):
        self.rows = rows
        self.elements = np.unique(rows[rows > 0])
        self.elements.setflags(write=False)
        self.patterns = Counter()
        self.order = defaultdict(Counter)
        self.transition = defaultdict(Counter)
        self.alpha = shrinkage
        self.cache = {}
        for row in rows:
            elements, counts = split_formula(row)
            self.patterns[counts] += 1
            previous = 0
            for j, z in enumerate(elements):
                self.order[len(counts), j][z] += 1
                self.transition[counts, j, previous][z] += 1
                previous = z
        self.keys = tuple(sorted(self.patterns))
        self.p = np.array([self.patterns[p] for p in self.keys]) / len(rows)

    def probabilities(self, pattern, j, prev):
        key = (pattern, j, prev)
        if key not in self.cache:
            remaining = len(pattern) - j - 1
            valid = self.elements[self.elements > prev]
            if remaining:
                valid = valid[:-remaining]
            marg = self.order[len(pattern), j]
            base = np.array([marg.get(v, 0) + 0.5 for v in valid])
            base /= base.sum()
            observed = self.transition.get(key, {})
            weights = np.array([observed.get(v, 0) for v in valid]) + self.alpha * base
            self.cache[key] = valid, weights / weights.sum()
        return self.cache[key]

    def sample(self, rng, n, kind):
        if kind == "empirical":
            return self.rows[rng.integers(len(self.rows), size=n)].copy()
        rows = np.zeros((n, 20), np.int16)
        for i, index in enumerate(rng.choice(len(self.keys), size=n, p=self.p)):
            pattern = self.keys[index]
            prev = offset = 0
            for j, count in enumerate(pattern):
                z, weights = self.probabilities(pattern, j, prev)
                prev = int(rng.choice(z, p=weights))
                rows[i, offset : offset + count] = prev
                offset += count
        return rows

    def validate(self, rows):
        losses = []
        missing_patterns = 0
        known_elements = set(self.elements.tolist())
        for row in rows:
            zs, counts = split_formula(row)
            if counts not in self.patterns or not set(zs) <= known_elements:
                missing_patterns += 1
                continue
            lp = np.log(self.patterns[counts] / len(self.rows))
            prev = 0
            for j, z in enumerate(zs):
                valid, weights = self.probabilities(counts, j, prev)
                lp += np.log(weights[np.searchsorted(valid, z)])
                prev = z
            losses.append(-lp)
        return dict(
            rows=len(rows),
            supported=len(losses),
            outside_support=missing_patterns,
            conditional_nll_nats=float(np.mean(losses)),
        )
