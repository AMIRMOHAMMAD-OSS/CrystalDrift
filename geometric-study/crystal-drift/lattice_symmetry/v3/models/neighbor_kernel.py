"""Scalar integer-image enumeration with the original directed, ordered edge contract."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def scalar_indices(cells, positions, numbers, inverse, cutoff):
    batch_size, sites = numbers.shape
    result = np.empty((max(16, batch_size * sites * 12), 5), dtype=np.int64)
    count = 0
    for b in range(batch_size):
        h = np.empty(3, dtype=np.float64)
        for k in range(3):
            h[k] = cutoff * np.sqrt((inverse[b, :, k] ** 2).sum()) + 1e-10
        for i in range(sites):
            if numbers[b, i] == 0:
                continue
            for j in range(i, sites):
                if numbers[b, j] == 0:
                    continue
                dx = positions[b, j, 0] - positions[b, i, 0]
                dy = positions[b, j, 1] - positions[b, i, 1]
                dz = positions[b, j, 2] - positions[b, i, 2]
                for a in range(math.ceil(-h[0] - dx), math.floor(h[0] - dx) + 1):
                    for v in range(math.ceil(-h[1] - dy), math.floor(h[1] - dy) + 1):
                        for c in range(math.ceil(-h[2] - dz), math.floor(h[2] - dz) + 1):
                            if i == j and a == 0 and v == 0 and c == 0:
                                continue
                            r2 = 0.
                            for k in range(3):
                                x = ((dx+a)*cells[b, 0, k] + (dy+v)*cells[b, 1, k]
                                     + (dz+c)*cells[b, 2, k])
                                r2 += x*x
                            if r2 >= cutoff*cutoff:
                                continue
                            if count+2 > len(result):
                                larger = np.empty((len(result)*2, 5),dtype=np.int64)
                                larger[:count] = result[:count]
                                result = larger
                            result[count,0] = b*sites+i
                            result[count,1] = b*sites+j
                            result[count,2] = a
                            result[count,3] = v
                            result[count,4] = c
                            count += 1
                            if i != j:
                                result[count,0] = b*sites+j
                                result[count,1] = b*sites+i
                                result[count,2] = -a
                                result[count,3] = -v
                                result[count,4] = -c
                                count += 1
    return result[:count]
