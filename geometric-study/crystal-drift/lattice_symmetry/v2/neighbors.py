"""Find detached integer image indices; reconstruct vectors in differentiable Torch code."""
import numpy as np
import torch

try:
    from numba import njit
except ImportError:
    njit = None


def _indices(cells, positions, numbers, inverse, cutoff):
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
                d = positions[b, j] - positions[b, i]
                lo = np.ceil(-h - d).astype(np.int64)
                hi = np.floor(h - d).astype(np.int64)
                for a in range(lo[0], hi[0] + 1):
                    for v in range(lo[1], hi[1] + 1):
                        for c in range(lo[2], hi[2] + 1):
                            if i == j and a == 0 and v == 0 and c == 0:
                                continue
                            r2 = 0.
                            for k in range(3):
                                x = ((d[0]+a)*cells[b, 0, k] + (d[1]+v)*cells[b, 1, k]
                                     + (d[2]+c)*cells[b, 2, k])
                                r2 += x*x
                            if r2 >= cutoff*cutoff:
                                continue
                            if count + 2 > len(result):
                                larger = np.empty((len(result)*2, 5), dtype=np.int64)
                                larger[:count] = result[:count]
                                result = larger
                            result[count, 0] = b*sites+i
                            result[count, 1] = b*sites+j
                            result[count, 2] = a
                            result[count, 3] = v
                            result[count, 4] = c
                            count += 1
                            if i != j:
                                result[count, 0] = b*sites+j
                                result[count, 1] = b*sites+i
                                result[count, 2] = -a
                                result[count, 3] = -v
                                result[count, 4] = -c
                                count += 1
    return result[:count]


fast_indices = njit(cache=True)(_indices) if njit is not None else None


def numpy_indices(cells, positions, numbers, inverse, cutoff):
    """Vectorized independent fallback for installations without Numba."""
    blocks = []
    for b, (cell, frac) in enumerate(zip(cells, positions)):
        n = len(numbers[b])
        mask = numbers[b] > 0
        src, dst = np.where(mask[:, None] & mask[None, :])
        d = frac[dst] - frac[src]
        k = np.rint(d).astype(np.int64)
        d -= k
        reach = np.floor(cutoff*np.linalg.norm(inverse[b], axis=0) + .5 + 1e-10).astype(int)
        shifts = np.stack(np.meshgrid(*(np.arange(-x,x+1) for x in reach), indexing="ij"),axis=-1).reshape(-1,3)
        block = max(1, 65536//len(d))
        for offset in range(0,len(shifts),block):
            chunk = shifts[offset:offset+block]
            cart = (d[None,:,:] + chunk[:,None,:]) @ cell
            keep = (cart*cart).sum(-1) < cutoff*cutoff
            keep &= (src!=dst)[None,:] | (chunk!=0).any(1)[:,None]
            im, pair = np.where(keep)
            blocks.append(np.column_stack((b*n+src[pair],b*n+dst[pair],chunk[im]-k[pair])))
    return np.concatenate(blocks)


def edges(cell, frac, z, cutoff):
    """Every directed image vector in an open ball, excluding only the trivial self image."""
    cells = cell.detach().to(device="cpu",dtype=torch.float64).contiguous().numpy()
    positions = frac.detach().to(device="cpu",dtype=torch.float64).contiguous().numpy()
    numbers = z.detach().cpu().contiguous().numpy()
    inverse = np.linalg.inv(cells)
    idx = (fast_indices if fast_indices is not None else numpy_indices)(cells,positions,numbers,inverse,float(cutoff))
    index = torch.as_tensor(idx, device=cell.device)
    s,d = index[:,0],index[:,1]
    batch = s//z.shape[1]
    flat = frac.flatten(0,1)
    delta = flat[d] - flat[s] + index[:,2:].to(frac.dtype)
    vectors = (delta[:,:,None]*cell[batch]).sum(1)
    return s,d,batch,vectors
