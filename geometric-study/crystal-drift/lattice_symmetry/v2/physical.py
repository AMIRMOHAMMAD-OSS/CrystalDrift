"""Batched periodic generator and fixed physical feature maps for composition banks."""
import math
import torch
from .neighbors import edges
from ..generator import PeriodicGenerator


class Generator(PeriodicGenerator):
    """Batched version of the prototype; parameter names also accept its state dictionary."""
    def __init__(self, cutoff=4.0, width=24, n_radial=5):
        super().__init__(cutoff, width, n_radial)

    def forward(self, cell, frac, z):
        B,N = z.shape
        s,d,b,vectors = edges(cell,frac,z,self.cutoff)
        scaled = vectors/self.cutoff
        squared = scaled.square().sum(-1,keepdim=True)
        radial = torch.exp(-self.radial_precision * (squared - self.centers).square())
        envelope = (1-squared).clamp_min(0).pow(3)
        mass = cell.new_zeros(B*N,1).index_add(0,s,envelope)
        weight = envelope/(1+mass[s])
        active = (z>0).to(cell.dtype)
        count = active.sum(1)
        elements = self.element(z.flatten())
        messages = self.message(torch.cat((elements[s],elements[d],radial),-1))
        aggregate = torch.zeros_like(elements).index_add(0,s,weight*messages)
        log_v = torch.linalg.slogdet(cell).logabsdet - (count*self.cutoff**3).log()
        inputs = torch.cat((elements,aggregate,log_v[:,None].expand(B,N).reshape(-1,1)),-1)
        hidden = self.site(inputs)
        shift,strain = (weight*self.edge(torch.cat((hidden[s],hidden[d],radial),-1)).tanh()).unbind(-1)
        v = frac.new_zeros(B*N,3).index_add(0,s,shift[:,None]*vectors).reshape(B,N,3)
        v = (v-v.sum(1,keepdim=True)/count[:,None,None])*active[:,:,None]
        df = torch.linalg.solve(cell.transpose(-1,-2),v.transpose(-1,-2)).transpose(-1,-2)
        pooled = (hidden.reshape(B,N,-1)*active[:,:,None]).sum(1)/count[:,None]
        iso = self.dilation(pooled).tanh()[:,:,None]*torch.eye(3,dtype=cell.dtype,device=cell.device)
        dyads = strain[:,None,None]*scaled[:,:,None]*scaled[:,None,:]
        aniso = cell.new_zeros(B,3,3).index_add(0,b,dyads)/count[:,None,None]
        return cell@torch.linalg.matrix_exp(.25*(iso+aniso)), (frac+df).remainder(1)*active[:,:,None]


def blocks(n_radial=2,species_slots=7,angular=True):
    """Named slices in the feature vector; local ranks are compared only at fixed composition."""
    channels = species_slots*n_radial
    lengths = dict(volume=1,chemistry=118,lattice=n_radial,
                   radial=species_slots*(species_slots+1)//2*n_radial)
    if angular:
        for ell in range(3):
            lengths[f"power{ell}"] = species_slots*channels*(channels+1)//2
    start = 0
    out = {}
    for name,length in lengths.items():
        out[name]=slice(start,start+length)
        start+=length
    return out


def harmonics(q):
    """Real regular solid harmonics with unit-sphere addition theorem P_ell(x.y)."""
    x,y,z = q.unbind(-1)
    h = math.sqrt(3.)
    return (torch.ones_like(q[:,:1]),q,
            torch.stack((h*x*y,h*x*z,h*y*z,.5*h*(x*x-y*y),z*z-.5*(x*x+y*y)),-1))


def features(cell,frac,z,angular=True,cutoff=4.0,n_radial=2,species_slots=7):
    """Log volume, counts, lattice and pair shells, plus pooled center-wise power blocks.

    Cells are in Angstroms. Element-pair ranks are sorted by atomic number within each cell.
    Compare local-rank features only within the exact composition of each training bank.
    Vacant rows have element zero. Each batch must contain at least one occupied site per cell.
    """
    B,N = z.shape
    K,R = species_slots,n_radial
    counts = cell.new_zeros(B,119).scatter_add(1,z,(z>0).to(cell.dtype))
    occupied = counts>0
    ranks = occupied.long().cumsum(1)-1
    rank = ranks.gather(1,z).clamp_min(0).flatten()
    # Extra ranks must never alias an adjacent crystal's center block.
    if (occupied.sum(1)>K).any():
        raise ValueError(f"Increase species_slots above {K} for this chemistry")
    s,d,b,vec = edges(cell,frac,z,cutoff)
    q = vec/cutoff
    t = q.square().sum(-1,keepdim=True)
    centers = torch.linspace(0,1,R,dtype=cell.dtype,device=cell.device)
    window = torch.exp(1-1/(1-t).clamp_min(torch.finfo(cell.dtype).eps))
    phi = window*torch.exp(-.5*((t-centers)*max(1,R-1)).square())
    first = (z>0).long().argmax(1)
    lattice_neighbors = (s==d)&(s%N==first[b])
    lattice = cell.new_zeros(B,R).index_add(0,b[lattice_neighbors],phi[lattice_neighbors])
    distinct = s!=d
    hi = torch.maximum(rank[s],rank[d])
    lo = torch.minimum(rank[s],rank[d])
    width = K*(K+1)//2
    pair = b*width+hi*(hi+1)//2+lo
    radial = cell.new_zeros(B*width,R).index_add(0,pair[distinct],phi[distinct]).reshape(B,-1)
    radial = radial/(2*(z>0).sum(1,keepdim=True))
    out = [torch.linalg.slogdet(cell).logabsdet[:,None], counts[:,1:],lattice,radial]
    if angular:
        C = K*R
        # Group coefficients by center and neighbor species before taking power products.
        center_species = s*K + rank[d]
        pooling = (torch.arange(B,device=z.device)[:,None]*K+rank.reshape(B,N)).flatten()
        flat_mask = (z.flatten()>0).to(cell.dtype)
        denom = cell.new_zeros(B*K).index_add(0,pooling,flat_mask).clamp_min(1)
        left,right = torch.triu_indices(C,C,device=z.device)
        for Y in harmonics(q):
            M = Y.shape[1]
            coeff = cell.new_zeros(B*N*K,R,M).index_add(
                0,center_species,phi[:,:,None]*Y[:,None,:]).reshape(B*N,C,M)
            power = torch.bmm(coeff,coeff.transpose(1,2))[:,left,right]
            average = cell.new_zeros(B*K,power.shape[1]).index_add(0,pooling,power)/denom[:,None]
            out.append(average.reshape(B,-1))
    return torch.cat(out,-1)
