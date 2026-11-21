"""Shared native one-call uFlowCSP modules for training, predictions and clocks."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if os.environ.get("UFLOW_TIMM"):
    sys.path.insert(0,os.environ["UFLOW_TIMM"])
elif (HERE/'vendor/timm').is_dir():
    sys.path.insert(0,str(HERE/'vendor'))
import timm
assert timm.__version__ == "1.0.21"
spec = importlib.util.spec_from_file_location("released_uflow", ROOT/"revision/full/external/generate.py")
released = importlib.util.module_from_spec(spec)
spec.loader.exec_module(released)
temporary = tempfile.TemporaryDirectory()
released.official_sources(Path(temporary.name))
from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import MeanFlowSiTRawFormulaAtomwiseSystem
from src.models.meanflow_transport import MeanFlowTransport

TIERS = {"small": (64, 2, 4), "medium": (160, 4, 8), "released": (768, 12, 12)}
chemistry = np.load(ROOT/"revision/full/external/chemistry.npz")

def model(tier, device):
    width, depth, heads = TIERS[tier]
    config = dict(input_size=100, atom_type_vocab_size=120, atom_type_embed_dim=32,
                  coord_dim=3, lattice_dim=9, hidden_size=width, depth=depth, num_heads=heads,
                  mlp_ratio=4, class_dropout_prob=.8, atom_type_condition_dropout=0.,
                  predict_atom_types=False, use_spacegroup_conditioning=False,
                  use_formula_embedding=True, use_atomwise_features=True, use_crystal_system_embedding=True)
    with torch.device("meta" if tier == "released" else "cpu"):
        net = MeanFlowSiTRawFormulaAtomwiseSystem(**config)
    if tier == "released":
        net.load_state_dict(released.safe_weights(Path(os.environ["UFLOW_WEIGHTS"])), strict=True, assign=True)
        from src.models.denoisers.meanflow_sit_raw_formula_atomwise_system import build_sg_to_crystal_system
        net.sg_to_crystal_system = build_sg_to_crystal_system(230)
    net.atomic_feature_table = torch.from_numpy(chemistry["table"].copy())
    return net.to(device)

def transport():
    return MeanFlowTransport(flow_ratio=.25, time_dist=["lognorm", -.4, 1.],
                             cfg_ratio=.8, cfg_scale=1.5, jvp_api="funtorch", atom_loss_weight=0.,
                             torus_coords=False, lattice_per_structure=False, proj_lattice=False,
                             proj_coords_anchor=False, proj_coords_rotavg=False, sym_loss_weight=0.)

def counts(net):
    disabled = sum(p.numel() for p in net.spacegroup_embedder.parameters())
    total = sum(p.numel() for p in net.parameters())
    return dict(registered=total, permanently_inactive=disabled, participating=total-disabled,
                frozen=0, ema_training_copy=total)

def formula_tokens(z):
    z = np.atleast_2d(z)
    key = chemistry["sort_key"]
    toks = np.zeros((len(z), 100), np.int64)
    for i, row in enumerate(z):
        active = row[row>0]
        toks[i,:len(active)] = sorted(active, key=lambda a: (key[a],a))
    return toks

@torch.inference_mode()
def propose(net, flow, z, noise=None, device=None):
    device = next(net.parameters()).device if net is not None else torch.device(device or "cpu")
    z = np.atleast_2d(z)
    types = torch.as_tensor(formula_tokens(z), device=device)
    if noise is None:
        noise = flow.sample_prior(len(z),100,12,"cpu")
    valid = types != 0
    cond = dict(atom_types=types, mask=valid, spacegroup=torch.zeros(len(types),dtype=torch.long,device=device),
                generate_atom_types=False)
    out = flow.sample(net, noise.to(device), cond, num_steps=1) if net is not None else noise.to(device)
    cell = (out[...,3:].masked_fill(~valid[...,None],0).sum(1)/valid.sum(1)[:,None]).view(-1,3,3)
    f = torch.remainder(out[...,:3] @ torch.linalg.pinv(cell),1.).masked_fill(~valid[...,None],0.)
    order = torch.argsort(torch.where(valid,types,120), dim=1, stable=True)[:,:z.shape[1]]
    return (cell.cpu().numpy(), f.gather(1,order[...,None].expand(-1,-1,3)).cpu().numpy(),
            torch.gather(types,1,order).cpu().numpy())

def atomic_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    temp.replace(path)
