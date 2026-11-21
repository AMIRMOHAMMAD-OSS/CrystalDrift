"""Fit the official Cartesian uFlowCSP transformer and MeanFlow objective on protected MP-20."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from runtime import HERE, TIERS, atomic_json, counts, model, propose, transport

def read_train(smoke):
    ids = json.loads((HERE.parent/"protocol/splits.json").read_text())["train_ids"]
    parts = sorted((HERE/"train-cache").glob("*.npz"))
    if smoke:
        parts = parts[:1]
    arrays = {key: [] for key in ("ids","x","z","sg")}
    for p in parts:
        with np.load(p) as data:
            for key in arrays:
                arrays[key].append(data[key])
    arrays = {k:np.concatenate(v) for k,v in arrays.items()}
    assert arrays["ids"].tolist() == (ids[:len(arrays["ids"])] if smoke else ids)
    return arrays

def atomic_pt(path, payload):
    temp = path.with_suffix(".part")
    torch.save(payload, temp)
    temp.replace(path)

def stored_resume(path):
    hint=path.parent/"resume-location.json"
    if path.is_file() or not hint.is_file():return path
    record=json.loads(hint.read_text())
    scratch=Path(record["scratch_file"])
    if scratch.is_file() and hashlib.sha256(scratch.read_bytes()).hexdigest()==record["sha256"]:
        return scratch
    return path

def main(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    if dev=="cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    flow = transport()
    net = model(args.tier,dev).train()
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(),lr=args.lr,betas=(.9,.95),weight_decay=0.)
    data = read_train(args.smoke)
    order = np.random.default_rng(args.seed).permutation(len(data["ids"]))
    quarter = math.ceil(len(order)/2)
    common=getattr(args,"common",False)
    if common:
        assert args.seed==11 and not args.smoke and getattr(args,"passes",1)==1
        inventory=json.loads((HERE.parent/"models/cache/inventory.json").read_text())
        allowed=inventory["eligible"]
        lookup={str(mid):i for i,mid in enumerate(data["ids"])}
        subset=[lookup[mid] for mid in allowed]
        data={name:values[subset] for name,values in data.items()}
        scheduled=np.array(inventory["order"])
        assert sorted(scheduled)==list(range(len(allowed))) and len(allowed)==26419
        rng=np.random.default_rng(args.seed)
        order=np.concatenate([scheduled[i:i+128][rng.permutation(len(scheduled[i:i+128]))]
                              for i in range(0,len(scheduled),128)])
        quarter=103*128
    chunks = [indices[j:j+args.batch] for indices in (order[:quarter],order[quarter:])
              for j in range(0,len(indices),args.batch)]
    per_pass = len(chunks)
    passes = getattr(args,"passes",1)
    if passes>1:
        assert not args.smoke and args.seed==11 and passes==8
        for epoch in range(2,passes+1):
            ix_epoch=np.random.default_rng(np.random.SeedSequence([args.seed,epoch])).permutation(len(data["ids"]))
            chunks.extend(indices[j:j+args.batch] for indices in (ix_epoch[:quarter],ix_epoch[quarter:])
                          for j in range(0,len(indices),args.batch))
    nupdates = len(chunks)
    middle = math.ceil(quarter/args.batch)
    if args.smoke:
        chunks = chunks[:args.smoke]
    tag = f"{args.tier}-{args.seed}" + ("-smoke" if args.smoke else "") + ("-common" if common else "")
    initial_out = HERE/"fits"/tag
    out = HERE/"fits"/(tag + ("-long" if passes>1 else ""))
    out.mkdir(exist_ok=True, parents=True)
    split_sha = hashlib.sha256((HERE.parent/"protocol/splits.json").read_bytes()).hexdigest()
    config = dict(tier=args.tier, layers=TIERS[args.tier], parameters=counts(net), seed=args.seed,
                  train_ids=len(data["ids"]), batch=args.batch, lr=args.lr, ema_decay=.99,
                  optimizer="AdamW (0.9,0.95), no weight decay", forward_dtype="float32",
                  max_grad_norm=1., updates=nupdates, half_update=middle, split_sha256=split_sha,
                  source_commit="e87af8483e6d45b0217b4ede430862e992d2b1d0",torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if dev=="cuda" else dev)
    atomic_json(out/"config.json", config)
    log, pos, seen, seconds = [],0,0,0.
    resume_path=out/"resume.pt"
    from_initial=False
    resume_path=stored_resume(resume_path)
    if passes>1 and not resume_path.exists():
        resume_path=stored_resume(initial_out/"resume.pt")
        assert (initial_out/"complete.json").is_file() and resume_path.is_file()
        from_initial=True
    if resume_path.exists():
        state = torch.load(resume_path,map_location="cpu",weights_only=False)
        expected=dict(config)
        if from_initial:
            expected["updates"]=per_pass
        assert state["config"] == expected
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["cpu_rng"])
        if dev=="cuda":
            torch.cuda.set_rng_state(state["gpu_rng"])
        np.random.set_state(state["numpy_rng"])
        log,pos,seen,seconds = state["log"],state["pos"],state["seen"],state["seconds"]
    start = time.perf_counter()
    milestones=(4*per_pass,8*per_pass) if passes>1 else (middle,len(chunks))
    for step, idx in enumerate(chunks[pos:],pos+1):
        n=len(idx)
        x = torch.zeros(n,100,12,device=dev)
        z = torch.zeros(n,100,device=dev,dtype=torch.long)
        x[:,:20] = torch.from_numpy(data["x"][idx]).to(dev)
        z[:,:20] = torch.from_numpy(data["z"][idx]).to(dev)
        sg = torch.from_numpy(data["sg"][idx]).long().to(dev)
        cond = dict(atom_types=z, mask=z!=0, spacegroup=sg, generate_atom_types=False)
        opt.zero_grad(set_to_none=True)
        loss,mse = flow.loss(net,x,cond,ema_model=ema)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
        # As in authors' training_step, update EMA before the current optimizer step.
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(net.parameters()), .01)
        opt.step()
        seen += n
        elapsed = seconds + time.perf_counter()-start
        log.append([step,seen,int((data["z"][idx]>0).sum()),float(loss.detach()),float(mse),float(norm),elapsed])
        if step==1 or step%64==0 or step in milestones:
            print(tag,step,seen,round(float(mse),5),round(elapsed,2),"s",flush=True)
            atomic_json(out/"curve.json",log)
        if step % 128 == 0 or step in milestones:
            atomic_pt(out/"resume.pt",dict(config=config,net=net.state_dict(),ema=ema.state_dict(),
                      opt=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                      gpu_rng=torch.cuda.get_rng_state() if dev=="cuda" else None,numpy_rng=np.random.get_state(),
                      log=log,pos=step,seen=seen,seconds=elapsed))
        if step in milestones:
            path=out/f"ema-{step:05}.pt"
            atomic_pt(path,ema.state_dict())
            record=dict(step=step,seen=seen,wall_seconds=elapsed,
                        max_allocated=torch.cuda.max_memory_allocated() if dev=="cuda" else 0,
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            atomic_json(path.with_suffix(".json"),record)
    # Formula-only sample replay, one-call instrumentation, across two restored instances.
    path=out/f"ema-{len(chunks):05}.pt"
    saved=torch.load(path, map_location="cpu",weights_only=True)
    probe = np.sort(data["z"][:4].copy(), axis=1)
    probe = np.stack([np.pad(row[row>0],(0,20-sum(row>0))) for row in probe])
    ema.eval()
    hook_calls=[]
    hook=ema.register_forward_hook(lambda *_: hook_calls.append(1))
    torch.manual_seed(291)
    first=propose(ema,flow,probe)
    hook.remove()
    net.load_state_dict(saved)
    net.eval()
    torch.manual_seed(291)
    second=propose(net,flow,probe)
    assert len(hook_calls)==1 and all(np.array_equal(a,b) for a,b in zip(first,second))
    atomic_json(out/"complete.json",dict(rows=seen,updates=len(chunks),nfe_per_draw=1,
        probes=len(probe),restored_arrays_identical=True,training_atoms=sum(x[2] for x in log),
        distinct_compositions=len(set(tuple(sorted(row[row>0])) for row in data["z"][order[:seen]])),
        seconds=log[-1][-1],peak_bytes=torch.cuda.max_memory_allocated() if dev=="cuda" else 0))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("tier",choices=("small","medium"))
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--lr",type=float,default=.0003)
    p.add_argument("--smoke",type=int,default=0)
    p.add_argument("--passes",type=int,choices=[1,8],default=1)
    p.add_argument("--common",action="store_true")
    main(p.parse_args())
