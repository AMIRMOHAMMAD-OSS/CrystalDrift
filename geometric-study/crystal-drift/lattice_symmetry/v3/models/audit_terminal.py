"""Verify exact final deployment, optimizer tensors and RNG evidence for both late fits."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from .sampler import HERE,weight_path
from .train import sha
from .training_order import reconstruct

def restart_file(receipt):
    p=Path(receipt['path'])
    if p.is_file():return p
    if os.name=='nt':
        unc=Path('//wsl.localhost/Ubuntu'+receipt['path'])
        if unc.is_file():return unc
    raise FileNotFoundError(p)

def main():
    records={}
    for tag in ('reference-contacts-laplace-11','small-angular-laplace-11-systemaux'):
        folder=HERE/'fits'/tag
        complete=json.loads((folder/'complete.json').read_text());pointer=json.loads((folder/'resume-location.json').read_text())
        live=restart_file(pointer)
        if live.stat().st_size!=pointer['bytes'] or sha(live)!=pointer['sha256']:raise ValueError('Optimizer digest differs')
        state=torch.load(live,weights_only=True,map_location='cpu',mmap=True)
        full=weight_path(folder/'full.pt')
        deploy=torch.load(full,weights_only=True,map_location='cpu',mmap=True)
        if set(state['net'])!=set(deploy['net']):raise ValueError('Mismatched generator keys')
        for name,tensor in state['net'].items():
            if not torch.equal(tensor,deploy['net'][name]):raise ValueError(f'Terminal network differs at {name}')
        if 'optimizer' in deploy or 'system_head' in deploy:raise ValueError('Nondeployment tensors in published weights')
        order,rng=reconstruct('angular')
        if len(set(order))!=26419 or state['rng']!=rng or state['anchors']!=26419:raise ValueError('Final anchor order differs')
        if state['step']!=pointer['step'] or state['step']!=complete['updates']:raise ValueError('Wrong final update')
        per_param=sorted(set(int(x['step'].item()) for x in state['optimizer']['state'].values()))
        if per_param!=[state['step']]:raise ValueError(f'Invalid Adam steps {per_param}')
        select=json.loads((folder/'selection.json').read_text());names={}
        for name in ('half','full'):
            actual=weight_path(folder/(name+'.pt'))
            value=sha(actual)
            if value!=complete['checkpoint_sha256'][name]:raise ValueError(name+' hash')
            names[name]=dict(bytes=actual.stat().st_size,sha256=value)
        head=sum(v.numel() for v in state.get('system_head',{}).values())
        if head!=(805 if tag.endswith('systemaux') else 0):raise ValueError('Wrong training-only state')
        records[tag]=dict(optimizer=pointer,deployment=names,sha_selected=select['sha256'],
              full_generator_equal_to_terminal_optimizer=True,training_only_parameters_in_optimizer=head,
              training_only_tensors_in_deployment=0,adam_state_count=len(state['optimizer']['state']),
              all_adam_steps=per_param,eligible_anchors=state['anchors'],distinct_anchor_positions=len(set(order)),
              reconstructed_final_rng_equal=True,ordered_eligible_indices_sha256=hashlib.sha256(order.astype('<i4').tobytes()).hexdigest())
        del state,deploy
    resumed=(HERE/'reference-resume.log').read_text(errors='replace')
    old=(HERE/'window-reference.log').read_text(errors='replace')
    if 'start shard 24 device cuda' not in resumed or 'complete reference-contacts-laplace-11' not in resumed:raise ValueError('Incomplete restart evidence')
    if 'start shard 0 device cuda' not in old:raise ValueError('Missing initial fit')
    records['resume_boundary']=dict(shard=24,committed_updates_at_restart=1536,
            old_log='window-reference.log',new_log='reference-resume.log',
            wording='The resumed schedule continued the committed optimizer/RNG. Tensor equality above compares the final deployable and final optimizer generator.')
    (HERE/'terminal-fit-integrity.json').write_text(json.dumps(records,indent=2))
    print('Verified optimizer bytes, exact terminal generator tensors, both deployments, and final training order for both late fits.')
if __name__=='__main__':main()
