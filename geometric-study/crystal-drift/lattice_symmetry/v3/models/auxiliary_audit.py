"""Check stripped inference and training information flow on actual train-only seeds."""
import hashlib,json
import numpy as np
import torch
from . import system_auxiliary as aux
from .train import CACHE,HERE,tensors
from .network import OrbitNet

def main():
    inv=json.loads((CACHE/'inventory.json').read_text())
    classes=aux.labels(inv['eligible'])
    torch.set_num_threads(1);torch.manual_seed(11)
    net=OrbitNet(114,3);head=torch.nn.Linear(114,7)
    with np.load(CACHE/'train-000.npz') as x:
        args=tensors(x,np.arange(8),'cpu')
        rows=x['pidx'][:4]
    called=[]
    hook=net.register_forward_hook(lambda *args:called.append(1))
    ordinary=net(*args)
    with_head=net(*args,return_pool=True)
    hook.remove()
    if not all(torch.equal(a,b) for a,b in zip(ordinary,with_head[:2])):raise RuntimeError('Changing training return changed geometry')
    loss,_,_=aux.loss(head,with_head[-1],2,classes,rows,'cpu');loss.backward()
    shared=float(sum((p.grad.square().sum() for p in net.parameters() if p.grad is not None)).sqrt())
    heads=float(sum((p.grad.square().sum() for p in head.parameters())).sqrt())
    if shared==0 or heads==0 or len(called)!=2:raise ValueError('Auxiliary gradient or call count')
    result=dict(training_ids=len(classes),class_counts=np.bincount(classes,minlength=7).tolist(),
        label_sha256=hashlib.sha256(classes.astype('<i4').tobytes()).hexdigest(),training_only_parameters=sum(p.numel() for p in head.parameters()),
        deployment_parameters=sum(p.numel() for p in net.parameters()),identical_train_branch_geometry=True,
        ordinary_calls=1,training_calls=1,auxiliary_gradient_to_shared=shared,auxiliary_head_gradient=heads)
    (HERE/'system-auxiliary-input-test.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
