"""Test canonical formula ordering and the Cartesian comparator's symmetry limits."""
import json
import numpy as np
import torch
from runtime import HERE,atomic_json,formula_tokens,model,propose,transport

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(47113)
    z=np.zeros((2,20),np.int64)
    z[0,:6]=[8,8,8,12,14,14]
    z[1,:5]=[6,6,6,14,14]
    ordered=torch.from_numpy(formula_tokens(z))
    valid=ordered!=0
    null=torch.zeros(2,dtype=torch.long)
    t=torch.ones(2);r=torch.zeros(2)
    flow=transport()
    noise=flow.sample_prior(2,100,12,"cpu")
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    def rotate(x):
        return (x.reshape(2,100,4,3)@R).reshape(x.shape)
    res={}
    for tier in ("small","medium"):
        fit=HERE/"fits"/f"{tier}-11-long"
        select=json.loads((fit/"selection.json").read_text())
        net=model(tier,"cpu").eval()
        net.load_state_dict(torch.load(fit/select["checkpoint"],map_location="cpu",weights_only=True))
        L,F,Z=propose(net,flow,z,noise)
        Ls,Fs,Zs=propose(net,flow,z[:,::-1].copy(),noise)
        for a,b in ((L,Ls),(F,Fs),(Z,Zs)):np.testing.assert_array_equal(a,b)
        context=dict(t=t,r=r,atom_types=ordered,mask=valid,spacegroup=null)
        velocity=net(noise,**context)
        # Find and swap a pair of identical occupied tokens on each formula.
        perturbed=noise.clone();perms=[]
        for i,tokens in enumerate(ordered):
            repeated=next(a for a in tokens.tolist() if a and (tokens==a).sum()>1)
            a,b=torch.where(tokens==repeated)[0][:2].tolist()
            p=list(range(100));p[a],p[b]=p[b],p[a];perms.append(p)
            perturbed[i]=noise[i,p]
        permuted=net(perturbed,**context)
        expected=torch.stack([velocity[i,p] for i,p in enumerate(perms)])
        site_err=float((permuted-expected).abs().max())
        rotated=net(rotate(noise),**context)
        rotation_err=float((rotated-rotate(velocity)).abs().max())
        assert site_err>1e-4 and rotation_err>1e-4
        res[tier]=dict(checkpoint_sha=select["checkpoint_sha"],formula_reordering_array_errors=0,
                       same_species_velocity_commutator_max=site_err,quarter_turn_velocity_commutator_max=rotation_err)
    atomic_json(HERE/"native-symmetry.json",dict(seed=47113,scope="fixed-noise pathwise checks of the Cartesian EMA comparator",
                formula_order="sort by electronegativity then atomic number",results=res))

if __name__=="__main__":main()
