"""Resumable full-traversal one-call fits under the locked v3 protocol."""
import argparse, hashlib, json, os, platform, shutil, sys, time
from functools import lru_cache
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import torch
from .network import OrbitNet,TIERS,make_model
from .sampler import HERE,V3,ROOT,Sampler,weight_path
from .prepare import CACHE,make_shard,safe_save
from ..geometry.descriptors import feature_blocks,combine,drift_loss
from lattice_symmetry.v2.physical import blocks
torch.set_num_threads(1)

def sha(p):
    h=hashlib.sha256()
    with weight_path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def save_pt(path,data):
    p=path.with_suffix('.building')
    def size(value):
        if torch.is_tensor(value):return value.nelement()*value.element_size()
        if isinstance(value,dict):return sum(size(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(size(v) for v in value)
        return 0
    need=size(data)+1024*1024
    while shutil.disk_usage(path.parent).free<need:
        print('waiting for checkpoint space',path.name,need,flush=True)
        time.sleep(20)
    torch.save(data,p);os.replace(p,path)

def base_dim(arm):
    b=blocks()
    return b['radial' if arm=='pair' else 'power0' if arm=='center0' else 'power2'].stop

def tensors(batch,at,device):
    ans=[]
    for j in range(8):
        a=batch['x'+str(j)][at]
        t=torch.as_tensor(a,device=device,dtype=torch.long if j in (2,3) else torch.float32)
        ans.append(t)
    return ans


@lru_cache(maxsize=2)
def validation_service(device,split_digest):
    # The immutable prior is reused across registered fits. Every target still enters
    # the original formula -> symbolic seed -> pack -> neural forward -> host path.
    service=Sampler(device)
    service.rows=();service.allz=None;service.by_formula={}
    return service

def cached_dev(net,panel):
    return (net is not None and next(net.parameters()).device.type=='cuda' and panel=='dev'
            and (CACHE/'dev-gpu-parity-ok.json').exists() and (CACHE/'dev-inputs.npz').exists())


def dev(s,net,path,panel='dev',force_cache=False):
    saved=CACHE/(panel+'-inputs.npz')
    if (force_cache or cached_dev(net,panel)) and net is not None and saved.exists():
        with np.load(saved) as db:
            ids=db['ids'];arr=[db['x'+str(i)] for i in range(8)]
        device=next(net.parameters()).device;Z=arr[2];newL=[];newF=np.zeros((*Z.shape,3),np.float32)
        with torch.inference_mode():
            for start in range(0,len(Z),40):
                take=slice(start,start+40);n=int((Z[take]>0).sum(1).max())
                inputs=[torch.as_tensor(v[take] if i==0 else v[take,:n],device=device) for i,v in enumerate(arr)]
                l,f=net(*inputs);newL.extend(l.cpu().numpy());newF[take,:n]=f.cpu().numpy()
        with path.with_suffix('.building').open('wb') as file:
            np.savez_compressed(file,ids=ids,L=np.array(newL).reshape(-1,5,3,3),F=newF.reshape(-1,5,20,3),Z=Z.astype(np.int16).reshape(-1,5,20))
        os.replace(path.with_suffix('.building'),path);return
    ids=json.loads((V3/'protocol/splits.json').read_text())[panel+'_ids']
    with np.load(ROOT/'scaled/mp20-data/val.npz') as src:
        z=src['z'];lookup={str(s):i for i,s in enumerate(src['ids'])}
    cells=[];sites=[];elements=[]
    for start in range(0,len(ids),8):
        seeds=[]
        for mid in ids[start:start+8]:
            i=lookup[mid]
            seed=int(np.random.SeedSequence([38471,i]).generate_state(1)[0])
            seeds.extend(s.seeds(z[i],5,seed))
        desc=s.apply(net,seeds)
        for L,F,Z in desc:
            cells.append(L);sites.append(np.pad(F,((0,20-len(Z)),(0,0))))
            elements.append(np.pad(Z,(0,20-len(Z))))
    with path.with_suffix('.building').open('wb') as file:
        np.savez_compressed(file,ids=ids,L=np.array(cells).reshape(-1,5,3,3),
                            F=np.array(sites).reshape(-1,5,20,3),Z=np.array(elements,dtype=np.int16).reshape(-1,5,20))
    os.replace(path.with_suffix('.building'),path)


def meanflow(net,x,goalL,goalF,rng,return_info=True):
    L,F,Z,O,m,free,J,P=x
    b=len(L);device=L.device
    u=rng.random((3,b));t=torch.as_tensor(np.maximum(u[0],u[1]),device=device,dtype=L.dtype)
    r=torch.as_tensor(np.where(u[2]<.30,np.maximum(u[0],u[1]),np.minimum(u[0],u[1])),device=device,dtype=L.dtype)
    vL=-goalL;vF=-goalF
    xL=L+(1-t[:,None,None])*goalL;xF=F+(1-t[:,None,None])*goalF
    def velocity(a,f,h):
        return net(a,f,Z,O,m,free,J,P,r,h,residual=True)
    # u points from the current state toward data (minus the paper's signed average velocity).
    _,derivative=torch.autograd.functional.jvp(velocity,(xL,xF,t),(vL,vF,torch.ones_like(t)),create_graph=False)
    pL,pF=velocity(xL,xF,t)
    interval=(t-r)[:,None,None]
    targetL=goalL-interval*derivative[0];targetF=goalF-interval*derivative[1]
    cell=((pL-targetL)/4).square().mean()
    site=(((pF-targetF)@L)/4).square().sum((1,2)).div((Z>0).sum(1)*3).mean()
    info=([dict(cell_loss=float(cell.detach()),site_loss=float(site.detach()),jvp='exact reverse-over-reverse')]
          if return_info else None)
    return .5*(cell+site),info


def one_fit(a):
    separate_time=getattr(a,'separate_time_rng',False)
    use_aux=getattr(a,'system_auxiliary',False)
    published=getattr(a,'published',False)
    label=f'{a.tier}-{a.arm}-{a.affinity}-{a.seed}'+('-local' if a.local else '')+('-published' if published else '')
    if separate_time:label+='-commonorder'
    if use_aux:label+='-systemaux'
    base_out=HERE/'fits'/label
    out=base_out if a.passes==1 else HERE/'fits'/(label+'-8pass')
    if a.profile:out=HERE/'profiles'/label
    if (out/'complete.json').exists():return
    out.mkdir(parents=True,exist_ok=True)
    with np.load(CACHE/'calibration.npz') as sc:
        temps=tuple(sc['temperatures']);stats={k:(sc[k+'_mean'],sc[k+'_scale']) for k in ('base','contacts','variance','reciprocal')}
    radial4=None
    if a.arm=='radial4':
        with np.load(CACHE/'radial4.npz') as extra:
            radial4=extra['values'];stats['radial4']=(extra['mean'],extra['scale'])
    inventory=json.loads((CACHE/'inventory.json').read_text())
    total=inventory['n'];shards=(total+127)//128
    device=torch.device(a.device);rng=np.random.default_rng(a.seed)
    event_rng=np.random.default_rng(a.seed+74419) if separate_time else rng
    torch.manual_seed(a.seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    arch=dict(width=TIERS[a.tier][0],depth=TIERS[a.tier][1],global_context=not a.local)
    if published:arch=dict(published_width=24 if a.tier=='tiny' else 180)
    if a.tier=='reference' and device.type=='cuda':
        # Avoid temporarily duplicating the 136-million-parameter model in host RAM.
        with torch.device(device):net=make_model(arch)
    else:net=make_model(arch).to(device)
    count=sum(p.numel() for p in net.parameters())
    head=None;params=list(net.parameters())
    if use_aux:
        from . import system_auxiliary as aux
        if a.arm!='angular' or published:raise ValueError('The registered auxiliary arm is the angular orbit model')
        classes=aux.labels(inventory['eligible'])
        head=torch.nn.Linear(arch['width'],7).to(device)
        params+=list(head.parameters())
    optim=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.99),weight_decay=.001)
    milestones={shards//2:'half',shards:'full'} if a.passes==1 else {4*shards:'epoch4',8*shards:'epoch8'}
    s=None;hist=[];start=time.perf_counter();before=0.;step=0;first=0;atoms=0;anchors=0
    plan=np.arange(shards)
    if a.seed!=11:
        np.random.default_rng(a.seed+1802).shuffle(plan)
    plan=np.tile(plan,a.passes)
    if a.profile:plan=plan[:1];milestones={}
    resume=out/'resume.pt';deployment=out
    scratch=os.getenv('CRYTALDRIFT_SCRATCH')
    if scratch and not a.profile:
        run_directory=Path(scratch).resolve()/out.name
        run_directory.mkdir(parents=True,exist_ok=True)
        resume=run_directory/'resume.pt'
        if a.tier in ('medium','large','reference'):deployment=run_directory
        (out/'resume-location.json').write_text(json.dumps(dict(path=str(resume),
             description='Atomic optimizer and RNG checkpoint. Deployment weights or their per-milestone checksum pointers are beside this receipt.'),indent=2))
    origin=resume if resume.exists() or not (out/'resume.pt').exists() else out/'resume.pt'
    if not origin.exists() and a.passes>1:
        if not (base_out/'complete.json').exists():raise FileNotFoundError('Complete the paired first traversal before its extension')
        origin=base_out/'resume.pt'
        pointer=base_out/'resume-location.json'
        if not origin.exists() and pointer.exists():origin=Path(json.loads(pointer.read_text())['path'])
        if not origin.exists():raise FileNotFoundError('The paired one-traversal optimizer is required to continue its trajectory')
    if origin.exists():
        if a.tier=='reference':r=torch.load(origin,map_location='cpu',weights_only=True,mmap=True)
        else:r=torch.load(origin,map_location=device,weights_only=True)
        assert r['arch']==arch and r['split']==inventory['split_sha256']
        net.load_state_dict(r['net']);optim.load_state_dict(r['optimizer']);first=r['shard']
        if head is not None:head.load_state_dict(r['system_head'])
        rng.bit_generator.state=r['rng'];hist=r['history'];before=r['seconds'];step=r['step'];atoms=r['atoms'];anchors=r['anchors']
        if separate_time:event_rng.bit_generator.state=r['event_rng']
        del r
    print(label,'params',count,'train',total,'start shard',first,'device',device,flush=True)
    def state(j):
        return dict(architecture=arch,arch=arch,net=net.state_dict(),shard=j,step=step,anchors=anchors,atoms=atoms,
                    split=inventory['split_sha256'],rng=rng.bit_generator.state,event_rng=event_rng.bit_generator.state,seconds=before+time.perf_counter()-start,
                    history=hist,method=a.arm,affinity=a.affinity,seed=a.seed,temperatures=tuple(map(float,temps)))
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    profile_durations=[]
    for pos,index in enumerate(plan):
        if pos<first:continue
        epoch=pos//shards
        folder=CACHE if epoch==0 else CACHE/f'epoch-{epoch:02}'
        fn=folder/f'train-{index:03}.npz'
        bank_scratch=os.getenv('CRYTALDRIFT_BANK_SCRATCH')
        mirror=Path(bank_scratch)/f'epoch-{epoch:02}'/fn.name if epoch and bank_scratch else None
        if not fn.exists() and mirror and mirror.exists():fn=mirror
        if fn.exists():
            with np.load(fn) as data:shard={k:data[k] for k in data.files}
        elif epoch:
            s=Sampler(device=device) if s is None else s
            shuffled=np.random.default_rng(55171+epoch).permutation(total)
            shard=make_shard(s,shuffled[index*128:index*128+128],int(index),epoch,in_memory=True)
            if mirror:
                mirror.parent.mkdir(parents=True,exist_ok=True)
                if shutil.disk_usage(mirror.parent).free>2**20:safe_save(mirror,**shard)
        else:raise FileNotFoundError(fn)
        length=len(shard['indices']);order=rng.permutation(length)
        for j in range(0,length,a.batch):
            if a.profile:
                if device.type=='cuda':torch.cuda.synchronize(device)
                step_start=time.perf_counter()
            ix=order[j:j+a.batch];n=len(ix)
            log_step=step==0 or (step+1)%50==0
            at=(ix[:,None]*4+np.arange(4)).ravel();args=tensors(shard,at,device)
            pidx=(ix[:,None]*2+np.arange(2)).ravel()
            if a.arm=='meanflow':
                # Two training seeds per anchor and two randomly paired exact-composition truths.
                choose=np.arange(n*4).reshape(n,4)[:,:2].ravel()
                x=[v[choose] for v in args]
                gL=torch.as_tensor(shard['targetL'][at[choose]],device=device)
                gF=torch.as_tensor(shard['targetF'][at[choose]],device=device)
                loss,info=meanflow(net,x,gL,gF,event_rng,return_info=log_step)
            else:
                if head is None:L,F=net(*args)
                else:L,F,pool=net(*args,return_pool=True)
                raw=feature_blocks(L,F,args[2],args[3],a.arm)
                y=combine(raw,stats).reshape(n,4,-1)
                source={k:torch.as_tensor(shard['p_'+k][pidx] if k!='radial4' else radial4[np.asarray(shard['pidx'])[pidx]],device=device) for k in raw}
                source['base']=source['base'][:,:base_dim(a.arm)]
                p=combine(source,stats).reshape(n,2,-1)
                observed=drift_loss(y[:,:2],p,y[:,2:],a.affinity,temps,return_info=log_step)
                loss,info=observed if log_step else (observed,None)
                if head is not None:
                    crossentropy,logits,targets=aux.loss(head,pool,n,classes,shard['pidx'][pidx],device)
                    loss=loss+aux.COEFFICIENT*crossentropy
                    if log_step:info.append(dict(system_crossentropy=float(crossentropy.detach()),
                                    system_accuracy=float((logits.argmax(-1)==targets).float().mean())))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True)
            rate=a.lr*min(1.,(step+1)/16.)
            for group in optim.param_groups:group['lr']=rate
            optim.step()
            step+=1;anchors+=n;atoms+=int((shard['x2'][at[::4]]>0).sum())
            if a.profile:
                if device.type=='cuda':torch.cuda.synchronize(device)
                profile_durations.append(time.perf_counter()-step_start)
            if log_step:
                hist.append(dict(step=step,anchors=anchors,loss=float(loss.detach()),grad=float(grad),affinity=info,
                                 seconds=before+time.perf_counter()-start))
                print(step,anchors,'loss',round(float(loss.detach()),4),'s',round(before+time.perf_counter()-start,1),flush=True)
        jcompleted=pos+1
        restart_interval=48 if a.tier=='reference' else 12
        if (jcompleted%restart_interval==0 or jcompleted in milestones) and not a.profile:
            obj=state(jcompleted)
            extra={'system_head':head.state_dict()} if head is not None else {}
            save_pt(resume,dict(obj,optimizer=optim.state_dict(),**extra))
        if jcompleted in milestones:
            which=milestones[jcompleted];ckpt=deployment/(which+'.pt')
            save_pt(ckpt,obj)
            if deployment!=out:
                distro=os.getenv('WSL_DISTRO_NAME','Ubuntu')
                receipt=dict(path=str(ckpt),windows_path='//wsl.localhost/'+distro+str(ckpt),sha256=sha(ckpt),
                        bytes=ckpt.stat().st_size,anchors=anchors,step=step,seconds=obj['seconds'])
                (out/(which+'.pt.pointer.json')).write_text(json.dumps(receipt,indent=2))
            check_service=None if cached_dev(net,'dev') else validation_service(str(device),inventory['split_sha256'])
            dev(check_service,net,out/(which+'-dev.npz'))
            print('saved',which,'at',anchors,'positive anchors',flush=True)
    elapsed=before+time.perf_counter()-start
    summary=dict(label=label,architecture=arch,parameters=count,training_atoms=atoms,distinct_anchors=min(anchors,total),anchor_exposures=anchors,
                 updates=step,reference_exposures=2*anchors,query_exposures=2*anchors,
                 negative_exposures=0 if a.arm=='meanflow' else 2*anchors,
                 seconds=elapsed,torch=str(torch.__version__),device=str(device),
                 gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                 peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                 split_sha256=inventory['split_sha256'],calibration_sha256=sha(CACHE/'calibration.npz'),float='FP32',
                 checkpoint_sha256={f:sha(deployment/(f+'.pt')) for f in milestones.values() if (deployment/(f+'.pt')).exists()},
                 lr=a.lr,batch=a.batch,platform=platform.platform(),history=hist)
    summary['periodic_optimizer_shards']=48 if a.tier=='reference' else 12
    if a.profile:
        warm=profile_durations[1:]
        summary['profile']=dict(warm_updates=len(warm),first_update_s=profile_durations[0],
             mean_s=float(np.mean(warm)),p50_s=float(np.median(warm)),p95_s=float(np.quantile(warm,.95)),
             update_s=profile_durations,fp32_full_traversal_updates=sum((min(128,total-i)+a.batch-1)//a.batch for i in range(0,total,128)))
    if head is not None:
        summary['system_auxiliary']=dict(training_only_parameters=sum(p.numel() for p in head.parameters()),
             total_training_parameters=sum(p.numel() for p in params),coefficient=aux.COEFFICIENT,
             target_source='spglib 0.1 angstrom, protected MP-20 training structures from uflow/train-cache',
             classes=aux.SYSTEM_NAMES,training_class_counts=np.bincount(classes,minlength=7).tolist(),
             eligible_label_sha256=hashlib.sha256(classes.astype('<i4').tobytes()).hexdigest(),
             inference='classifier discarded; generator state dictionary only, one native orbit forward with the ordinary formula-derived seed')
    (out/'complete.json').write_text(json.dumps(summary,indent=1))
    if s is not None:s.shortlist.cache_clear()
    if not a.profile and resume!=out/'resume.pt' and resume.exists():
        (out/'resume-location.json').write_text(json.dumps(dict(path=str(resume),sha256=sha(resume),
            bytes=resume.stat().st_size,step=step),indent=2))
    print('complete',label,elapsed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tier',choices=TIERS,default='small')
    p.add_argument('--arm',choices=['pair','center0','angular','variance','contacts','reciprocal','meanflow','radial4'],default='angular')
    p.add_argument('--affinity',choices=['laplace','gaussian'],default='laplace');p.add_argument('--seed',type=int,default=11)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--device',default='cuda');p.add_argument('--local',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--passes',type=int,choices=[1,8],default=1)
    p.add_argument('--published',action='store_true',help='Use the width-scaled v2 site graph')
    p.add_argument('--system-auxiliary',action='store_true',help='Training-only 7-system classifier on the pooled orbit embedding')
    p.add_argument('--separate-time-rng',action='store_true',help='Match all minibatch orders to drift, including subsequent shards')
    one_fit(p.parse_args())

