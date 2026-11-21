"""Registered measured-capacity decision for one whole comparator-sized traversal."""
import json,os,shutil,tempfile,time
from pathlib import Path
from .sampler import HERE


def strongest():
    candidates=[]
    for arm in ('contacts','angular'):
        fit=HERE/'fits'/f'medium-{arm}-laplace-11-8pass'
        chosen=json.loads((fit/'selection.json').read_text())
        key=Path(chosen['chosen']).stem
        candidates.append((arm,chosen['dev'][key]['hits'],chosen['sha256']))
    return max(candidates,key=lambda row:row[1])[0],candidates


def measure_scratch(scratch,required):
    host=Path('/mnt/c')
    if not host.is_dir():return dict(sufficient=True,scope='native scratch free space')
    chunk=16*2**20;floor=2**30;wrote=0;start=time.monotonic()
    before=shutil.disk_usage(host).free;minimum=before;samples=[]
    if before>=required+floor:return dict(sufficient=True,scope='host already exceeds transient checkpoint and 1 GiB reserve',
                                         host_free_before=before)
    # Real nonzero blocks and fsync distinguish previously allocated VHD extents
    # from virtual guest capacity. Release only this unique staging file.
    df,path=tempfile.mkstemp(prefix='reference-capacity-',suffix='.probe',dir=scratch)
    failure=None
    try:
        with os.fdopen(df,'wb',buffering=0) as f:
            while wrote<required:
                observed=shutil.disk_usage(host).free
                minimum=min(minimum,observed)
                if observed<=floor+2*chunk:break
                size=min(chunk,required-wrote)
                if f.write(os.urandom(size))!=size:raise OSError('Incomplete scratch probe write')
                os.fsync(f.fileno());wrote+=size
                if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),wrote-size,size,os.POSIX_FADV_DONTNEED)
                if wrote%(8*chunk)==0 or wrote==required:
                    free=shutil.disk_usage(host).free;minimum=min(minimum,free)
                    samples.append([wrote,free])
    except OSError as exc:
        failure=str(exc)
    finally:Path(path).unlink(missing_ok=True)
    return dict(sufficient=wrote==required and minimum>=floor,scope='fsynced nonzero Linux ext4 payload',
                required_bytes=required,written_bytes=wrote,host_free_before=before,minimum_host_free=minimum,
                host_free_after=shutil.disk_usage(host).free,reserved_host_margin=floor,samples=samples,
                seconds=time.monotonic()-start,error=failure)


def gate(arm,batch=2):
    base=HERE/'profiles'/f'reference-{arm}-laplace-11'
    measured=json.loads((base/'complete.json').read_text())
    p=measured['profile'];params=measured['parameters'];updates=p['fp32_full_traversal_updates']
    mean_seconds=p['mean_s']*updates
    scratch=Path(os.getenv('CRYTALDRIFT_SCRATCH',str(HERE)))
    available=shutil.disk_usage(scratch).free
    # At the last atomic optimizer replacement only the half deployment exists.
    # The previous optimizer is gone before writing the full deployment.
    required=(1+3+3)*4*params+4*2**20
    host=shutil.disk_usage('/mnt/c').free if Path('/mnt/c').exists() else None
    arm0,dev=strongest()
    if arm!=arm0:raise ValueError('Comparator-sized profile differs from frozen development winner')
    proof=(measure_scratch(scratch,required) if available>=required and mean_seconds<12*3600 else
           dict(sufficient=False,scope='guest space or single-job projection gate'))
    supported=bool(proof['sufficient'])
    return dict(arm=arm,predeclared_dev_choice=dev,parameter_count=params,microbatch=batch,
        profile_updates=measured['updates'],warm_update_count=p['warm_updates'],
        warm_mean_update_s=p['mean_s'],warm_p50_update_s=p['p50_s'],warm_p95_update_s=p['p95_s'],
        projected_updates=updates,projected_single_traversal_s=mean_seconds,
        projected_p95_updates_s=p['p95_s']*updates,measured_peak_allocated_MiB=measured['peak_allocated_MiB'],
        projected_anchors=26419,standard_microbatch_16_updates=1652,optimizer='AdamW, one update per microbatch',
        checkpoint_space_bytes=required,scratch_free_bytes=available,host_free_bytes=host,physical_capacity_check=proof,
        single_job_window_s=12*3600,supported=supported,
        assumptions='FP32 on this device; extrapolates the measured warm update loop, excluding checkpoint I/O, validation, outages and CPU contention. One complete traversal and two deployable milestones with atomic optimizer replacement. WSL virtual free space alone does not guarantee host space.')
