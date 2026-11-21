"""Show local durable interim staging, data recovery and host storage status."""
from pathlib import Path
import json,shutil
D=Path('/home/amir/crytaldrift-v3-evidence/recovered-ephemeral/crytaldrift-v3-20261121')
for name in ('recovery-volumes.log','rebuilt-backup.log','bank-recompute.log','bank-restoration.log'):
    p=D/name
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-300));tail=f.read().decode(errors='replace').splitlines()
        print(name,tail[-1] if tail else 'starting')
print('received-packs',len(list((D/'drive-backup-recovery-base').glob('pack-*.zip'))),'new-packs',
      len([p for p in (D/'drive-backup-final').glob('pack-*.zip') if int(p.stem[5:])>=318]),
      'live-banks',len(list((D/'banks').glob('epoch-*/train-*.npz'))),'host-free',shutil.disk_usage('/mnt/c').free)
