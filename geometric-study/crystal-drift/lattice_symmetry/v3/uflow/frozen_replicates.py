"""Full frozen native trial for independent one-pass training seeds, with identical ordered proposal noise."""
import gc
import json
import torch
from argparse import ArgumentParser,Namespace
from datetime import datetime,timezone
from generate import HERE,main as forecast
from runtime import atomic_json

def main(handoff=False):
    gate=HERE.parent/'models/uflow-replication-exclusive.json'
    if handoff and not gate.is_file():raise RuntimeError('GPU handover not yet registered')
    for panels in (['family','shared'],['test']):
        for seed in (23,37):
            for tier in ('small','medium'):
                tag=f'{tier}-{seed}';selection=json.loads((HERE/'fits'/tag/'selection.json').read_text())
                assert selection['cohort']=='dev' and len(selection['results'])==2
                forecast(Namespace(tier=tier,seed=seed,mode='frozen',extended=False,common=False,device='cuda',batch=64,panels=panels))
                gc.collect();torch.cuda.empty_cache()
    if handoff:
        atomic_json(HERE.parent/'models/uflow-replication-finished.json',dict(utc=datetime.now(timezone.utc).isoformat(),
                    tags=['small-23','medium-23','small-37','medium-37'],panels=['family','shared','test'],draws=20))
if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--handoff',action='store_true');main(p.parse_args().handoff)
