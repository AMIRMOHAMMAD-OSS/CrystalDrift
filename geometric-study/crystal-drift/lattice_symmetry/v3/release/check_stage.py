"""Exercise interrupted finalization and bytewise change detection with tiny fixture payloads."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import build_evidence as pack

def main():
    with TemporaryDirectory() as folder:
        work=Path(folder);source=work/'array.npz';source.write_bytes(b'first-array')
        stage=work/'staging';final=work/'research.zip';tex=pack.BASE+'paper/example.tex'
        entries={'scaled/example.npz':source,tex:b'working version'}
        pack.manifest_sources=lambda *args:entries.copy()
        def build(prepare=False):pack.archive(final,'full',None,0,None,stage,prepare)
        build(True)
        assert not final.exists()
        with ZipFile(stage/'payloads.zip','a') as z:z.writestr('interrupted-append',b'incomplete')
        assert list(pack.read_stage(stage,entries))==['scaled/example.npz']
        original=source.stat()
        source.write_bytes(b'other-array')
        os.utime(source,ns=(original.st_atime_ns,original.st_mtime_ns))
        try:pack.read_stage(stage,entries)
        except ValueError:pass
        else:raise AssertionError('A changed scientific array passed.')
        source.write_bytes(b'first-array')
        entries[tex]=b'frozen version';entries['scaled/late.npz']=b'new data'
        build()
        # A scientific replay may uncover a stale downstream table after the ZIP
        # CRC check. Reuse the frozen prefix for a corrected unverified attempt.
        entries[tex]=b'corrected version';build()
        with ZipFile(final) as z:
            assert z.testzip() is None and len(z.namelist())==4
            assert z.read(tex)==entries[tex] and z.read('scaled/example.npz')==b'first-array'
            assert z.read('scaled/late.npz')==b'new data'
        receipt=final.with_suffix('.receipt.json');data=json.loads(receipt.read_text())
        receipt.write_text(json.dumps(dict(data,clean_replay_verified=True)))
        try:build()
        except FileExistsError:pass
        else:raise AssertionError('Verified prior release was not protected.')
    print('Passed staged-source, interrupted-append and previous-release checks.')

if __name__=='__main__':main()
