"""Recover only the published uFlowCSP EMA tensors without loading checkpoint objects."""
import concurrent.futures
import io
import json
import pickletools
import struct
import tempfile
import time
import zipfile
import zlib
from pathlib import Path

import requests
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


HERE = Path(__file__).resolve().parent
RESULTS = HERE/"uflowcsp-results"
CACHE = Path(tempfile.gettempdir())/"codex-uflowcsp-69274087"
URL = "https://ndownloader.figshare.com/files/69274087"
SIZE = 3148174712
BLOCK = 8*1024*1024


def fetch(pair):
    low,high=pair
    name=CACHE/f"{low}-{high}.bin"
    expected=high-low+1
    if name.exists() and name.stat().st_size==expected:
        return name
    for attempt in range(5):
        try:
            response=requests.get(URL,headers={"Range":f"bytes={low}-{high}"},timeout=(15,70))
            response.raise_for_status()
            assert response.headers.get("Content-Range")==f"bytes {low}-{high}/{SIZE}"
            assert len(response.content)==expected
            temp=name.with_suffix(".partial")
            temp.write_bytes(response.content)
            temp.replace(name)
            return name
        except (requests.RequestException,AssertionError):
            if attempt==4:
                raise
            time.sleep(attempt+1)


def shapes_and_storages(prefix):
    name_size,extra_size=struct.unpack_from("<HH",prefix,26)
    assert prefix[30:30+name_size]==b"archive/data.pkl"
    payload=prefix[30+name_size+extra_size:]
    ops=list(pickletools.genops(payload))
    end=next(i+1 for op,_,i in ops if op.name=="SETITEMS")
    # This prefix ends at state_dict's SETITEMS, before any saved Python object.
    allowed={"collections OrderedDict","torch._utils _rebuild_tensor_v2","torch FloatStorage"}
    assert {v for o,v,i in ops if i<end and o.name=="GLOBAL"}<=allowed
    key=None
    storages={}
    for op,arg,i in ops:
        if i>=end:break
        if op.name=="BINUNICODE" and isinstance(arg,str):
            if arg.startswith(("denoiser.","ema_denoiser.")):
                key=arg
            elif key and arg.isdecimal():
                storages[key]=arg
                key=None
    dummy=io.BytesIO()
    with zipfile.ZipFile(dummy,"w") as z:
        z.writestr("archive/data.pkl",payload[:end]+b".")
        z.writestr("archive/version","3\n")
        z.writestr("archive/byteorder","little")
        for storage_id in storages.values():
            z.writestr("archive/data/"+storage_id,b"")
    dummy.seek(0)
    with FakeTensorMode():
        shapes=torch.load(dummy,weights_only=True,map_location="cpu")
    assert set(shapes)==set(storages) and len(set(storages.values()))==len(shapes)
    return {k.removeprefix("ema_denoiser."):(list(t.shape),storages[k])
            for k,t in shapes.items() if k.startswith("ema_denoiser.")
            and t.dtype==torch.float32 and t.is_contiguous() and t.storage_offset()==0}


def central_directory(tail):
    eocd=tail.rfind(b"PK\x05\x06")
    entries=struct.unpack_from("<4s4H2LH",tail,eocd)[4]
    pos=tail.find(b"PK\x01\x02")
    members={}
    for _ in range(entries):
        header=struct.unpack_from("<4s6H3I5H2I",tail,pos)
        assert header[0]==b"PK\x01\x02" and header[4]==0
        crc,size,n,extra,comment,offset=header[7],header[9],header[10],header[11],header[12],header[16]
        name=tail[pos+46:pos+46+n].decode()
        members[name]=(offset,size,crc)
        pos+=46+n+extra+comment
    return members


def main():
    CACHE.mkdir(exist_ok=True)
    target=CACHE/"ema.safetensors"
    prefix=fetch((0,1048575)).read_bytes()
    tail=fetch((SIZE-200000,SIZE-1)).read_bytes()
    tensors=shapes_and_storages(prefix)
    assert len(tensors)==141
    members=central_directory(tail)
    selected=[members["archive/data/"+storage] for _,storage in tensors.values()]
    first=min(off for off,sz,crc in selected)
    last=max(off+sz+512 for off,sz,crc in selected)
    ranges=[(i,min(i+BLOCK-1,last-1)) for i in range(first,last,BLOCK)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(fetch,ranges),1):
            if i%10==0:print(i,"of",len(ranges),"blocks available",flush=True)
    def read_at(start,count):
        parts=[]
        while count:
            ix=(start-first)//BLOCK
            a,b=ranges[ix]
            chunk_size=min(count,b-start+1)
            with (CACHE/f"{a}-{b}.bin").open("rb") as f:
                f.seek(start-a)
                parts.append(f.read(chunk_size))
            start+=chunk_size
            count-=chunk_size
        return b"".join(parts)
    manifest={}
    position=0
    for key,(shape,storage) in tensors.items():
        size=members["archive/data/"+storage][1]
        assert torch.Size(shape).numel()*4==size
        manifest[key]={"dtype":"F32","shape":shape,"data_offsets":[position,position+size]}
        position+=size
    head=json.dumps(manifest,separators=(",",":")).encode()
    head+=b" "*((-len(head))%8)
    temp=target.with_suffix(".tmp")
    with temp.open("wb") as out:
        out.write(struct.pack("<Q",len(head)));out.write(head)
        for key,(_,storage) in tensors.items():
            name="archive/data/"+storage
            start,size,crc=members[name]
            loc=read_at(start,30)
            assert loc[:4]==b"PK\x03\x04"
            n,e=struct.unpack_from("<HH",loc,26)
            assert read_at(start+30,n).decode()==name
            raw=read_at(start+30+n+e,size)
            assert zlib.crc32(raw)==crc
            out.write(raw)
    temp.replace(target)
    print("Verified",len(manifest),"EMA tensor CRCs and",position,"bytes",flush=True)


if __name__=="__main__":
    main()
