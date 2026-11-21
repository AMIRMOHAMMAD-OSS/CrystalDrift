"""Read the public Lightning checkpoint with inert metadata and tensor-only output."""
import argparse,builtins,collections,hashlib,io,json,pickle,struct,typing
from pathlib import Path
from zipfile import ZipFile
import numpy as np
SHA="b08366eb04d47698da7ea09dbccb1c4a1304a4c3f506fdc1d8c294910ade6dc2"
class Metadata: pass
def tensor(storage,offset,size,stride,*metadata):return (storage,offset,size,stride)
class Reader(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith("omegaconf.") and name in ("DictConfig","ContainerMetadata","AnyNode","Metadata","ListConfig"):return Metadata
        return {("collections","OrderedDict"):collections.OrderedDict,("collections","defaultdict"):collections.defaultdict,
            ("torch._utils","_rebuild_tensor_v2"):tensor,("torch","FloatStorage"):np.float32,("torch","LongStorage"):np.int64,
            ("typing","Any"):typing.Any,("__builtin__","dict"):dict,("__builtin__","list"):list,("__builtin__","long"):int}[(module,name)]
    def persistent_load(self,pid):
        kind,dtype,key,loc,count=pid
        assert kind=="storage"
        return dtype,key,count
def plain(obj):
    if isinstance(obj,Metadata):return plain(obj._content if hasattr(obj,"_content") else obj._val)
    if isinstance(obj,dict):return {k:plain(v) for k,v in obj.items()}
    if isinstance(obj,list):return list(map(plain,obj))
    return obj
def run(path,out,expected=SHA):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):digest.update(chunk)
    if digest.hexdigest()!=expected:raise ValueError("Unexpected CrystalFlow checkpoint SHA-256")
    with path.open("rb") as stream:
        header=struct.unpack("<IHHHHHIIIHH",stream.read(30));name=stream.read(header[-2]);stream.read(header[-1])
        assert name==b"archive/data.pkl"
        obj=Reader(io.BytesIO(stream.read(1024**2))).load()
    hp=plain(obj["hyper_parameters"]);print(hp["decoder"],{k:v for k,v in hp.items() if isinstance(v,(float,int,bool))},flush=True)
    arrays={}
    with ZipFile(path) as z:
        for k,(s,offset,size,stride) in obj["state_dict"].items():
            if not k.startswith("decoder."):continue
            dtype,key,count=s
            a=np.frombuffer(z.read("archive/data/"+key),dtype=dtype)
            assert len(a)==count and offset==0 and a.size==int(np.prod(size))
            a=a.reshape(size)
            assert list(np.array(a.strides)//a.itemsize)==list(stride)
            arrays[k[len("decoder."):]]=a
    np.savez(out/"weights.npz",**arrays)
    (out/"model.json").write_text(json.dumps(hp,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path);p.add_argument("out",type=Path)
    p.add_argument("--joint-manifest",type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    expected=SHA
    if a.joint_manifest:
        record=json.loads(a.joint_manifest.read_text())
        assert record["member"]=="ckpt-v1.0.0/DNG-mp-20/epoch=2694-step=72765.ckpt" and record["crc32"]=="f9e57f09"
        assert record["sha256"]=="abd359240252e6b2fe1740f006b6d80ee6df2f4b11a60f7030f4c9a75f617780"
        expected=record["sha256"]
    run(a.path,a.out,expected)
