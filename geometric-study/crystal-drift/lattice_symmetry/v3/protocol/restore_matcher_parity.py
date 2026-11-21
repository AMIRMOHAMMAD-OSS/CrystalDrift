"""Recover the three exact archived cross-platform matcher outputs from the published ZIPs."""
import hashlib
import io
import json
import zipfile

import numpy as np

from backup_delta import HERE, RESEARCH
from verify_durable import MOUNT


def main():
    ev=HERE.parent/"evaluation"
    desired={r["source_sha"]:r for r in json.loads((ev/"matcher-platform-parity.json").read_text())}
    win={}
    for fingerprint,r in desired.items():
        win[fingerprint]=ev/"scores/family"/r["method"]/"00000-00096.npz"
    size_low=min(p.stat().st_size for p in win.values())-450
    size_high=max(p.stat().st_size for p in win.values())+450
    recovered={}
    folder=RESEARCH/"parity"
    for index in range(263,212,-1):
        if len(recovered)==len(desired):break
        volume=f"pack-{index:05}.zip"
        with zipfile.ZipFile(MOUNT/volume) as pack:
            candidates=[v for v in pack.infolist() if size_low<=v.file_size<=size_high]
            print("Volume",index,"size candidates",len(candidates),flush=True)
            for candidate in reversed(candidates):
                data=pack.read(candidate)
                if not data.startswith(b"PK\x03\x04"):continue
                try:
                    with np.load(io.BytesIO(data),allow_pickle=False) as obj:
                        if "source_sha" not in obj.files or "rms" not in obj.files:continue
                        key=str(obj["source_sha"])
                        if key not in desired or key in recovered:continue
                        reference=desired[key]
                        with np.load(win[key]) as orig:
                            for name in ("ids","error","strict_fit"):np.testing.assert_array_equal(obj[name],orig[name])
                            a,b=obj["rms"],orig["rms"]
                            np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b))
                            delta=float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(a)])))
                            if delta!=reference["max_rms_error"] or int(np.isfinite(a).sum())!=reference["matches"]:continue
                        if list(a.shape)!=reference["shape"]:continue
                except (ValueError,zipfile.BadZipFile):
                    continue
                if hashlib.sha256(data).hexdigest()!=candidate.filename:raise ValueError("Unverified original payload")
                folder.mkdir(exist_ok=True)
                path=folder/("linux-"+reference["method"]+".npz")
                if path.exists():
                    if path.read_bytes()!=data:raise ValueError("Different existing parity output")
                else:
                    with path.open("xb") as f:f.write(data)
                recovered[key]=dict(method=reference["method"],pack=volume,sha256=candidate.filename,
                                    bytes=len(data),max_rms_error=delta,matches=reference["matches"])
                print("Restored",path.name,volume,flush=True)
    if set(recovered)!=set(desired):raise ValueError("Could not identify every original parity result")
    report=HERE/"matcher-parity-restoration.json"
    if report.exists():
        if json.loads(report.read_text())!=recovered:raise ValueError("Previous parity restoration differs")
    else:
        with report.open("x") as f:json.dump(recovered,f,indent=2)


if __name__=="__main__":
    main()
