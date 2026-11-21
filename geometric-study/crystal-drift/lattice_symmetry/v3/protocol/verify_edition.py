"""Read and reconstruct the complete sealed backup stream and record the verification."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from backup import digest,restore

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",type=Path);p.add_argument("receipt",type=Path);a=p.parse_args()
    buff=io.StringIO()
    with redirect_stdout(buff):restore(a.stage,None,None,True)
    output=buff.getvalue().strip()
    count=int(output.rsplit(" ",1)[1])
    with (a.stage/"FILES.ndjson").open(encoding="utf8") as fp:
        members=sum(not json.loads(line)["path"].endswith("/") for line in fp)
    assert count==members
    info=json.loads((a.stage/"BACKUP.json").read_text())
    a.receipt.write_text(json.dumps({"verified":True,"reconstructed_files_and_links":count,"logical_bytes":info["logical_bytes"],
        "checksum":digest(a.stage/"BACKUP.json"),"stored_objects":len(info["volumes"]),"result":output},indent=2)+"\n")
    print(output,flush=True)
