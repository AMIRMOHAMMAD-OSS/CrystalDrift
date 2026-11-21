"""Locate ordered candidate shards in the release or its optional research scratch volume."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def pool(panel, tag, write=False):
    local = HERE / "raw" / panel / tag
    alternative = os.environ.get("CRYTALDRIFT_RAW")
    if alternative:
        portable = Path(alternative) / panel / tag
        if portable.exists():
            return portable
        if write and not local.exists():
            portable.mkdir(parents=True, exist_ok=True)
            pointer = HERE / "raw-locations" / panel / (tag + ".json")
            pointer.parent.mkdir(exist_ok=True, parents=True)
            pointer.write_text(json.dumps({"relative": f"evaluation/raw/{panel}/{tag}", "actual": str(portable)}, indent=2))
            return portable
    return local
