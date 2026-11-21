"""Fetch only the unconditional MP-20 member of the authors' public release ZIP."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
from urllib.parse import urlsplit
import zlib

URL = "https://github.com/ixsluo/CrystalFlow/releases/download/v1.0.0-alpha.1/ckpt-v1.0.0.zip"
NAME = "ckpt-v1.0.0/DNG-mp-20/epoch=2694-step=72765.ckpt"
START, STOP = 231460608, 440676603
COMPRESSED, CRC = 209215807, 0xf9e57f09
SHA256 = "abd359240252e6b2fe1740f006b6d80ee6df2f4b11a60f7030f4c9a75f617780"


def run(dest, workers):
    dest.mkdir(parents=True, exist_ok=True)
    pieces = dest / "joint-download"
    pieces.mkdir(exist_ok=True)
    chunk = 4 * 1024 ** 2
    ranges = [(lo, min(lo + chunk - 1, STOP)) for lo in range(START, STOP + 1, chunk)]

    def get(bounds):
        lo, hi = bounds
        path = pieces / f"{lo}-{hi}.bin"
        length = hi - lo + 1
        failures = 0
        suffix = path.with_suffix(".part")
        head = path.with_suffix(".headers")
        while True:
            have = path.stat().st_size if path.exists() else 0
            if have == length:break
            if not 0<=have<length:raise ValueError(path)
            start = lo + have
            stop = min(hi,start + 524287)
            suffix.unlink(missing_ok=True)
            remote=URL+f"?memberpart={start}&attempt={failures}"
            if failures%2==1:
                for headers in pieces.glob("*.headers"):
                    locations=[l.partition(": ")[2].strip() for l in headers.read_text(errors="ignore").splitlines()
                               if l.lower().startswith("location:")]
                    if locations and urlsplit(locations[0]).hostname=="release-assets.githubusercontent.com":
                        remote=locations[0]
                        break
            completed = subprocess.run(["curl", "-fsSL", "--connect-timeout", "30", "--max-time", "120",
                            "--max-filesize",str(stop-start+1),"--range",f"{start}-{stop}",
                            "-D",str(head),remote,"-o",str(suffix)],
                            capture_output=True)
            expected = f"content-range: bytes {start}-{stop}/882862517"
            fragment = suffix.read_bytes() if suffix.exists() else b""
            if fragment and expected in head.read_text(errors="replace").lower() and len(fragment)<=stop-start+1:
                with path.open("ab") as f:f.write(fragment)
                failures=0
            else:
                failures+=1
                if failures>=15:raise RuntimeError(f"Range {start}-{stop} unavailable: {completed.stderr[-280:]!r}")
                time.sleep(min(failures,8))
        assert path.stat().st_size == length
        return path

    with ThreadPoolExecutor(workers) as pool:
        paths = list(pool.map(get, ranges))
    sha = hashlib.sha256()
    crc = size = 0
    inflator = zlib.decompressobj(-15)
    left = COMPRESSED
    target = dest / "joint.ckpt"
    with target.open("wb") as out:
        for n, path in enumerate(paths):
            raw = path.read_bytes()
            if n == 0:
                header = struct.unpack_from("<IHHHHHIIIHH", raw)
                assert header[0] == 0x04034b50 and header[3] == 8 and header[6] == CRC
                assert raw[30:30 + header[-2]].decode() == NAME
                raw = raw[30 + header[-2] + header[-1]:]
            encoded = raw[:left]
            left -= len(encoded)
            block = inflator.decompress(encoded)
            out.write(block)
            sha.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    assert left == 0 and inflator.eof and crc == CRC and sha.hexdigest() == SHA256
    summary = dict(url=URL, member=NAME, archive_bytes=882862517, local_header=START,
                   compressed_bytes=COMPRESSED, bytes=size, crc32=f"{crc:08x}",
                   sha256=sha.hexdigest())
    (dest / "joint-source.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", type=Path)
    p.add_argument("--workers", type=int, default=10)
    a=p.parse_args()
    run(a.dest, a.workers)
