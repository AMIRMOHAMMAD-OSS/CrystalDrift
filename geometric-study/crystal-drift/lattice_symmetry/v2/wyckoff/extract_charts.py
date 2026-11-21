"""Build exact row-affine conventional Wyckoff charts from the PyXtal 1.0.9 wheel."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import string
import zipfile

import numpy as np
from pymatgen.core.operations import SymmOp

HERE = Path(__file__).resolve().parent

def main(wheel):
    with zipfile.ZipFile(wheel) as z:
        source = ast.parse(z.read("pyxtal/symmetry.py").decode())
        halls = next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=="pyxtal_hall_numbers" for t in n.targets))
        payload = z.read("pyxtal/database/wyckoff_list.csv")
        records = list(csv.reader(io.StringIO(payload.decode())))
        (HERE/"PyXtal-LICENSE.txt").write_bytes(z.read("pyxtal-1.0.9.dist-info/licenses/LICENSE.txt"))
    matrices,offsets,positions,groups = [],[],[0],[0]
    for number,text in records[2:]:
        assert int(number)==len(groups)
        for orbit in reversed(ast.literal_eval(text)):
            for eq in orbit:
                op = SymmOp.from_xyz_str(eq)
                M = np.rint(op.rotation_matrix)
                t = np.rint(24*op.translation_vector)
                np.testing.assert_allclose(op.rotation_matrix,M)
                np.testing.assert_allclose(op.translation_vector,t/24)
                matrices.append(M.T);offsets.append(t)
            positions.append(len(matrices))
        groups.append(len(positions)-1)
    np.savez_compressed(HERE/"charts.npz",matrices=np.array(matrices,dtype='i1'),
                        t24=np.array(offsets,dtype='i1'),positions=positions,groups=groups,halls=halls)
    (HERE/"chart-source.json").write_text(json.dumps(dict(version="1.0.9",url="https://pypi.org/project/pyxtal/1.0.9/",
                wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                csv_sha256=hashlib.sha256(payload).hexdigest(),groups=len(groups)-1,positions=len(positions)-1,
                coset_images=len(matrices)),indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("wheel",type=Path);main(p.parse_args().wheel)
