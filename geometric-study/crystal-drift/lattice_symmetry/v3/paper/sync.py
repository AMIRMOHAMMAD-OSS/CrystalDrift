"""Copy authoritative mathematical sections and the exact editable figure outputs to the paper."""
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
V3=HERE.parent
def sync():
    sets={
        "geometry":["supplement.tex","occupancy-supplement.tex","conditioned-supplement.tex","crystal-gallery-extended.tex",
                    "template-coverage.pdf","fourier-cutoff.pdf","crystal-gallery-extended.pdf"],
        "denovo/catalog":["spacegroups.pdf"],
        "uflow":["native-math.tex","uflow-learning.pdf"],
        "models":["learning-curves.pdf"]}
    for group,names in sets.items():
        for name in names:
            src=V3/group/name
            if not src.exists():raise FileNotFoundError(src)
            shutil.copyfile(src,HERE/name)
            if name.endswith(".pdf") and src.with_suffix(".svg").exists():
                shutil.copyfile(src.with_suffix(".svg"),HERE/(src.stem+".svg"))
    from write import literature
    literature()
if __name__=="__main__":sync()
