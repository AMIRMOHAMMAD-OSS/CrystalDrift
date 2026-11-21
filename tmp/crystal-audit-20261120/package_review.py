from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from pypdf import PdfReader, PdfWriter
import hashlib
from io import BytesIO

work = Path(__file__).resolve().parent
delivery = work.parents[1] / "output" / "crystal-review"
delivery.mkdir(parents=True, exist_ok=True)
figdir = work / "figures"
stems = ["01_motif_and_fourier_features","02_recovery_and_time","03_recovery_definitions"]
names = [s+"."+suffix for s in stems for suffix in ("pdf","svg")]
names.extend(["README.md","make_figures.py","binary_seed_prefixes.csv","native_recovery.csv"])
archive = delivery / "CrystalDrift_Editable_Figures.zip"
with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
    for name in names:
        z.write(figdir/name, name)
with ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist()) == set(names)

source = work / "CrystalDrift_Manuscript_Review.pdf"
report = delivery / source.name
reader = PdfReader(source)
writer = PdfWriter(clone_from=reader)
writer.add_attachment(archive.name, archive.read_bytes())
with report.open("wb") as handle:
    writer.write(handle)
result = PdfReader(report)
assert len(result.pages)==12
assert [p.extract_text() for p in reader.pages]==[p.extract_text() for p in result.pages]
assert all(p.extract_text().strip() for p in result.pages)
assert [i.title for i in result.outline]==[i.title for i in reader.outline]
assert result.attachments[archive.name][0]==archive.read_bytes()
print(report.name,report.stat().st_size,"bytes",len(result.pages),"pages")
print(archive.name,archive.stat().st_size,"bytes",len(names),"files",hashlib.sha256(archive.read_bytes()).hexdigest())
print("Pages with figure XObjects",[(i+1,len(p.get('/Resources').get_object().get('/XObject',{}))) for i,p in enumerate(result.pages) if p.get('/Resources').get_object().get('/XObject',{})])

