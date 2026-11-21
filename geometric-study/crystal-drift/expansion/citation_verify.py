"""Reconcile manuscript citation keys and public depositor metadata."""

import concurrent.futures
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DEST = ROOT / "expansion/citation-metadata.json"


def normalize(s):
    text = unicodedata.normalize("NFKD", s).casefold()
    return "".join(x for x in text if x.isalnum())


def fetch(entry):
    key, url, title = entry
    primary = url
    if "doi.org/" in url:
        doi = url.split("doi.org/", 1)[1]
        primary = "https://api.crossref.org/works/" + doi
        r = requests.get(primary, timeout=45)
        r.raise_for_status()
        record = r.json()["message"]
        registered = BeautifulSoup(record["title"][0], "html.parser").get_text()
        authors = [a.get("given", "") + " " + a.get("family", "") for a in record["author"]]
        date = record.get("published", record.get("issued"))["date-parts"][0]
    else:
        if "arxiv.org/html/" in url:
            primary = url.replace("/html/", "/abs/")
        r = requests.get(primary, timeout=45)
        r.raise_for_status()
        h = BeautifulSoup(r.content, "html.parser")

        def tag(name):
            return [
                e["content"].strip()
                for e in h.select(f'meta[name="{name}"], meta[property="{name}"]')
                if e.get("content")
            ]

        registered = (
            tag("citation_title") or tag("dc.Title") or tag("og:title") or [h.title.get_text()]
        )[0]
        authors = tag("citation_author") or tag("DC.Creator")
        date = (tag("citation_publication_date") or tag("citation_date") or tag("dc.Date") or [""])[
            0
        ]
    return key, dict(
        manuscript_title=title,
        official_title=registered,
        title_agrees=normalize(title) == normalize(registered),
        official_authors=authors,
        publication_date=date,
        manuscript_url=url,
        metadata_source=primary,
    )


def main():
    text = (PAPER / "bibliography.tex").read_text(encoding="utf8")
    refs = re.findall(r"\\bibitem\{([^}]+)\}[^\n]*?\\linkto\{([^}]+)\}\{([^}]+)\}", text)

    def visit(name, seen):
        file = PAPER / (name + ".tex")
        if file in seen:
            return ""
        seen.add(file)
        src = file.read_text(encoding="utf8")
        return (
            src
            + "\n"
            + "\n".join(
                visit(other, seen)
                for other in re.findall(r"\\input\{([^}]+)\}", src)
                if (PAPER / (other + ".tex")).exists()
            )
        )

    manuscript = visit("main", set())
    uses = set(
        k.strip()
        for block in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for k in block.split(",")
    )
    keys = {k for k, _, _ in refs}
    assert len(refs) == len(keys) and not uses - keys
    existing = json.loads(DEST.read_text(encoding="utf8"))["entries"] if DEST.exists() else {}
    missing = [
        row
        for row in refs
        if row[0] not in existing or existing[row[0]]["manuscript_url"] != row[1]
    ]
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        tasks = {pool.submit(fetch, row): row[0] for row in missing}
        for future in concurrent.futures.as_completed(tasks):
            try:
                k, details = future.result()
                existing[k] = details
            except Exception as e:
                failures[tasks[future]] = str(e)
    result = dict(
        cited_keys=sorted(uses),
        unused_bibliography_keys=sorted(keys - uses),
        failed_requests=failures,
        entries={key: existing[key] for key, _, _ in refs if key in existing},
    )
    DEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(
        "links",
        len(refs),
        "metadata",
        len(existing),
        "mismatch",
        [k for k, r in existing.items() if not r["title_agrees"]],
        "failed",
        failures,
    )


if __name__ == "__main__":
    main()
