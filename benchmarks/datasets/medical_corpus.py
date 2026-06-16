"""
Medical corpus preparation (PHASE 3-1 of the ACM upgrade plan).

Tries to download open-access medical PDFs; on failure (offline sandbox) uses
the biomedical PDFs already present in ``data/raw/``. Returns the list of
available medical PDF paths.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
MEDICAL_DL_DIR = RAW_DIR / "medical"

OPEN_MEDICAL_SOURCES = [
    # PubMed Central Open Access (example article PDF endpoint)
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8460250/pdf/",
    # WHO guideline (public domain)
    "https://apps.who.int/iris/bitstream/handle/10665/272596/9789241565653-eng.pdf",
]

# Biomedical PDFs already in data/raw (used when downloads are blocked).
LOCAL_MEDICAL_PDFS = [
    "b87261e8_생체의공학개론#10.pdf",
    "c7b780a9_생체의공학개론#11.pdf",
    "92928ecd_생체의공학개론_보고서.pdf",
    "61dd7aa0_s41598-026-41649-2_reference.pdf",
]


def _try_download(url: str, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").split("/")[-1] or "download"
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    dest = dest_dir / name
    with urllib.request.urlopen(url, timeout=20) as resp:  # nosec - public sources
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    return str(dest)


def prepare_medical_corpus() -> List[str]:
    """Return paths to available medical PDFs (downloaded or local fallback)."""
    paths: List[str] = []

    # 1. Attempt downloads (best-effort, per-source try/except).
    for url in OPEN_MEDICAL_SOURCES:
        try:
            paths.append(_try_download(url, MEDICAL_DL_DIR))
        except Exception:
            continue

    # 2. Always include the local biomedical PDFs that exist.
    for name in LOCAL_MEDICAL_PDFS:
        p = RAW_DIR / name
        if p.exists():
            paths.append(str(p))

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


if __name__ == "__main__":
    found = prepare_medical_corpus()
    print("Medical corpus: {0} PDF(s)".format(len(found)))
    for p in found:
        print("  " + os.path.basename(p))
