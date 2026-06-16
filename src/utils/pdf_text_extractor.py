"""
PDF plain-text extraction (PHASE 1-2 of the ACM upgrade plan).

Extracts full plain text from a PDF using ``pypdf`` (already a project
dependency), inserting page markers so downstream consumers (e.g. the RAPTOR
baseline) can recover page references.

The module is import-safe even when pypdf is unavailable: extraction then
returns an empty string rather than raising at import time.
"""
from __future__ import annotations

import os
from typing import List

PAGE_MARKER = "--- PAGE {n} ---"


def extract_text(pdf_path: str) -> str:
    """Return concatenated plain text of all pages with page markers.

    Format::

        --- PAGE 1 ---
        {page 1 text}
        --- PAGE 2 ---
        {page 2 text}
        ...

    On any failure (missing file, unreadable PDF, pypdf absent) returns "".
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        parts: List[str] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            parts.append(PAGE_MARKER.format(n=i))
            parts.append(text.strip())
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_directory(raw_dir: str, out_dir: str) -> List[str]:
    """Extract every PDF in ``raw_dir`` to ``out_dir/{stem}.txt``.

    Returns the list of written .txt paths. Skips non-PDF files. Deterministic
    ordering (sorted) for reproducibility.
    """
    written: List[str] = []
    if not os.path.isdir(raw_dir):
        return written
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(raw_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(raw_dir, name)
        text = extract_text(pdf_path)
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(out_dir, stem + ".txt")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            written.append(out_path)
        except Exception:
            continue
    return written


def page_count(text: str) -> int:
    """Count page markers in extracted text (cheap page estimate)."""
    return text.count("--- PAGE ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract plain text from PDFs (PHASE 1-2)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/raw_text")
    args = parser.parse_args()

    paths = extract_directory(args.raw_dir, args.out_dir)
    print("Extracted {0} PDFs -> {1}".format(len(paths), args.out_dir))
    for p in paths:
        print("  " + os.path.basename(p))
