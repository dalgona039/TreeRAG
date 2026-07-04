#!/usr/bin/env python3
"""
Regenerate the medical-domain figure with PageTree-RAG labels and swap it into
TreeRAG_TIST_ACM.docx (Figure 5). Run once from the project root:

    python scripts/fix_medical_figure.py

- Reads data/benchmark_reports/medical_results.json (summary.{sys}.rouge_l /
  .medical_entity_recall) so the figure always matches the paper's Table 4.
- Writes data/benchmark_reports/figures/figure_5_medical.{png,pdf}.
- Replaces the image embedded above the "Figure 5:" caption in the .docx,
  preserving width and adjusting height to the new aspect ratio.

Idempotent and self-contained (needs: matplotlib, python-docx).
"""
import json, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "benchmark_reports" / "medical_results.json"
FIGDIR = ROOT / "data" / "benchmark_reports" / "figures"
DOCX = ROOT / "TreeRAG_TIST_ACM.docx"

LABELS = {
    "bm25": "BM25", "dense": "Dense", "flatrag": "FlatRAG", "raptor": "RAPTOR",
    "treerag_dfs": "PageTree-RAG (DFS)", "treerag_beam": "PageTree-RAG (Beam)",
}
ORDER = ["bm25", "dense", "flatrag", "raptor", "treerag_dfs", "treerag_beam"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#CC79C7", "#64B5CD"]

# Fallback matches paper Table 4 (used only if the JSON can't be parsed).
FALLBACK = {
    "bm25": (0.358, 1.000), "dense": (0.315, 0.992), "flatrag": (0.271, 1.000),
    "raptor": (0.053, 0.895), "treerag_dfs": (0.366, 1.000), "treerag_beam": (0.265, 1.000),
}


def load_medical():
    try:
        s = json.load(open(REPORT, encoding="utf-8"))["summary"]
        return {k: (float(s[k]["rouge_l"]),
                    float(s[k].get("medical_entity_recall", s[k].get("entity_recall", 1.0))))
                for k in ORDER}
    except Exception as e:
        print(f"  ! could not read {REPORT} ({e}); using Table-4 fallback values")
        return FALLBACK


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = [LABELS[k] for k in ORDER]
    rouge = [data[k][0] for k in ORDER]
    recall = [data[k][1] for k in ORDER]
    best = rouge.index(max(rouge))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(ORDER))
    ax.bar(x, rouge, color=COLORS, edgecolor="black", linewidth=0.5, width=0.62)
    ax.set_ylabel("ROUGE-L"); ax.set_ylim(0, 0.55)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.annotate("★", (best, rouge[best] + 0.015), ha="center", va="bottom",
                fontsize=18, color="gold")
    ax2 = ax.twinx()
    ax2.plot(x, recall, "k--o", linewidth=1.8, markersize=8)
    ax2.set_ylabel("Medical Entity Recall"); ax2.set_ylim(0.75, 1.07)
    ax.set_title("Medical Domain Results (n=42)\n★ = best ROUGE-L")
    ax.legend(handles=[Line2D([0], [0], color="#4C72B0", lw=8, label="ROUGE-L (left axis)"),
                       Line2D([0], [0], color="k", ls="--", marker="o",
                              label="Entity Recall (right axis)")],
              loc="upper right", framealpha=0.9)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGDIR / "figure_5_medical.png"
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"figure_5_medical.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ wrote {out_png}")
    return out_png


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


def swap_into_docx(png):
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(str(DOCX))

    # find the image rId that precedes the "Figure 5:" caption
    target = None
    last = None
    for p in doc.paragraphs:
        blips = p._p.findall(".//" + qn("a:blip"))
        if blips:
            last = blips[-1].get(qn("r:embed"))
        if p.style.name == "FigureCaption" and p.text.strip().startswith("Figure 5:"):
            target = last
            break
    if target is None:
        raise SystemExit("Could not locate the image above the 'Figure 5:' caption.")

    w, h = png_size(png)
    for shape in doc.inline_shapes:
        blip = shape._inline.graphic.graphicData.pic.blipFill.blip
        if blip.get(qn("r:embed")) == target:
            inline = shape._inline
            cx = int(inline.extent.get("cx"))
            cy = int(round(cx * h / w))
            inline.extent.set("cy", str(cy))
            for ext in inline.findall(".//" + qn("a:ext")):
                ext.set("cx", str(cx)); ext.set("cy", str(cy))
            break
    with open(png, "rb") as f:
        doc.part.related_parts[target]._blob = f.read()
    doc.save(str(DOCX))
    print(f"  ✓ replaced Figure 5 image ({target}) in {DOCX.name}")


if __name__ == "__main__":
    print("Regenerating medical figure with PageTree-RAG labels ...")
    png = make_figure(load_medical())
    print("Swapping into docx (make sure Word has the file CLOSED) ...")
    swap_into_docx(png)
    print("Done.")
