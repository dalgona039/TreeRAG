"""Unit tests for the human-evaluation framework (PHASE 4). Offline, no network."""
import csv
import json

import pytest

from benchmarks.human_eval.annotation_schema import (
    ANNOTATION_DIMENSIONS,
    DIMENSION_NAMES,
    is_valid_score,
)
from benchmarks.human_eval.compute_agreement import (
    compute_system_scores,
    krippendorff_alpha,
)
from benchmarks.human_eval.generate_annotation_tasks import generate_annotation_tasks


def test_schema_dimensions_present():
    assert set(DIMENSION_NAMES) == {"faithfulness", "relevance", "citation_quality"}
    assert ANNOTATION_DIMENSIONS["citation_quality"]["scale"] == [0, 1, 2]


def test_is_valid_score():
    assert is_valid_score("faithfulness", 5)
    assert not is_valid_score("faithfulness", 6)
    assert is_valid_score("citation_quality", 0)
    assert not is_valid_score("relevance", "x")


def test_krippendorff_perfect_agreement():
    ann = {"a": {"t1": 5, "t2": 1}, "b": {"t1": 5, "t2": 1}}
    assert krippendorff_alpha(ann) == pytest.approx(1.0)


def test_krippendorff_disagreement_below_perfect():
    ann = {"a": {"t1": 5, "t2": 1}, "b": {"t1": 1, "t2": 5}}
    assert krippendorff_alpha(ann) < 0.5


def test_krippendorff_ignores_single_rating_units():
    import math

    ann = {"a": {"t1": 3}}  # only one rating -> not pairable
    assert math.isnan(krippendorff_alpha(ann))


def _tiny_report_and_dataset(tmp_path):
    report = {
        "systems": ["raptor", "treerag_beam"],
        "per_question": {
            "raptor": [{"question_id": "q{0}".format(i), "answer": "r ans {0}".format(i)} for i in range(6)],
            "treerag_beam": [{"question_id": "q{0}".format(i), "answer": "t ans [doc, p.1]"} for i in range(6)],
        },
    }
    dataset = {
        "questions": [
            {
                "question_id": "q{0}".format(i),
                "question": "Q{0}?".format(i),
                "expected_answer_hint": "hint {0}".format(i),
                "difficulty": ["easy", "medium", "hard"][i % 3],
                "category": ["factual", "multi_hop"][i % 2],
            }
            for i in range(6)
        ]
    }
    rp = tmp_path / "report.json"
    ds = tmp_path / "dataset.json"
    rp.write_text(json.dumps(report), encoding="utf-8")
    ds.write_text(json.dumps(dataset), encoding="utf-8")
    return rp, ds


def test_generate_tasks_blinded_and_keyed(tmp_path):
    rp, ds = _tiny_report_and_dataset(tmp_path)
    out_csv = tmp_path / "tasks.csv"
    summary = generate_annotation_tasks(
        str(rp), n_questions=4, systems=["raptor", "treerag_beam"],
        output_path=str(out_csv), dataset_path=str(ds),
    )
    assert summary["total_rows"] == summary["n_questions"] * 2
    with open(out_csv, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert "system" not in header  # blinded
    key = json.loads((tmp_path / "annotation_key.json").read_text(encoding="utf-8"))
    assert len(key) == summary["total_rows"]
    assert set(key.values()) <= {"raptor", "treerag_beam"}


def test_compute_system_scores(tmp_path):
    rp, ds = _tiny_report_and_dataset(tmp_path)
    out_csv = tmp_path / "tasks.csv"
    generate_annotation_tasks(
        str(rp), n_questions=4, systems=["raptor", "treerag_beam"],
        output_path=str(out_csv), dataset_path=str(ds),
    )
    key_path = tmp_path / "annotation_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    # Fill annotations: treerag scores high, raptor low.
    rows = []
    for task_id, system in key.items():
        score = 5 if system == "treerag_beam" else 2
        rows.append({"task_id": task_id, "annotator_id": "a", "faithfulness": score,
                     "relevance": score, "citation_quality": 2 if system == "treerag_beam" else 0})
    filled = tmp_path / "filled.csv"
    with open(filled, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task_id", "annotator_id", "faithfulness", "relevance", "citation_quality"])
        w.writeheader()
        w.writerows(rows)
    result = compute_system_scores(str(filled), str(key_path))
    assert result["mean_scores"]["treerag_beam"]["faithfulness"] == pytest.approx(5.0)
    assert result["mean_scores"]["raptor"]["faithfulness"] == pytest.approx(2.0)
