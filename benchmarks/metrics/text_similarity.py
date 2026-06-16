"""
Text-similarity metrics (PHASE C-1 of the KCI publication plan).

Provides three scorers, each returning a float in [0.0, 1.0]:

* :func:`rouge_l_score`  – ROUGE-L F1 (``rouge-score`` library).
* :func:`bertscore_f1`   – BERTScore F1 (``bert-score`` library). Falls back to a
  deterministic token-F1 proxy when the model cannot be loaded (e.g. offline /
  no network), so the pipeline stays runnable; swap in the real model locally
  for publication numbers.
* :func:`exact_match`    – 1.0 iff the normalised strings are identical.

Plus :func:`batch_evaluate` which runs the requested metrics concurrently.
"""
from __future__ import annotations

import re
import string
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Dict, List, Optional

_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation + "·、，。！？〈〉《》「」『』"))
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+|[一-鿿]")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


# --------------------------------------------------------------------------- #
# ROUGE-L
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _rouge_scorer():
    from rouge_score import rouge_scorer
    from rouge_score.tokenizers import Tokenizer

    class _MultilingualTokenizer(Tokenizer):
        # rouge-score's default tokenizer strips non-ASCII, which zeroes out
        # Korean/CJK text. Use a Unicode-aware tokenizer instead.
        def tokenize(self, text):
            return _tokens(text)

    return rouge_scorer.RougeScorer(["rougeL"], tokenizer=_MultilingualTokenizer())


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 using the rouge-score library."""
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    try:
        result = _rouge_scorer().score(reference, hypothesis)
        return float(result["rougeL"].fmeasure)
    except Exception:
        return _lcs_f1(_tokens(hypothesis), _tokens(reference))


def _lcs_f1(hyp: List[str], ref: List[str]) -> float:
    if not hyp or not ref:
        return 0.0
    n, m = len(hyp), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if hyp[i - 1] == ref[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# BERTScore
# --------------------------------------------------------------------------- #
def _token_f1(hypothesis: str, reference: str) -> float:
    """Multiset token-F1 — the offline proxy for BERTScore."""
    hyp, ref = _tokens(hypothesis), _tokens(reference)
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    from collections import Counter

    common = Counter(hyp) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(hyp), overlap / len(ref)
    return 2 * prec * rec / (prec + rec)


def bertscore_f1(hypothesis: str, reference: str, lang: str = "ko") -> float:
    """Compute BERTScore F1 using the bert-score library.

    Model: ``klue/roberta-base`` for Korean, ``roberta-base`` for English.
    Falls back to a token-F1 proxy if the model/library is unavailable.
    """
    if not hypothesis and not reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0
    try:
        from bert_score import score as _bert_score

        model = "klue/roberta-base" if lang == "ko" else "roberta-base"
        _, _, f1 = _bert_score(
            [hypothesis], [reference], lang=lang, model_type=model, verbose=False
        )
        return float(f1.mean().item())
    except Exception:
        return _token_f1(hypothesis, reference)


# --------------------------------------------------------------------------- #
# Exact match
# --------------------------------------------------------------------------- #
def exact_match(hypothesis: str, reference: str) -> float:
    """Return 1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(hypothesis) == _normalize(reference) else 0.0


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
_METRIC_FUNCS: Dict[str, Callable[[str, str], float]] = {
    "rouge_l": rouge_l_score,
    "bertscore": bertscore_f1,
    "exact_match": exact_match,
}


# --------------------------------------------------------------------------- #
# PHASE 3-3: medical entity recall
# --------------------------------------------------------------------------- #
MEDICAL_TERMS = [
    # Anatomy / physiology
    "cardiac", "pulmonary", "hepatic", "renal", "cerebral", "vascular",
    "myocardial", "neural", "skeletal", "endocrine", "cardiovascular",
    "musculoskeletal", "gait", "stroke volume", "heart rate",
    # Common conditions
    "hypertension", "diabetes", "myocardial infarction", "stroke", "sepsis",
    "pneumonia", "arrhythmia", "fibrillation", "thrombosis", "embolism",
    # Measurements / units
    "mmhg", "bpm", "ml/min", "mg/dl", "mcg/kg", "iu/l", "khz", "mhz", "hz",
    "decibel", "frequency", "wavelength", "amplitude",
    # Imaging / biomedical engineering
    "ultrasound", "doppler", "transducer", "piezoelectric", "impedance",
    "electrode", "biosignal", "prosthetic", "scaffold", "biocompatibility",
    "in vitro", "in vivo", "signal-to-noise", "attenuation", "reflection",
    "acoustic", "echo", "b-mode", "sonar", "imaging", "diagnosis",
    "resolution", "penetration", "probe", "gel", "tissue",
    # Korean biomedical terms
    "초음파", "압전", "감쇠", "임피던스", "주파수", "파장", "도플러",
    "심장", "혈류", "진단", "영상", "전극", "생체", "조직", "반사",
    "보행", "근골격계", "심박", "프로브", "탐촉자", "음향", "해상도",
]


def medical_entity_recall(hypothesis: str, reference: str, terms: Optional[List[str]] = None) -> float:
    """Recall of medical entities present in the reference, found in hypothesis.

    Deterministic keyword approach over :data:`MEDICAL_TERMS` (no external APIs).
    Returns count_matched / count_in_reference, or 1.0 if the reference contains
    no medical terms (vacuously complete).
    """
    vocab = [t.lower() for t in (terms if terms is not None else MEDICAL_TERMS)]
    hyp = (hypothesis or "").lower()
    ref = (reference or "").lower()
    in_ref = [t for t in vocab if t in ref]
    if not in_ref:
        return 1.0
    matched = sum(1 for t in in_ref if t in hyp)
    return matched / len(in_ref)


# Registered after definition (the dict is declared earlier in the module).
_METRIC_FUNCS["medical_entity_recall"] = medical_entity_recall


def batch_evaluate(
    hypotheses: List[str],
    references: List[str],
    metrics: List[str],
) -> Dict[str, List[float]]:
    """Run all requested metrics over paired hyp/ref lists, concurrently.

    Returns a dict mapping each metric name to a list of per-pair scores.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    unknown = [m for m in metrics if m not in _METRIC_FUNCS]
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}")

    results: Dict[str, List[float]] = {}

    def run_metric(name: str) -> List[float]:
        fn = _METRIC_FUNCS[name]
        return [fn(h, r) for h, r in zip(hypotheses, references)]

    with ThreadPoolExecutor(max_workers=max(1, len(metrics))) as pool:
        futures = {name: pool.submit(run_metric, name) for name in metrics}
        for name, fut in futures.items():
            results[name] = fut.result()
    return results
