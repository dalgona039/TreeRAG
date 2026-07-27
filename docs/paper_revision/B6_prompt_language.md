# B6 — Language-Mismatch Confound Audit

## 1. Are the traversal/node-scoring prompts hardcoded Korean?

**Yes, unconditionally, with no language parameter of any kind.**

- `src/core/tree_traversal.py::_select_most_relevant_children` (DFS branch selection,
  lines ~248-289): the entire prompt — role line, instructions, selection criteria, JSON
  schema description — is a literal Korean f-string
  (`"당신은 문서 탐색 전문가입니다.\n다음 하위 섹션들 중에서..."`, line 265-266). No
  `language` argument exists anywhere in this method's signature or call chain.
- `src/core/beam_search.py::_batch_llm_score` (beam-search node scoring, lines ~250-300+):
  same pattern — `"당신은 문서 검색 관련성 평가 전문가입니다..."`. Also no language parameter.
  This function does have a kill-switch, but it's an *entirely-skip* switch, not a
  language switch: `TREERAG_DISABLE_LLM_SCORING=1` skips LLM scoring altogether (falls
  back to keyword+structure scoring only), default is **off**, so LLM scoring — in Korean
  — runs by default (`beam_search.py:265-270`).

Both prompts fire on **every** traversal step of **every** query, regardless of what
language the query or the document tree is in.

## 2. Was the Korean traversal prompt actually used on English benchmark runs?

Checked `data/benchmark_reports/online_local_llama_govreport_v2.json` (a stored
GovReport — English — fair-protocol result): `mode: "online"`, `gen_backend: "ollama"`.
`"online"` mode means real LLM traversal (as opposed to the offline keyword-only fallback
path audited separately in B7), and there is no traversal-side language switch or
disable flag that would have been active for this run. **Conclusion: yes** — English
GovReport queries were scored against Korean-language node-selection prompts during
traversal. (The equivalent HotpotQA result file,
`exp2_multihop_hotpotqa_20260707_181812.json`, does not carry top-level `mode`/
`gen_backend` fields in the same schema as the general/govreport files, so this could not
be verified the same way for HotpotQA specifically — flagged as not fully confirmed for
that file.)

## 3. Where does §6.3's Korean-output bug (28/100) come from?

Traced the full generation path. There are two separate generation entry points:

- **Baselines (BM25/Dense/FlatRAG/RAPTOR)**: all go through the *shared*
  `_llm_generate` helper in `benchmarks/run_real_evaluation.py:157-201`. This helper
  detects the question's language via a Korean-character regex
  (`is_korean = bool(re.search(r"[가-힣]", question))`, line 169) and builds an
  **entirely** English or entirely Korean prompt accordingly (lines 170-186) — a clean,
  complete language match with no Korean leakage for English questions.
- **PageTree-RAG** (`_run_treerag`, lines 290-333): does **not** use `_llm_generate` at
  all. It calls `TreeRAGReasoner.query()` in `src/core/reasoner.py`, which builds its own
  prompt and is where the residual Korean-output bug lives:
  - `DOMAIN_PROMPTS` (`reasoner.py:18-52`) — the "system role" string prepended to every
    answer-generation prompt — is **Korean-only for all five domains** ("general",
    "medical", "legal", "financial", "academic"), with no English variant. It is inserted
    verbatim via `domain_prompt = DOMAIN_PROMPTS.get(domain_template, ...)`
    (`reasoner.py:339`) into **every** prompt variant, including the "fixed"
    `_SIMPLE_PROMPT_EN` template used for English questions with the local Ollama
    backend (`reasoner.py:78-87, 351-357`).
  - When `use_simple_prompt=False` (the default whenever `gen-backend != ollama`, i.e.
    the Gemini/complex-scaffold path — `run_real_evaluation.py:296`, default
    `--gen-backend` is `"gemini"`, `run_real_evaluation.py:646`), the entire multi-step
    procedural scaffold (`reasoner.py:361-408`: step-by-step instructions, rules, answer
    template — "### 📋 답변 작성 단계", "STEP 1: 질문 핵심 파악", etc.) is 100% Korean.
    Only one line, `{language_instruction}`, switches to English
    (`"**IMPORTANT: You MUST respond in English only.**"`, `reasoner.py:56`). Everything
    else surrounding it — including the JSON key names and citation format examples the
    model is told to imitate — is Korean.
  - A code comment at `reasoner.py:63-65` self-documents the mechanism of the residual
    bug: *"8B models drift to Korean when the surrounding markers are Korean even with an
    English language_instruction"* — this was written to justify the `_SIMPLE_PROMPT_EN`
    fix, but that fix only replaced the *skeleton* markers, not `DOMAIN_PROMPTS`, so the
    same drift mechanism the comment describes still applies to the Korean
    `domain_prompt` string injected into every prompt, simple or complex.

**Conclusion: the §6.3 bug most plausibly originates in `reasoner.py`'s `DOMAIN_PROMPTS`
(and, for the Gemini/complex-scaffold path, the entire procedural scaffold), not in the
traversal prompts audited in §1.** The traversal prompts never produce user-visible text
directly (their JSON output is `{"selected_indices": [...], "reason": "..."}`, and only
the indices are consumed — `tree_traversal.py:299-311` — the Korean `"reason"` field is
discarded), so they cannot be the direct source of Korean text appearing in a final
answer. They *could* still be an indirect contributor (e.g., biasing which nodes get
selected, or destabilizing the local 8B model's language state across the multi-call
session before the final generation call), but that is a hypothesis, not something
confirmed by this code audit — it would need the controlled experiment described below to
test.

## 4. Is this confound exclusive to PageTree-RAG?

**Yes, structurally.** Baselines make exactly one LLM call per query
(`_llm_generate`, fully language-matched). PageTree-RAG makes that same kind of
call (via `reasoner.query()`, only *partially* language-matched per §3) **plus** a
variable number of additional traversal-time LLM calls (DFS branch selection and/or beam
node scoring) that are **always** Korean, **never** language-matched, for every query
regardless of language. There is no code path by which a baseline system is exposed to
the Korean traversal prompts — they don't traverse a tree with LLM scoring at all. This
confirms the concern as stated: the language-mismatch confound applies only to the
PageTree-RAG condition, not to any baseline, in a benchmark that is majority English
(HotpotQA, GovReport).

## 5. Controlled A/B experiment (English-translated traversal prompt vs. Korean)

**Not run.** This would require: (a) writing an English-translated variant of the two
traversal prompts gated behind an env var so the existing Korean behavior is unchanged by
default, (b) running PageTree-RAG (DFS and/or Beam) over ~30 HotpotQA questions under both
conditions with the same local Ollama backend, and (c) comparing LLM-Judge/ROUGE-L between
conditions. This is a real experiment requiring live local-LLM inference time, not a code
read — flagged for a follow-up step rather than attempted in this code-audit pass. Given
how directly it bears on interpreting §6.3 and the English-benchmark tables, it's a
strong candidate for the next thing to run if you want B6 fully closed out.

## Answered vs. unmeasured

- **Answered**: traversal prompts are unconditionally Korean (code-certain); confirmed
  exercised on a real English-benchmark run (GovReport, online mode); confirmed the
  confound is asymmetric (PageTree-RAG only); traced the most plausible source of the
  §6.3 Korean-output bug to `DOMAIN_PROMPTS` / the complex-scaffold template in
  `reasoner.py`, not the traversal prompts.
- **Not measured**: whether HotpotQA-specific runs used online (LLM) vs. offline
  (keyword) traversal (schema didn't expose it the way the GovReport file did — would
  need to check the script invocation history or re-derive from `run_exp2_multihop.py`
  logic, not done here); the controlled English-vs-Korean traversal-prompt A/B experiment
  (§5); whether the traversal prompts have any *indirect* effect on answer language via
  model state (would require the same A/B experiment).
