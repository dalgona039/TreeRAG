import json
import os
import re
from typing import Any, List, Dict, Optional, Literal
from src.config import Config
from src.core.tree_traversal import TreeNavigator, format_traversal_results
from src.core.beam_search import BeamSearchNavigator, format_beam_results
from src.core.contextual_compressor import ContextualCompressor, format_compressed_context
from src.core.reference_resolver import ReferenceResolver
from src.core.adaptive_policy import score_root_children, choose_traversal_algorithm
from src.utils.cache import get_cache
from src.utils.hallucination_detector import create_detector

# Traversal algorithm types
TraversalAlgorithm = Literal["dfs", "beam_search", "auto"]

DOMAIN_PROMPTS = {
    "general": """당신은 전문 문서 분석 AI 어시스턴트입니다.
제공된 문서의 인덱스를 사용하여 사용자의 질문에 정확하게 답변하세요.""",
    
    "medical": """당신은 의료 전문 AI 어시스턴트입니다.
**의료 문서 분석 원칙:**
- 의학 용어를 정확하게 사용하고 필요시 설명을 추가하세요
- 임상 가이드라인과 근거 기반 의학(EBM)을 준수하세요
- 진단, 치료, 약물에 대한 정보는 반드시 페이지 참조와 함께 제공하세요
- 질문이 **진단/치료/약물/시술의 안전성**을 직접 요구할 때만 부작용·금기사항·주의사항을 명시하세요
- 불확실성 고지는 질문이 요구한 정보 범위 안에서만 작성하세요""",
    
    "legal": """당신은 법률 전문 AI 어시스턴트입니다.
**법률 문서 분석 원칙:**
- 법조문과 조항을 정확히 인용하고 페이지 번호를 명시하세요
- 조건, 예외사항, 단서조항을 빠짐없이 포함하세요
- "~할 수 있다", "~하여야 한다" 등의 법률 용어를 정확히 사용하세요
- 판례나 선례가 언급된 경우 명확히 표시하세요
- 법적 해석이 필요한 부분은 여러 관점을 제시하세요""",
    
    "financial": """당신은 재무/금융 전문 AI 어시스턴트입니다.
**재무 문서 분석 원칙:**
- 숫자, 지표, 통계는 절대적으로 정확해야 하며 반드시 출처를 명시하세요
- 재무제표 항목(자산, 부채, 수익 등)을 정확히 구분하세요
- 회계 기준(K-IFRS, GAAP 등)이 명시된 경우 이를 고려하세요
- 전년 대비 증감률, 비율 등을 제시할 때 계산 근거를 설명하세요
- 리스크 요인, 우발채무 등 주요 재무 위험을 명확히 표시하세요""",
    
    "academic": """당신은 학술 연구 전문 AI 어시스턴트입니다.
**학술 문서 분석 원칙:**
- 연구 방법론, 실험 설계, 데이터 분석 방법을 명확히 구분하세요
- 연구 결과와 저자의 해석/주장을 구분하여 제시하세요
- 통계적 유의성(p-value), 신뢰구간 등 정량적 지표를 정확히 인용하세요
- 선행연구와의 관계, 연구의 한계점을 명시하세요
- 인용 형식을 정확히 따르고 페이지 번호를 반드시 포함하세요"""
}

LANGUAGE_INSTRUCTIONS = {
    "ko": "**중요: 모든 답변은 반드시 한국어로 작성하세요.**",
    "en": "**IMPORTANT: You MUST respond in English only.**",
    "ja": "**重要：必ず日本語で回答してください。**"
}

class TreeRAGReasoner:
    PROMPT_CACHE_VERSION = "2026-03-01-v3"

    # Simple prompts for local LLM backends — one per language so the skeleton
    # language never contradicts the instruction (8B models drift to Korean when
    # the surrounding markers are Korean even with an English language_instruction).
    _SIMPLE_PROMPT_KO = (
        "{domain_prompt}\n\n"
        "{language_instruction}\n\n"
        "아래 컨텍스트를 바탕으로 질문에 한국어로 답하세요. "
        "컨텍스트에 없는 내용은 추측하지 마세요.\n\n"
        "### 컨텍스트:\n{context}\n\n"
        "### 질문:\n{question}\n\n"
        "### 답변:"
    )
    _SIMPLE_PROMPT_EN = (
        "{domain_prompt}\n\n"
        "{language_instruction}\n\n"
        "Answer the question using ONLY the context below. "
        "Do NOT speculate beyond the context. Respond in English.\n\n"
        "### Context:\n{context}\n\n"
        "### Question:\n{question}\n\n"
        "### Answer:"
    )
    _SIMPLE_PROMPT_JA = (
        "{domain_prompt}\n\n"
        "{language_instruction}\n\n"
        "以下のコンテキストのみを使用して質問に答えてください。"
        "コンテキストにない内容は推測しないでください。\n\n"
        "### コンテキスト:\n{context}\n\n"
        "### 質問:\n{question}\n\n"
        "### 回答:"
    )
    # Fallback (legacy alias) — Korean kept for backward compat with cached results.
    _SIMPLE_PROMPT = _SIMPLE_PROMPT_KO
    # Separate cache-version tag so simple-prompt results don't mix with
    # the complex-prompt cache written by the Gemini path.
    PROMPT_CACHE_VERSION_SIMPLE = "2026-07-05-simple-v3"

    @staticmethod
    def _normalize_model_answer(answer_text: str) -> str:
        """Extract a clean answer string from the model's raw output.

        Handles:
        - Plain text (simple prompt path) — returned as-is after whitespace cleanup.
        - JSON-wrapped answers ``{"answer": "..."}`` or similar.
        - Truncated / malformed JSON — extracts the longest string value found,
          or falls back to the raw text so the answer is never empty.
        """
        text = (answer_text or "").strip()
        if not text:
            return text

        # Strip markdown code fences
        if text.startswith("```"):
            inner = text[7:] if text.startswith("```json") else text[3:]
            if "```" in inner:
                text = inner.rsplit("```", 1)[0].strip()

        # Unescape common escape sequences before JSON attempts
        def _unescape(s: str) -> str:
            return s.replace("\\n", "\n").replace("\\t", "\t")

        # --- Attempt 1: full JSON parse ---
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Prefer "answer" key, then common Korean/English alternatives
                for key in ("answer", "답변", "response", "content", "result", "text"):
                    if isinstance(parsed.get(key), str) and parsed[key].strip():
                        return _unescape(parsed[key].strip())
                # Last resort: join all non-empty string values
                parts = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
                if parts:
                    return _unescape(" ".join(parts))
            elif isinstance(parsed, str):
                return _unescape(parsed.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # --- Attempt 2: partial / truncated JSON — regex extraction ---
        if text.lstrip().startswith("{"):
            for pattern in (
                r'"answer"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                r'"답변"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                r'"response"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                r'"content"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                # any key whose value is at least 20 chars
                r'"\w+"\s*:\s*"((?:[^"\\]|\\.){20,})"',
            ):
                m = re.search(pattern, text)
                if m:
                    return _unescape(m.group(1).strip())

            # Strip leading JSON punctuation and return what's left
            stripped = re.sub(r'^[\{\[":\s\w]*', "", text).strip()
            if len(stripped) > 15:
                return _unescape(stripped)

        # --- Fallback: return original text with unescape ---
        return _unescape(text)

    def __init__(
        self, 
        index_filenames: List[str], 
        use_deep_traversal: bool = True,
        traversal_algorithm: TraversalAlgorithm = "beam_search",
        beam_width: int = 5,
        enable_compression: bool = True,
        enable_reference_resolver: bool = True,
        margin_cutoff: float = 0.15
    ):
        self.index_trees: List[Dict[str, Any]] = []
        self.index_filenames = index_filenames
        self.use_deep_traversal = use_deep_traversal
        self.traversal_algorithm = traversal_algorithm
        self.beam_width = beam_width
        self.margin_cutoff = margin_cutoff
        self.enable_compression = enable_compression
        self.enable_reference_resolver = enable_reference_resolver
        self.compressor = ContextualCompressor() if enable_compression else None
        
        for index_filename in index_filenames:
            path = os.path.join(Config.INDEX_DIR, index_filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self.index_trees.append(json.load(f))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in index file {index_filename}: {e}")
            except IOError as e:
                raise IOError(f"Failed to read index file {index_filename}: {e}")


    @staticmethod
    def _detect_language(text: str) -> str:
        if re.search(r'[가-힣]', text):
            return "ko"
        if re.search(r'[ぁ-ゖァ-ヺ一-龯]', text):
            return "ja"
        return "en"

    def _resolve_language(self, user_question: str, language: Optional[str]) -> str:
        if language in LANGUAGE_INSTRUCTIONS:
            return language
        return self._detect_language(user_question)

    def _build_no_context_response(self, language: str) -> str:
        if language == "ko":
            return (
                "제공된 인덱스에서 질문과 직접적으로 관련된 섹션을 찾지 못했습니다.\n\n"
                "현재 문서만으로는 답변을 확정할 수 없어 추가 확인이 필요합니다.\n\n"
                "📚 참조 페이지: 없음"
            )
        if language == "ja":
            return (
                "提供されたインデックスから、質問に直接関連するセクションを見つけられませんでした。\n\n"
                "現在の文書だけでは回答を確定できないため、追加確認が必要です。\n\n"
                "📚 参照ページ: なし"
            )
        return (
            "No sections directly relevant to the question were found in the provided index.\n\n"
            "The current document alone is insufficient to provide a definitive answer, so further verification is required.\n\n"
            "📚 Reference Pages: None"
        )

    def query(self, user_question: str, enable_comparison: bool = True, max_depth: int = 5, max_branches: int = 3, domain_template: str = "general", language: Optional[str] = "auto", node_context: Optional[dict] = None, use_simple_prompt: bool = False) -> tuple[str, dict]:
        if not user_question or not user_question.strip():
            raise ValueError("user_question cannot be empty")

        language = self._resolve_language(user_question, language)
        cache_node_context = dict(node_context) if node_context else {}
        # Use a distinct cache-version for simple-prompt runs so they don't
        # collide with (or return) complex-prompt / Gemini cached results.
        cache_version = (
            self.PROMPT_CACHE_VERSION_SIMPLE if use_simple_prompt
            else self.PROMPT_CACHE_VERSION
        )
        cache_node_context["__prompt_cache_version"] = cache_version
        
        cache = get_cache()
        cached_response = cache.get(
            question=user_question,
            index_files=self.index_filenames,
            use_deep_traversal=self.use_deep_traversal,
            max_depth=max_depth,
            max_branches=max_branches,
            domain_template=domain_template,
            language=language,
            node_context=cache_node_context
        )
        
        if cached_response:
            print(f"✅ Cache HIT - Returning cached response")
            normalized_cached = self._normalize_model_answer(cached_response["answer"])
            return normalized_cached, cached_response["metadata"]
        
        print(f"❌ Cache MISS - Generating new response")
        
        traversal_info = {
            "used_deep_traversal": self.use_deep_traversal,
            "nodes_visited": [],
            "nodes_selected": [],
            "max_depth": max_depth,
            "max_branches": max_branches
        }
        
        reference_context = ""
        resolved_refs = []
        for tree in (self.index_trees if self.enable_reference_resolver else []):
            resolver = ReferenceResolver(tree)
            refs = resolver.detect_references(user_question)
            if refs:
                resolved_nodes = resolver.resolve_all_references(user_question)
                if resolved_nodes:
                    resolved_refs.extend(resolved_nodes)
                    ref_context = resolver.format_resolved_context(resolved_nodes)
                    if ref_context:
                        reference_context += ref_context
                        print(f"📎 Resolved {len(resolved_nodes)} cross-references: {[r.get('title') for r in resolved_nodes]}")
        
        if self.use_deep_traversal:
            print("🌲 Using deep tree traversal mode")
            context_str, trav_data = self._build_context_with_traversal(user_question, max_depth, max_branches)
            traversal_info.update(trav_data)
        else:
            print("📄 Using flat context mode (legacy)")
            context_str = self._build_flat_context()
        
        if reference_context:
            context_str = reference_context + "\n\n" + context_str

        if not traversal_info["nodes_selected"] and not resolved_refs:
            traversal_info["fallback_reason"] = "no_relevant_sections"
            traversal_info["detected_language"] = language
            traversal_info["selected_documents"] = [
                filename.replace("_index.json", "") for filename in self.index_filenames
            ]
            fallback_answer = self._build_no_context_response(language)
            print(
                "⚠️ No relevant sections selected "
                f"(algorithm={self.traversal_algorithm}, docs={len(self.index_filenames)}, "
                f"max_depth={max_depth}, max_branches={max_branches})"
            )
            return fallback_answer, traversal_info
        
        is_multi_doc = len(self.index_filenames) > 1
        comparison_prompt = ""
        
        if is_multi_doc and enable_comparison:
            comparison_prompt = f"""

### 📊 다중 문서 비교 분석 (필수):
여러 문서가 제공되었으므로, 반드시 다음 형식으로 비교 분석을 포함하세요:

**1. 공통점 (Commonalities)**
- 모든 문서에서 일치하는 내용
- 예: "모든 교육과정에서 졸업 학점은 130학점 이상 [문서A, p.5], [문서B, p.3]"

**2. 차이점 (Differences)**
표 형식으로 명확히 구분:
| 항목 | {self.index_filenames[0].replace('_index.json', '')} | {self.index_filenames[1].replace('_index.json', '') if len(self.index_filenames) > 1 else '기타'} |
|------|------|------|
| 예: 필수학점 | 18학점 [p.5] | 21학점 [p.4] |
| 예: 선택과목 | 10개 [p.7] | 15개 [p.6] |

**3. 문서 우선순위 (해당시)**
- 충돌하는 내용이 있다면, 어떤 문서가 최신/공식인지 명시
- 예: "최신 버전(2024)의 내용이 적용됩니다 [문서A, p.10]"
"""

        # Precomputed to keep the prompt f-string free of backslash-in-expression
        # (portable across Python 3.10–3.12).
        comparison_template = "\n[문서 비교 분석: 공통점/차이점 표]" if is_multi_doc else ""

        domain_prompt = DOMAIN_PROMPTS.get(domain_template, DOMAIN_PROMPTS["general"])

        language_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["ko"])

        # ------------------------------------------------------------------ #
        # Prompt selection: simple (local models) vs complex scaffold (Gemini)
        # ------------------------------------------------------------------ #
        if use_simple_prompt:
            _lang_templates = {
                "ko": self._SIMPLE_PROMPT_KO,
                "ja": self._SIMPLE_PROMPT_JA,
            }
            _tmpl = _lang_templates.get(language, self._SIMPLE_PROMPT_EN)
            prompt = _tmpl.format(
                domain_prompt=domain_prompt,
                language_instruction=language_instruction,
                context=context_str,
                question=user_question,
            )
        else:
            prompt = f"""
{domain_prompt}

{language_instruction}

### 📋 답변 작성 단계 (반드시 순서대로):

**STEP 1: 질문 핵심 파악**
- 질문에서 요구하는 핵심 정보가 무엇인지 명확히 파악
- 예: "졸업 학점은?" → 숫자(학점) 찾기, "필수 과목은?" → 과목명 리스트 찾기

**STEP 2: 인덱스에서 정확한 정보 검색**
- 제공된 인덱스 JSON에서 질문과 관련된 섹션 찾기
- page_ref, title, summary 필드를 활용하여 정확한 위치 특정

**STEP 3: 핵심 답변 먼저 작성 (1-2문장)**
- 질문에 대한 직접적인 답변을 먼저 명확하게 제시
- 반드시 페이지 참조 포함: [문서명, p.페이지]
- 예: "졸업 학점은 130학점입니다 [인공지능반도체, p.2]."

**STEP 4: 상세 설명 추가 (필요시)**
- 핵심 답변 이후 추가 맥락이나 상세 정보 제공
- 모든 문장에 페이지 참조 포함

**STEP 5: 참조 페이지 요약**
- 답변 마지막에 📚 **참조 페이지** 형식으로 모든 출처 나열

### ⚠️ 중요 규칙:

1. **인덱스에 없는 정보는 절대 추측하지 마세요** - "인덱스에 해당 정보가 명시되어 있지 않습니다"라고 답변
2. **페이지 번호 필수** - 모든 사실적 진술에 [문서명, p.번호] 형식으로 표기
3. **간결하고 정확하게** - 질문에 직접 답하는 정보를 우선 제시
4. **숫자/이름은 정확히** - 학점 수, 과목명, 날짜 등은 인덱스에 있는 그대로 기재
5. **질문 범위 밖 문장 금지** - 질문에서 묻지 않은 주제(예: 임상적 부작용/금기사항, 법적 면책 문구, 투자 주의문)는 임의로 추가하지 마세요
6. **마무리 문장 제한** - 답변 마지막에는 참조 페이지만 작성하고, 일반론적 주의 문구를 덧붙이지 마세요
{comparison_prompt}

### 답변 템플릿:

[핵심 답변 1-2문장 + 페이지 참조]

[상세 설명 (필요시) + 페이지 참조]
{comparison_template}

📚 **참조 페이지**: [문서명, p.X], [문서명, p.Y-Z]

### 컨텍스트:
{context_str}

### 질문:
{user_question}

### 답변 (위 규칙을 철저히 따라 작성):
"""

        try:
            from src.config import _client_override
            if _client_override is not None and use_simple_prompt:
                # Simple prompt → plain natural-language answer.
                # Do NOT pass response_mime_type="application/json" — that forces
                # the local model into JSON mode and corrupts the output.
                gen_cfg = None
            elif _client_override is not None:
                gen_cfg = Config.get_generation_config(max_output_tokens=2048)
            else:
                gen_cfg = Config.get_generation_config()

            response = Config.get_client("reasoning").models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=gen_cfg,
            )
            if not response.text:
                raise ValueError("Empty response from model")

            answer_text = self._normalize_model_answer(response.text)
            if not answer_text:
                # If normalization stripped everything, return raw text
                answer_text = (response.text or "").strip()
            if len(answer_text) < 10:
                print(f"⚠️ Very short answer ({len(answer_text)} chars): {answer_text!r} — using raw response")
                answer_text = (response.text or "").strip() or answer_text
            
            if resolved_refs:
                traversal_info["resolved_references"] = [
                    {
                        "title": ref.get("title", ""),
                        "page_ref": ref.get("page_ref"),
                        "summary": ref.get("summary")
                    }
                    for ref in resolved_refs
                ]
            
            detector = create_detector(sentence_threshold=0.55, overall_threshold=0.45)
            
            source_nodes = list(traversal_info.get("nodes_selected", []))
            if not source_nodes:
                if self.use_deep_traversal:
                    for tree_idx, tree in enumerate(self.index_trees):
                        doc_name = self.index_filenames[tree_idx].replace("_index.json", "")
                        navigator = TreeNavigator(tree, doc_name)
                        relevant_nodes, _ = navigator.search(
                            query=user_question,
                            max_depth=max_depth,
                            max_branches=max_branches
                        )
                        source_nodes.extend([node["node"] for node in relevant_nodes])
                else:
                    for tree in self.index_trees:
                        source_nodes.extend(self._extract_all_nodes(tree))
            
            if resolved_refs:
                source_nodes.extend(resolved_refs)
            
            detection_result = detector.detect(answer_text, source_nodes)
            
            traversal_info["hallucination_detection"] = {
                "overall_confidence": detection_result["overall_confidence"],
                "is_reliable": detection_result["is_reliable"],
                "hallucinated_count": detection_result["hallucinated_count"],
                "total_sentences": detection_result["total_sentences"]
            }
            
            if detection_result["is_reliable"]:
                print(f"✅ Hallucination check: {detection_result['overall_confidence']:.1%} confidence")
            else:
                print(f"⚠️ Hallucination detected: {detection_result['hallucinated_count']}/{detection_result['total_sentences']} sentences low confidence")
            

            cache = get_cache()
            cache_data = {
                "answer": answer_text,
                "metadata": traversal_info
            }
            cache.set(
                question=user_question,
                index_files=self.index_filenames,
                use_deep_traversal=self.use_deep_traversal,
                max_depth=max_depth,
                max_branches=max_branches,
                domain_template=domain_template,
                language=language,
                response=cache_data,
                node_context=cache_node_context
            )
            print(f"💾 Response cached")
            
            return answer_text, traversal_info
        except Exception as e:
            print(f"❌ Query failed: {e}")
            raise
    
    def _build_context_with_traversal(self, query: str, max_depth: int, max_branches: int) -> tuple[str, dict]:
        all_results = []
        all_visited = []
        all_selected = []
        auto_selected_algorithms = []

        for idx, tree in enumerate(self.index_trees):
            doc_name = self.index_filenames[idx].replace("_index.json", "")

            effective_algorithm = self.traversal_algorithm
            if self.traversal_algorithm == "auto":
                root_scores = score_root_children(tree, query)
                effective_algorithm = choose_traversal_algorithm(
                    root_scores, margin_cutoff=self.margin_cutoff
                )
                auto_selected_algorithms.append({
                    "document": doc_name,
                    "selected": effective_algorithm,
                    "root_children_scores": root_scores
                })
                print(f"🎯 Auto policy: {effective_algorithm} "
                      f"(root scores={['%.2f' % s for s in root_scores]}, "
                      f"margin_cutoff={self.margin_cutoff})")

            # 알고리즘에 따라 다른 Navigator 사용
            if effective_algorithm == "beam_search":
                print(f"🔍 Using Beam Search (width={self.beam_width})")
                navigator = BeamSearchNavigator(tree, doc_name, beam_width=self.beam_width)
                relevant_nodes, trav_stats = navigator.search(
                    query=query,
                    max_depth=max_depth,
                    min_score_threshold=0.2
                )
                formatted = format_beam_results(relevant_nodes, doc_name)
            else:
                print(f"🔍 Using DFS (branches={max_branches})")
                navigator = TreeNavigator(tree, doc_name)
                relevant_nodes, trav_stats = navigator.search(
                    query=query,
                    max_depth=max_depth,
                    max_branches=max_branches
                )
                formatted = format_traversal_results(relevant_nodes, doc_name)

            all_results.append(formatted)

            # 알고리즘에 따라 다른 통계 형식 처리
            if effective_algorithm == "beam_search":
                # Beam Search: nodes_visited는 ID 목록, nodes_selected는 점수 포함
                all_visited.extend([f"{doc_name}: node_{i}" for i in range(trav_stats.get("nodes_visited", 0) if isinstance(trav_stats.get("nodes_visited"), int) else len(trav_stats.get("nodes_visited", [])))])
                all_selected.extend([{
                    "document": doc_name,
                    "id": node["node"].get("id", ""),
                    "title": node["node"].get("title", "Untitled"),
                    "page_ref": node["node"].get("page_ref", "N/A"),
                    "score": node.get("score", 0.0),
                    "content": node["node"].get("summary", "")
                } for node in relevant_nodes])
            else:
                # DFS: visited_titles 목록
                all_visited.extend([f"{doc_name}: {title}" for title in trav_stats.get("visited_titles", [])])
                all_selected.extend([{
                    "document": doc_name,
                    "id": node["node"].get("id", ""),
                    "title": node["node"].get("title", "Untitled"),
                    "page_ref": node["node"].get("page_ref", "N/A"),
                    "content": node["node"].get("summary", "")
                } for node in relevant_nodes])
        
        traversal_data = {
            "algorithm": self.traversal_algorithm,
            "beam_width": self.beam_width if self.traversal_algorithm == "beam_search" else None,
            "nodes_visited": all_visited,
            "nodes_selected": all_selected
        }
        if self.traversal_algorithm == "auto":
            traversal_data["auto_selected_algorithm"] = auto_selected_algorithms
            traversal_data["margin_cutoff"] = self.margin_cutoff
        
        final_context = "\n\n---\n\n".join(all_results)
        
        if self.enable_compression and self.compressor and all_selected:
            print(f"🗜️ Applying contextual compression ({len(all_selected)} nodes)")
            compression_result = self.compressor.compress(all_selected, query)
            
            if compression_result.compressed_count < compression_result.original_count:
                final_context = format_compressed_context(compression_result)
                traversal_data["compression"] = {
                    "original_count": compression_result.original_count,
                    "compressed_count": compression_result.compressed_count,
                    "ratio": compression_result.compression_ratio,
                    "tokens_saved": compression_result.total_tokens_saved
                }
                print(f"   Compressed: {compression_result.original_count} → {compression_result.compressed_count} nodes")
                print(f"   Tokens saved: ~{compression_result.total_tokens_saved}")
        
        return final_context, traversal_data
    
    def _build_flat_context(self) -> str:
        combined_context = []
        for idx, tree in enumerate(self.index_trees):
            doc_name = self.index_filenames[idx].replace("_index.json", "")
            combined_context.append({
                "document": doc_name,
                "content": tree
            })
        
        return json.dumps(combined_context, ensure_ascii=False)
    
    def _extract_all_nodes(self, tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = []
        
        def traverse(node):
            if isinstance(node, dict):
                nodes.append(node)
                if "children" in node and isinstance(node["children"], list):
                    for child in node["children"]:
                        traverse(child)
        
        traverse(tree)
        return nodes