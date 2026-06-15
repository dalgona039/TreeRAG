# TreeRAG KCI 등재 준비 계획서

**목표**: 한국컴퓨터정보학회논문지 또는 한국정보처리학회논문지 KCI 등재  
**예상 소요 기간**: 4-6주  
**현재 상태**: 시스템 구현 완료 (PHASE 1-3), 실험 설계 보강 필요

---

## 현재 갭 분석 (Gap Analysis)

| 항목 | 현재 상태 | KCI 요구 수준 | 우선순위 |
|------|-----------|---------------|----------|
| 테스트 데이터셋 | 자체 제작 5개 Q&A | 공개 벤치마크 50개+ | 🔴 Critical |
| 정량 평가 지표 | 수동 평가 100% | ROUGE, BERTScore, LLM-judge | 🔴 Critical |
| 비교 Baseline | FlatRAG 1개 | BM25, Dense Retrieval 포함 3개+ | 🔴 Critical |
| Ablation Study | 코드 존재, 실제 수치 없음 | 실제 실험 결과표 | 🟡 Important |
| 통계 유의성 검정 | 코드 존재, 미실행 | t-test / Wilcoxon p-value | 🟡 Important |
| DSPy 최적화 결과 | 개선 0% | 원인 분석 및 재실험 or 제외 | 🟢 Optional |

---

## PHASE A: 실험 데이터셋 구축 (1주)

### A-1. 공개 벤치마크 데이터셋 적용
**목표**: 재현 가능한 공개 데이터로 신뢰성 확보

선택지 (현실적 난이도 순):
- **KorQuAD 2.0** (한국어, 문서 기반 QA) — 가장 권장
- **NaturalQuestions subset** (영어, 검색 기반)
- **HotpotQA subset** (영어, 멀티홉 추론 — TreeRAG 강점 부각에 유리)

구현 위치: `benchmarks/datasets/korquad_loader.py`

```python
# 최소 요구사항
- 50개+ Q&A 쌍
- 문서당 10개+ 질문
- Easy/Medium/Hard 난이도 분류 포함
```

### A-2. 도메인 특화 데이터셋 자동 생성
**목표**: 실제 업로드된 PDF 기반 Q&A 자동 생성 (Gemini 활용)

현재 업로드 PDF 활용:
- `생체의공학개론#10.pdf`, `#11.pdf` → Medical/Academic 도메인
- `2204.08939v1-2.pdf` → Research 도메인 (ArXiv)
- `2025학년도 교육과정 반도체공학과.pdf` → Academic 도메인

구현 위치: `benchmarks/datasets/auto_qa_generator.py`

```python
# 목표: PDF 1개당 10-15개 Q&A 자동 생성
# 질문 유형:
#   - Factual (사실 확인): "X는 무엇인가?"
#   - Multi-hop (멀티섹션): "A와 B의 관계는?"
#   - Comparative (비교): "X와 Y의 차이점은?"
```

**산출물**: `benchmarks/datasets/full_benchmark.json` (100개+ Q&A)

---

## PHASE B: Baseline 확장 (1주)

### B-1. BM25 Baseline 구현
**목표**: 키워드 기반 고전적 검색과 비교

구현 위치: `src/core/bm25_baseline.py` (신규)

```python
class BM25Retriever:
    # rank-bm25 라이브러리 사용
    # 문서 청크를 BM25로 인덱싱
    # 쿼리에 대해 상위 K개 청크 반환
    # FlatRAG와 동일한 인터페이스
```

### B-2. Dense Retrieval Baseline 구현
**목표**: 임베딩 기반 벡터 검색과 비교 (현재 TreeRAG의 최대 경쟁 방식)

구현 위치: `src/core/dense_retrieval_baseline.py` (신규)

```python
class DenseRetriever:
    # sentence-transformers: jhgan/ko-sroberta-multitask (한국어) 
    # 또는 intfloat/multilingual-e5-base
    # FAISS 인덱스로 ANN 검색
    # 동일한 평가 인터페이스 준수
```

### B-3. Baseline 통합 비교 테이블

최종 비교 대상 (논문 Table 1):
| System | Description |
|--------|-------------|
| BM25 | 키워드 기반 고전적 검색 |
| Dense Retrieval | 임베딩 + FAISS |
| FlatRAG | Hybrid (BM25 60% + Semantic 25% + Structural 15%) |
| **TreeRAG-DFS** | **제안 방법 (DFS 탐색)** |
| **TreeRAG-Beam** | **제안 방법 (Beam Search 탐색)** |

---

## PHASE C: 자동화 평가 파이프라인 구축 (1주)

### C-1. 텍스트 유사도 지표 구현
구현 위치: `benchmarks/metrics/text_similarity.py` (신규 또는 기존 확장)

```python
# 필수 지표:
metrics = {
    "ROUGE-L": rouge_scorer,       # 문자열 최장공통부분수열 기반
    "BERTScore": bertscore,        # 임베딩 기반 의미 유사도
    "ExactMatch": exact_match,     # 정확 매칭 (간단한 질문용)
}
```

### C-2. LLM-as-Judge 평가기 구현
구현 위치: `benchmarks/metrics/llm_judge.py` (신규)

```python
class GeminiJudge:
    """
    Gemini가 3가지 기준으로 0-5점 채점:
    1. Faithfulness: 소스 문서에 근거한 답변인가?
    2. Relevance: 질문에 실제로 답하는가?
    3. Completeness: 핵심 정보를 모두 포함하는가?
    
    최종 점수 = 3가지 평균 / 5 (0.0-1.0 정규화)
    """
```

### C-3. 통합 평가 실행기 완성
구현 위치: `benchmarks/run_real_evaluation.py` (기존 파일 완성)

```python
# 실행 시 자동으로:
# 1. 각 baseline에 동일 질문 세트 실행
# 2. ROUGE-L, BERTScore, LLM-judge 점수 산출
# 3. 응답 시간, 컨텍스트 토큰 수 기록
# 4. 결과를 JSON + CSV로 저장
# python benchmarks/run_real_evaluation.py --dataset full_benchmark.json
```

**산출물**: `data/benchmark_reports/final_results.json`

---

## PHASE D: Ablation Study 실제 실행 (1주)

### D-1. Ablation 설계
`scripts/ablation_study.py` 기반으로 실제 수치 생성

```
실험 조합 (각각 동일 질문 세트로 평가):
┌─────────────────────────────────────────────────┐
│ Config | Beam | Compress | RefResolver | Score  │
│   1    |  ✗   |    ✗     |     ✗       |  ???   │ ← 기준선
│   2    |  ✓   |    ✗     |     ✗       |  ???   │ ← +Beam
│   3    |  ✓   |    ✓     |     ✗       |  ???   │ ← +Compression
│   4    |  ✓   |    ✓     |     ✓       |  ???   │ ← Full (제안)
└─────────────────────────────────────────────────┘
→ 각 컴포넌트의 개별 기여도 수치화
```

### D-2. 통계적 유의성 검증
`benchmarks/metrics/statistical_tests.py` 실행 (기존 코드 활용)

```python
# TreeRAG-Beam vs FlatRAG:
#   - paired t-test (p < 0.05 필요)
#   - Cohen's d (effect size)
#   - 95% Bootstrap CI
```

### D-3. 시각화 자동 생성
`scripts/plot_results.py` 실행

산출물:
- `Figure 1`: 시스템 전체 아키텍처 다이어그램
- `Figure 2`: Baseline 비교 Bar chart (ROUGE-L, BERTScore)
- `Figure 3`: Ablation study 결과 (컴포넌트별 기여도)
- `Figure 4`: Context reduction vs Accuracy trade-off curve
- `Figure 5`: 응답 시간 분포 box plot

---

## PHASE E: 논문 작성 (2-3주)

### E-1. 논문 구조 (7섹션, 8페이지 목표)

```
1. Introduction (1p)
   - RAG 한계: flat chunk → 문맥 손실
   - TreeRAG 제안 동기: 계층 구조 보존
   - 주요 기여 3가지 bullet

2. Related Work (1p)
   - Traditional RAG (Dense Retrieval, BM25)
   - Hierarchical Retrieval (LlamaIndex, PageIndex)
   - Hallucination Detection in RAG

3. Methodology (2p)
   3.1 Tree Indexing (PDF → JSON Tree)
   3.2 Tree Traversal: DFS vs Beam Search
   3.3 Contextual Compression
   3.4 Hallucination Detection (5-signal)
   
4. System Architecture (0.5p)
   - FastAPI + Redis + Celery 구조도
   - 2-stage pipeline 다이어그램

5. Experiments (2p)
   5.1 Dataset & Evaluation Setup
   5.2 Baseline Comparison (Table 1)
   5.3 Ablation Study (Table 2)
   5.4 Efficiency Analysis (Table 3)
   5.5 Case Study (질적 분석 1개)

6. Discussion (0.5p)
   - 성능 분석
   - 한계점 (DSPy 개선 미달, 영어 문서 한계 등)

7. Conclusion (0.5p)
```

### E-2. 핵심 테이블 3개 (논문의 핵심)

**Table 1: Main Results**
| System | ROUGE-L | BERTScore | LLM-Judge | Latency(s) | CTX(K) |
|--------|---------|-----------|-----------|------------|--------|
| BM25 | ? | ? | ? | ? | ? |
| Dense | ? | ? | ? | ? | ? |
| FlatRAG | ? | ? | ? | ? | ? |
| TreeRAG-DFS | ? | ? | ? | ? | ? |
| **TreeRAG-Beam** | **?** | **?** | **?** | **?** | **?** |

**Table 2: Ablation Study**
| Config | ROUGE-L | Δ vs Full |
|--------|---------|-----------|
| w/o Beam Search | ? | -?% |
| w/o Compression | ? | -?% |
| w/o RefResolver | ? | -?% |
| **Full System** | **?** | **—** |

**Table 3: Efficiency**
| Doc Size | TreeRAG CTX | Flat CTX | Reduction |
|----------|-------------|----------|-----------|
| <50p | ? | ? | ?% |
| 50-100p | ? | ? | ?% |
| >100p | ? | ? | ?% |

### E-3. 투고 전 체크리스트

- [ ] p-value < 0.05 확보 (통계 유의성)
- [ ] 공개 데이터셋 명시 (재현성)
- [ ] 코드 GitHub 공개 링크 포함
- [ ] 영문 Abstract 200단어 이내
- [ ] 영문 Keywords 5개
- [ ] 참고문헌 15개+ (한국어 + 영어 혼합)
- [ ] 투고 양식 준수 (한컴 워드, A4, 2단)

---

## 투고 타깃 학술지 (우선순위 순)

| 순위 | 학술지 | 이유 |
|------|--------|------|
| 1 | 한국컴퓨터정보학회논문지 | 심사 빠름(월 발행), AI/CS 친화적 |
| 2 | 한국정보처리학회논문지 | 시스템 논문 강점, 인지도 높음 |
| 3 | 정보과학회논문지 (소프트웨어 및 응용) | 전통 강호, 심사 기간 김 |
| 4 | 대한의용생체공학회지 | 의료 문서 특화 시 적합 |

---

## 총 실행 일정

| 주차 | 작업 |
|------|------|
| 1주 | PHASE A: 데이터셋 구축 |
| 2주 | PHASE B: Baseline 구현 |
| 3주 | PHASE C: 평가 파이프라인 |
| 4주 | PHASE D: 실험 실행 + 결과 수집 |
| 5-7주 | PHASE E: 논문 작성 |
| 8주 | 투고 |
