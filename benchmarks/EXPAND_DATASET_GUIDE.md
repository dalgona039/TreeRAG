# HotpotQA 평가 확장 가이드 (n=20 → 100~200)

리뷰어의 "표본 크기" 지적을 해소하기 위해 멀티홉(HotpotQA) 평가를 100~200문항으로
확장하는 절차입니다. **이 작업은 (1) HotpotQA dev set 다운로드와 (2) LLM 백엔드가
필요하므로, API 키와 네트워크가 있는 본인 머신에서 실행해야 합니다.** (Cowork 샌드박스는
외부 데이터셋 호스트 접근이 차단되어 있고 Ollama/Gemini 백엔드가 없어 실제 실행은 불가합니다.)

## 1. HotpotQA dev set 내려받기 (1회)

```bash
mkdir -p data/hotpotqa
# 배포 미러에서 distractor dev set 다운로드 (약 45MB)
curl -L -o data/hotpotqa/hotpot_dev_distractor_v1.json \
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
# 위 호스트가 403이면 HuggingFace에서:
#   pip install datasets
#   python -c "from datasets import load_dataset as L; import json; \
#     d=L('hotpotqa/hotpot_qa','distractor',split='validation'); \
#     json.dump([dict(x) for x in d], open('data/hotpotqa/hotpot_dev_distractor_v1.json','w'))"
```

로더는 `data/hotpotqa/hotpot_dev_distractor_v1.json`(또는 환경변수
`HOTPOTQA_LOCAL=/경로/파일.json`)을 자동으로 우선 사용합니다. 파일이 있으면 이후 실행은
완전히 오프라인으로 동작합니다.

## 2. LLM 백엔드 준비

`run_exp2_multihop.py`는 Ollama(`llama3.1:8b`)로 생성·평가합니다.

```bash
# Ollama 사용 시
ollama pull llama3.1:8b
# 또는 Gemini 사용 시 .env에 실제 키 입력
#   GOOGLE_API_KEY=...  (GOOGLE_API_KEY_REASONING 등)
```

## 3. 확장 평가 실행

```bash
# n=100 (권장 시작점)
python benchmarks/run_exp2_multihop.py            # 기본 build_hotpotqa_dataset(n=100)

# n=200 으로 늘리려면 run_exp2_multihop.py의 build_hotpotqa_dataset(n=100)
#   호출을 n=200 으로 바꾸거나, --limit 으로 상한을 지정
```

실행이 끝나면 `data/benchmark_reports/exp2_multihop_hotpotqa_<timestamp>.json/.md`가
생성됩니다.

## 4. 강건 통계 재계산 (확장 데이터 반영)

새 결과 파일명으로 `scripts/robust_stats.py`의 `BENCH` 딕셔너리 HotpotQA 항목을 교체한 뒤:

```bash
python scripts/robust_stats.py
```

부트스트랩 CI·순열검정·검정력·필요표본수가 재계산되어
`data/benchmark_reports/robust_stats_summary.json`에 저장됩니다. 이 수치로 논문 Table 5와
5.6/6.4 절을 갱신하면 됩니다.

## 권장 규모

- **HotpotQA**: 20 → **100** (효과크기가 매우 커서 100이면 충분히 과포화된 검정력 확보).
  여력이 되면 200까지.
- **Medical (n=42)**: BM25 대비 ROUGE-L 차이가 작아(d≈0.09) 1,000+ 문항이 필요하므로,
  ROUGE-L 우위 주장 대신 entity recall·citation 지표 중심 서술 유지 권장(논문 5.6 반영됨).
- **General (n=204)**: 이미 충분.
