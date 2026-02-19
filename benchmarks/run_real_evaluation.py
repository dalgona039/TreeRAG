
import sys
import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.run_evaluation import (
    EvaluationRunner, EvaluationConfig, BenchmarkDataset, 
    BenchmarkQuestion, SystemResult
)
from benchmarks.compare_baselines import BaselineType
from benchmarks.metrics.retrieval_metrics import (
    RetrievalResult, QueryResult, create_query_result
)
from benchmarks.metrics.efficiency_metrics import LatencyMeasurement, TokenUsage
from benchmarks.metrics.fidelity_metrics import FidelityAnalysis


class RealTreeRAGEvaluator(EvaluationRunner):
    
    def __init__(
        self, 
        config: EvaluationConfig,
        api_base_url: str = "http://localhost:8000"
    ):

        super().__init__(config)
        self.api_base_url = api_base_url.rstrip('/')
        
    async def _call_treerag_api(
        self, 
        question: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        
        url = f"{self.api_base_url}/chat"
        
        payload = {
            "question": question,
            "use_contextual_compression": True,
            "use_hallucination_detection": True
        }
        
        if document_id:
            payload["document_id"] = document_id
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise Exception(f"API 오류: {error}")
                
                return await response.json()
    
    def _evaluate_treerag(self) -> SystemResult:
        
        print("TreeRAG 실제 시스템 평가 시작...")
        
        result = SystemResult(
            system_name="TreeRAG (Real)",
            system_type=BaselineType.TREE_RAG
        )
        
        query_results = []
        latency_measurements = []
        token_usages = []
        fidelity_analyses = []
        
        loop = asyncio.get_event_loop()
        
        for idx, question in enumerate(self.dataset.questions, 1):
            print(f"  [{idx}/{len(self.dataset.questions)}] {question.question_id}")
            
            try:
                start_time = time.perf_counter()

                api_response = loop.run_until_complete(
                    self._call_treerag_api(
                        question.question,
                        question.document_id
                    )
                )
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000

                answer = api_response.get("answer", "")
                context = api_response.get("context", "")
                traversal = api_response.get("traversal", {})

                visited_sections = traversal.get("visited_sections", [])
                retrieved = []
                
                for i, section in enumerate(visited_sections[:10]):
                    section_id = section.get("id", f"section_{i}")
                    score = section.get("score", 0.0)

                    is_relevant = section_id in question.relevant_sections
                    
                    retrieved.append(RetrievalResult(
                        doc_id=section_id,
                        rank=i + 1,
                        score=score,
                        relevance=1.0 if is_relevant else 0.0
                    ))

                qr = QueryResult(
                    query_id=question.question_id,
                    query_text=question.question,
                    retrieved=retrieved,
                    relevant_doc_ids=set(question.relevant_sections),
                    latency_ms=latency_ms
                )
                query_results.append(qr)

                latency_measurements.append(LatencyMeasurement(
                    query_id=question.question_id,
                    total_ms=latency_ms,
                    traversal_ms=traversal.get("time_ms", 0),
                    llm_ms=api_response.get("llm_time_ms", 0)
                ))

                tokens_info = api_response.get("tokens", {})
                token_usages.append(TokenUsage(
                    query_id=question.question_id,
                    input_tokens=tokens_info.get("input", 0),
                    output_tokens=tokens_info.get("output", 0),
                    total_tokens=tokens_info.get("total", 0),
                    context_tokens=len(context) // 4,
                    original_document_tokens=tokens_info.get("original", 10000)
                ))

                fidelity_analyses.append(
                    self.fidelity_metrics.analyze_answer(
                        question.question_id,
                        answer,
                        context
                    )
                )

                result.per_query_scores[question.question_id] = {
                    "latency": latency_ms,
                    "tokens": tokens_info.get("total", 0)
                }
                
                print(f"    ✓ 완료: {latency_ms:.1f}ms")
                
            except Exception as e:
                print(f"    ✗ 오류: {e}")
                query_results.append(QueryResult(
                    query_id=question.question_id,
                    query_text=question.question,
                    retrieved=[],
                    relevant_doc_ids=set(question.relevant_sections),
                    latency_ms=0.0
                ))

        result.retrieval_metrics = self.retrieval_metrics.compute_all_metrics(
            query_results, self.config.k_values
        )
        
        for measurement in latency_measurements:
            self.efficiency_metrics.record_latency(measurement)
        for usage in token_usages:
            self.efficiency_metrics.record_tokens(usage)
        result.efficiency_metrics = self.efficiency_metrics.compute_all()
        
        result.fidelity_metrics = self.fidelity_metrics.compute_metrics(fidelity_analyses)
        
        self.efficiency_metrics.clear()
        
        print("  ✓ TreeRAG 평가 완료")
        return result


def main():
    """실제 TreeRAG 시스템 평가 실행"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="실제 TreeRAG API로 벤치마크 평가"
    )
    
    parser.add_argument(
        "--questions", "-q",
        default="benchmarks/datasets/my_benchmark_questions.json",
        help="질문 데이터셋 경로"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="TreeRAG API URL"
    )
    
    parser.add_argument(
        "--experiment", "-e",
        default="real_evaluation",
        help="실험 이름"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="benchmarks/results",
        help="결과 저장 디렉토리"
    )
    
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"❌ 질문 데이터셋을 찾을 수 없습니다: {questions_path}")
        print("\n먼저 질문 데이터셋을 작성하세요.")
        print("예시: benchmarks/datasets/my_benchmark_questions.json")
        return

    print(f"🔗 TreeRAG API 연결 확인: {args.api_url}")
    import requests
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  API 응답 이상: {response.status_code}")
            print("계속 진행하시겠습니까? (y/n)")
            if input().lower() != 'y':
                return
    except Exception as e:
        print(f"⚠️  API 연결 실패: {e}")
        print("TreeRAG 서버가 실행 중인지 확인하세요.")
        print("계속 진행하시겠습니까? (y/n)")
        if input().lower() != 'y':
            return

    print("✓ API 연결 확인 완료\n")

    config = EvaluationConfig(
        experiment_name=args.experiment,
        systems=["treerag"],
        questions_path=args.questions,
        output_dir=args.output,
        verbose=True
    )

    evaluator = RealTreeRAGEvaluator(config, api_base_url=args.api_url)
    report = evaluator.run()
    
    print("\n" + "=" * 70)
    print("실제 시스템 평가 완료!")
    print("=" * 70)
    print(f"\n결과 확인:")
    print(f"  python scripts/view_results.py benchmarks/results/{args.experiment}/evaluation_report.json")
    print()


if __name__ == "__main__":
    main()
