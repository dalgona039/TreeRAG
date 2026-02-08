
import time
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os

from .metrics import EvaluationMetrics


@dataclass
class QueryTestCase:
    """
    단일 테스트 케이스
    
    Attributes:
        query: 사용자 질문
        relevant_docs: 정답 문서 ID 집합 (ground truth)
        relevant_scores: 문서별 relevance score (NDCG용)
        expected_citations: 기대되는 인용 (예: {'doc1#p10', 'doc2#p5'})
        category: 질문 유형 (fact, comparison, multi-hop 등)
        domain: 도메인 (medical, legal, academic, etc.)
    """
    query: str
    relevant_docs: List[str]
    relevant_scores: Optional[Dict[str, float]] = None
    expected_citations: Optional[List[str]] = None
    category: str = "general"
    domain: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """
    벤치마크 실행 결과
    
    Attributes:
        test_case: 테스트 케이스
        retrieved_docs: 시스템이 반환한 문서 ID들
        generated_answer: 생성된 답변
        latency_ms: 응답 시간 (밀리초)
        context_size: 사용된 컨텍스트 크기 (tokens)
        metrics: 계산된 메트릭들
    """
    test_case: QueryTestCase
    system_name: str
    retrieved_docs: List[str]
    generated_answer: str
    latency_ms: float
    context_size: int
    metrics: Dict[str, float]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'test_case': self.test_case.to_dict(),
            'system_name': self.system_name,
            'retrieved_docs': self.retrieved_docs,
            'generated_answer': self.generated_answer,
            'latency_ms': self.latency_ms,
            'context_size': self.context_size,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }
        return result


class BenchmarkFramework:
    """
    TreeRAG 평가를 위한 종합 벤치마크 프레임워크
    
    Usage:
        framework = BenchmarkFramework()
        
        # 테스트 케이스 추가
        framework.add_test_case(QueryTestCase(
            query="인슐린 저항성 치료는?",
            relevant_docs=['doc1_node5', 'doc1_node12'],
            relevant_scores={'doc1_node5': 1.0, 'doc1_node12': 0.8},
            category='medical'
        ))
        
        # 실행
        results = framework.run_benchmark(tree_rag_system, flat_rag_system)
        
        # 리포트 생성
        report = framework.generate_report(results)
    """
    
    def __init__(self):
        self.test_cases: List[QueryTestCase] = []
        self.results: List[BenchmarkResult] = []
    
    def add_test_case(self, test_case: QueryTestCase):
        """테스트 케이스 추가"""
        self.test_cases.append(test_case)
    
    def add_test_cases_from_json(self, json_path: str):
        """
        JSON 파일에서 테스트 케이스 일괄 로드
        
        Format:
        {
            "test_cases": [
                {
                    "query": "...",
                    "relevant_docs": ["doc1", "doc2"],
                    "relevant_scores": {"doc1": 1.0, "doc2": 0.8},
                    "expected_citations": ["doc1#p10"],
                    "category": "medical",
                    "domain": "medical"
                }
            ]
        }
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for tc in data.get('test_cases', []):
            self.add_test_case(QueryTestCase(**tc))
        
        print(f"✅ Loaded {len(data.get('test_cases', []))} test cases from {json_path}")
    
    def run_single_query(
        self,
        system: Any,
        test_case: QueryTestCase,
        system_name: str
    ) -> BenchmarkResult:
        """
        단일 쿼리 실행 및 평가
        
        Args:
            system: TreeRAGReasoner 또는 FlatRAGBaseline 인스턴스
            test_case: 테스트 케이스
            system_name: 'TreeRAG' 또는 'FlatRAG'
            
        Returns:
            BenchmarkResult
        """
        print(f"  🔍 Query: {test_case.query[:60]}...")
        
        start_time = time.time()
        
        try:
            answer, metadata = system.query(
                test_case.query,
                max_depth=5,
                max_branches=3
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            retrieved_docs = self._extract_retrieved_docs(metadata)
            context_size = metadata.get('context_size', 0)
            
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            return BenchmarkResult(
                test_case=test_case,
                system_name=system_name,
                retrieved_docs=[],
                generated_answer="",
                latency_ms=0.0,
                context_size=0,
                metrics={
                    'precision@3': 0.0,
                    'recall@3': 0.0,
                    'f1@3': 0.0,
                    'error': str(e)
                }
            )
        
        metrics = self._calculate_metrics(
            test_case=test_case,
            retrieved_docs=retrieved_docs,
            generated_answer=answer,
            context_size=context_size
        )
        
        print(f"    ✅ P@3={metrics['precision@3']:.3f}, R@3={metrics['recall@3']:.3f}, "
              f"F1@3={metrics['f1@3']:.3f}, {latency_ms:.0f}ms")
        
        return BenchmarkResult(
            test_case=test_case,
            system_name=system_name,
            retrieved_docs=retrieved_docs,
            generated_answer=answer,
            latency_ms=latency_ms,
            context_size=context_size,
            metrics=metrics
        )
    
    def run_benchmark(
        self,
        tree_rag_system: Any,
        flat_rag_system: Optional[Any] = None,
        save_results: bool = True,
        output_dir: str = "benchmark_results"
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        전체 벤치마크 실행
        
        Args:
            tree_rag_system: TreeRAGReasoner 인스턴스
            flat_rag_system: FlatRAGBaseline 인스턴스 (None이면 TreeRAG만 테스트)
            save_results: 결과를 JSON으로 저장할지 여부
            output_dir: 결과 저장 디렉토리
            
        Returns:
            {
                'TreeRAG': [BenchmarkResult, ...],
                'FlatRAG': [BenchmarkResult, ...]  # flat_rag_system이 있는 경우만
            }
        """
        if not self.test_cases:
            raise ValueError("No test cases added. Use add_test_case() first.")
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Benchmark with {len(self.test_cases)} test cases")
        print(f"{'='*60}\n")
        
        results = {'TreeRAG': []}
        
        print("📊 Evaluating TreeRAG...")
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}]", end=" ")
            result = self.run_single_query(tree_rag_system, test_case, 'TreeRAG')
            results['TreeRAG'].append(result)
        
        if flat_rag_system:
            results['FlatRAG'] = []
            print(f"\n📊 Evaluating FlatRAG...")
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"[{i}/{len(self.test_cases)}]", end=" ")
                result = self.run_single_query(flat_rag_system, test_case, 'FlatRAG')
                results['FlatRAG'].append(result)
        
        print(f"\n{'='*60}")
        print("✅ Benchmark completed!")
        print(f"{'='*60}\n")
        
        if save_results:
            self._save_results(results, output_dir)
        
        self.results = results
        return results
    
    def _extract_retrieved_docs(self, metadata: Dict[str, Any]) -> List[str]:
        """메타데이터에서 검색된 문서 ID 추출"""
        if 'traversal_info' in metadata:
            nodes = metadata['traversal_info'].get('nodes_selected', [])
            return [node.get('node', {}).get('id', '') for node in nodes]
        
        if 'retrieved_docs' in metadata:
            return metadata['retrieved_docs']
        
        return []
    
    def _calculate_metrics(
        self,
        test_case: QueryTestCase,
        retrieved_docs: List[str],
        generated_answer: str,
        context_size: int
    ) -> Dict[str, float]:
        """각 결과에 대한 메트릭 계산"""
        
        relevant_set = set(test_case.relevant_docs)
        metrics = {}
        
        for k in [1, 3, 5]:
            metrics[f'precision@{k}'] = EvaluationMetrics.precision_at_k(
                retrieved_docs, relevant_set, k
            )
            metrics[f'recall@{k}'] = EvaluationMetrics.recall_at_k(
                retrieved_docs, relevant_set, k
            )
            metrics[f'f1@{k}'] = EvaluationMetrics.f1_at_k(
                retrieved_docs, relevant_set, k
            )
        
        if test_case.relevant_scores:
            for k in [3, 5]:
                metrics[f'ndcg@{k}'] = EvaluationMetrics.ndcg_at_k(
                    retrieved_docs, test_case.relevant_scores, k
                )
        
        if test_case.expected_citations:
            citation_acc, citation_details = EvaluationMetrics.citation_accuracy(
                generated_answer,
                set(test_case.expected_citations)
            )
            metrics['citation_accuracy'] = citation_acc
            metrics['citations_found'] = citation_details['correct']
            metrics['citations_missing'] = citation_details['missing']
        
        if retrieved_docs and generated_answer:
            metrics['answer_length'] = len(generated_answer)
        
        metrics['context_size'] = context_size
        
        return metrics
    
    def _save_results(self, results: Dict[str, List[BenchmarkResult]], output_dir: str):
        """결과를 JSON 파일로 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        serializable_results = {}
        for system_name, result_list in results.items():
            serializable_results[system_name] = [r.to_dict() for r in result_list]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def generate_report(
        self,
        results: Optional[Dict[str, List[BenchmarkResult]]] = None
    ) -> str:
        """
        벤치마크 결과 종합 리포트 생성
        
        Args:
            results: run_benchmark()의 결과 (None이면 self.results 사용)
            
        Returns:
            사람이 읽기 쉬운 텍스트 리포트
        """
        if results is None:
            results = self.results
        
        if not results:
            return "No results to report. Run benchmark first."
        
        report = []
        report.append("=" * 80)
        report.append("TreeRAG COMPREHENSIVE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Cases: {len(self.test_cases)}")
        report.append("")
        
        for system_name, result_list in results.items():
            report.append("-" * 80)
            report.append(f"📊 {system_name} Results")
            report.append("-" * 80)
            
            all_metrics = [r.metrics for r in result_list]
            aggregated = EvaluationMetrics.aggregate_metrics(all_metrics)
            
            # Retrieval Quality
            report.append("\n🎯 Retrieval Quality:")
            for metric_name in ['precision@1', 'precision@3', 'precision@5',
                                'recall@1', 'recall@3', 'recall@5',
                                'f1@1', 'f1@3', 'f1@5']:
                if metric_name in aggregated:
                    stats = aggregated[metric_name]
                    report.append(
                        f"  {metric_name:20s}: {stats['mean']:.4f} "
                        f"(±{stats['std']:.4f}) "
                        f"[{stats['min']:.4f} - {stats['max']:.4f}]"
                    )
            
            # NDCG
            if 'ndcg@3' in aggregated or 'ndcg@5' in aggregated:
                report.append("\n📈 Ranking Quality (NDCG):")
                for metric_name in ['ndcg@3', 'ndcg@5']:
                    if metric_name in aggregated:
                        stats = aggregated[metric_name]
                        report.append(
                            f"  {metric_name:20s}: {stats['mean']:.4f} "
                            f"(±{stats['std']:.4f})"
                        )
            
            # Citation Accuracy
            if 'citation_accuracy' in aggregated:
                report.append("\n📝 Citation Quality:")
                stats = aggregated['citation_accuracy']
                report.append(
                    f"  Citation Accuracy    : {stats['mean']:.4f} "
                    f"(±{stats['std']:.4f})"
                )
            
            # Efficiency
            report.append("\n⚡ Efficiency:")
            total_latency = sum(r.latency_ms for r in result_list)
            avg_latency = total_latency / len(result_list) if result_list else 0
            report.append(f"  Avg Latency          : {avg_latency:.2f} ms")
            report.append(f"  Total Time           : {total_latency/1000:.2f} s")
            
            if 'context_size' in aggregated:
                stats = aggregated['context_size']
                report.append(
                    f"  Avg Context Size     : {stats['mean']:.0f} tokens "
                    f"(±{stats['std']:.0f})"
                )
            
            report.append("")
        
        if 'TreeRAG' in results and 'FlatRAG' in results:
            report.append("-" * 80)
            report.append("⚔️  TreeRAG vs FlatRAG Comparison")
            report.append("-" * 80)
            
            tree_metrics = EvaluationMetrics.aggregate_metrics(
                [r.metrics for r in results['TreeRAG']]
            )
            flat_metrics = EvaluationMetrics.aggregate_metrics(
                [r.metrics for r in results['FlatRAG']]
            )
            
            for metric_name in ['precision@3', 'recall@3', 'f1@3']:
                if metric_name in tree_metrics and metric_name in flat_metrics:
                    tree_val = tree_metrics[metric_name]['mean']
                    flat_val = flat_metrics[metric_name]['mean']
                    diff = tree_val - flat_val
                    symbol = "🟢" if diff > 0 else "🔴" if diff < 0 else "🟡"
                    report.append(
                        f"{symbol} {metric_name:20s}: TreeRAG={tree_val:.4f} vs "
                        f"FlatRAG={flat_val:.4f} (Δ={diff:+.4f})"
                    )
            
            if 'context_size' in tree_metrics and 'context_size' in flat_metrics:
                tree_ctx = tree_metrics['context_size']['mean']
                flat_ctx = flat_metrics['context_size']['mean']
                reduction = EvaluationMetrics.context_reduction_rate(flat_ctx, tree_ctx)
                report.append(
                    f"\n💡 Context Reduction  : {reduction*100:.1f}% "
                    f"({flat_ctx:.0f} → {tree_ctx:.0f} tokens)"
                )
            
            tree_latency = sum(r.latency_ms for r in results['TreeRAG']) / len(results['TreeRAG'])
            flat_latency = sum(r.latency_ms for r in results['FlatRAG']) / len(results['FlatRAG'])
            latency_comp = EvaluationMetrics.latency_comparison(tree_latency, flat_latency)
            
            report.append(
                f"⚡ Latency            : TreeRAG={tree_latency:.2f}ms vs "
                f"FlatRAG={flat_latency:.2f}ms "
                f"({latency_comp['speedup']:.2f}x {latency_comp['faster_system']})"
            )
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_comparison_table(
        self,
        results: Optional[Dict[str, List[BenchmarkResult]]] = None,
        output_format: str = 'markdown'
    ) -> str:
        """
        TreeRAG vs FlatRAG 비교 테이블 생성
        
        Args:
            results: 벤치마크 결과
            output_format: 'markdown' 또는 'latex'
            
        Returns:
            비교 테이블 (마크다운 또는 LaTeX 형식)
        """
        if results is None:
            results = self.results
        
        if 'TreeRAG' not in results or 'FlatRAG' not in results:
            return "Cannot generate comparison table without both TreeRAG and FlatRAG results."
        
        tree_metrics = EvaluationMetrics.aggregate_metrics(
            [r.metrics for r in results['TreeRAG']]
        )
        flat_metrics = EvaluationMetrics.aggregate_metrics(
            [r.metrics for r in results['FlatRAG']]
        )
        
        if output_format == 'markdown':
            return self._comparison_table_markdown(tree_metrics, flat_metrics, results)
        elif output_format == 'latex':
            return self._comparison_table_latex(tree_metrics, flat_metrics, results)
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    def _comparison_table_markdown(
        self,
        tree_metrics: Dict,
        flat_metrics: Dict,
        results: Dict
    ) -> str:
        """마크다운 비교 테이블 생성"""
        table = []
        table.append("| Metric | TreeRAG | FlatRAG | Improvement |")
        table.append("|--------|---------|---------|-------------|")
        
        for metric_name in ['precision@3', 'recall@3', 'f1@3', 'ndcg@3']:
            if metric_name in tree_metrics and metric_name in flat_metrics:
                tree_val = tree_metrics[metric_name]['mean']
                flat_val = flat_metrics[metric_name]['mean']
                improvement = ((tree_val - flat_val) / flat_val * 100) if flat_val > 0 else 0
                
                table.append(
                    f"| {metric_name} | {tree_val:.4f} | {flat_val:.4f} | "
                    f"{improvement:+.1f}% |"
                )
        
        if 'context_size' in tree_metrics and 'context_size' in flat_metrics:
            tree_ctx = tree_metrics['context_size']['mean']
            flat_ctx = flat_metrics['context_size']['mean']
            reduction = (1 - tree_ctx/flat_ctx) * 100 if flat_ctx > 0 else 0
            
            table.append(
                f"| Context Size (tokens) | {tree_ctx:.0f} | {flat_ctx:.0f} | "
                f"{reduction:.1f}% reduction |"
            )
        
        tree_latency = sum(r.latency_ms for r in results['TreeRAG']) / len(results['TreeRAG'])
        flat_latency = sum(r.latency_ms for r in results['FlatRAG']) / len(results['FlatRAG'])
        latency_improvement = ((flat_latency - tree_latency) / flat_latency * 100) if flat_latency > 0 else 0
        
        table.append(
            f"| Latency (ms) | {tree_latency:.2f} | {flat_latency:.2f} | "
            f"{latency_improvement:+.1f}% |"
        )
        
        return "\n".join(table)
    
    def _comparison_table_latex(
        self,
        tree_metrics: Dict,
        flat_metrics: Dict,
        results: Dict
    ) -> str:
        """LaTeX 비교 테이블 생성 (논문용)"""
        table = []
        table.append("\\begin{table}[h]")
        table.append("\\centering")
        table.append("\\caption{TreeRAG vs Flat RAG Performance Comparison}")
        table.append("\\begin{tabular}{lccc}")
        table.append("\\hline")
        table.append("Metric & TreeRAG & Flat RAG & Improvement \\\\")
        table.append("\\hline")
        
        for metric_name in ['precision@3', 'recall@3', 'f1@3']:
            if metric_name in tree_metrics and metric_name in flat_metrics:
                tree_val = tree_metrics[metric_name]['mean']
                flat_val = flat_metrics[metric_name]['mean']
                improvement = ((tree_val - flat_val) / flat_val * 100) if flat_val > 0 else 0
                
                name_display = metric_name.replace('@', '@')
                table.append(
                    f"{name_display} & {tree_val:.4f} & {flat_val:.4f} & "
                    f"{improvement:+.1f}\\% \\\\"
                )
        
        table.append("\\hline")
        table.append("\\end{tabular}")
        table.append("\\end{table}")
        
        return "\n".join(table)
