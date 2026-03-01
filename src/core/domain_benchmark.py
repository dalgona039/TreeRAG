"""
Multi-domain Benchmark Module for TreeRAG.

This module provides evaluation and benchmarking capabilities
across different document domains (medical, legal, technical, etc.).

Key Features:
- Domain classification and detection
- Domain-specific evaluation metrics
- Benchmark dataset management
- Cross-domain performance comparison
"""

import json
import os
import time
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from src.config import Config


class DocumentDomain(Enum):
    """Supported document domains for benchmarking."""
    MEDICAL = "medical"
    LEGAL = "legal"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    GENERAL = "general"
    
    @classmethod
    def from_string(cls, s: str) -> "DocumentDomain":
        for member in cls:
            if member.value == s.lower():
                return member
        return cls.GENERAL


@dataclass
class DomainMetrics:
    """Metrics specific to a domain."""
    domain: DocumentDomain
    terminology_coverage: float = 0.0  # % of domain terms correctly identified
    structure_compliance: float = 0.0  # % compliance with domain structure
    citation_accuracy: float = 0.0      # Accuracy of reference handling
    precision: float = 0.0              # Answer precision
    recall: float = 0.0                 # Answer recall
    f1_score: float = 0.0               # F1 score
    response_time_ms: float = 0.0       # Average response time
    hallucination_rate: float = 0.0     # Rate of hallucinated content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "terminology_coverage": round(self.terminology_coverage, 4),
            "structure_compliance": round(self.structure_compliance, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "response_time_ms": round(self.response_time_ms, 2),
            "hallucination_rate": round(self.hallucination_rate, 4)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainMetrics":
        return cls(
            domain=DocumentDomain.from_string(data.get("domain", "general")),
            terminology_coverage=data.get("terminology_coverage", 0.0),
            structure_compliance=data.get("structure_compliance", 0.0),
            citation_accuracy=data.get("citation_accuracy", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            response_time_ms=data.get("response_time_ms", 0.0),
            hallucination_rate=data.get("hallucination_rate", 0.0)
        )


@dataclass
class BenchmarkQuestion:
    """A single benchmark question with expected answer."""
    id: str
    question: str
    expected_answer: str
    domain: DocumentDomain
    difficulty: str = "medium"  # easy, medium, hard
    requires_reasoning: bool = False
    expected_sections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "domain": self.domain.value,
            "difficulty": self.difficulty,
            "requires_reasoning": self.requires_reasoning,
            "expected_sections": self.expected_sections,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuestion":
        return cls(
            id=data["id"],
            question=data["question"],
            expected_answer=data["expected_answer"],
            domain=DocumentDomain.from_string(data.get("domain", "general")),
            difficulty=data.get("difficulty", "medium"),
            requires_reasoning=data.get("requires_reasoning", False),
            expected_sections=data.get("expected_sections", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark evaluation."""
    question_id: str
    actual_answer: str
    is_correct: bool
    partial_score: float  # 0.0 to 1.0
    sections_found: List[str]
    section_recall: float
    response_time_ms: float
    has_hallucination: bool
    reasoning_path_found: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "actual_answer": self.actual_answer[:500],  # Truncate for storage
            "is_correct": self.is_correct,
            "partial_score": round(self.partial_score, 4),
            "sections_found": self.sections_found,
            "section_recall": round(self.section_recall, 4),
            "response_time_ms": round(self.response_time_ms, 2),
            "has_hallucination": self.has_hallucination,
            "reasoning_path_found": self.reasoning_path_found,
            "error": self.error
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a domain."""
    domain: DocumentDomain
    document_name: str
    total_questions: int
    correct_count: int
    partial_score_avg: float
    section_recall_avg: float
    response_time_avg_ms: float
    hallucination_rate: float
    reasoning_success_rate: float
    results: List[BenchmarkResult]
    run_timestamp: str = ""
    
    def __post_init__(self):
        if not self.run_timestamp:
            self.run_timestamp = datetime.now().isoformat()
    
    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_questions if self.total_questions > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain.value,
            "document_name": self.document_name,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "partial_score_avg": round(self.partial_score_avg, 4),
            "section_recall_avg": round(self.section_recall_avg, 4),
            "response_time_avg_ms": round(self.response_time_avg_ms, 2),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "reasoning_success_rate": round(self.reasoning_success_rate, 4),
            "run_timestamp": self.run_timestamp,
            "results": [r.to_dict() for r in self.results]
        }


class DomainClassifier:
    """Classifies documents into domains based on content analysis."""
    
    # Domain-specific keyword patterns
    DOMAIN_KEYWORDS = {
        DocumentDomain.MEDICAL: [
            "진단", "치료", "환자", "증상", "질병", "약물", "수술", "혈액",
            "diagnosis", "treatment", "patient", "symptom", "disease", "drug",
            "의료", "임상", "병원", "의사", "간호", "검사", "처방", "예후",
            "SNOMED", "ICD", "의공학", "생체", "biomedical", "clinical"
        ],
        DocumentDomain.LEGAL: [
            "법률", "조항", "규정", "계약", "소송", "판결", "법원", "변호사",
            "legal", "law", "regulation", "contract", "lawsuit", "court",
            "조례", "헌법", "민법", "형법", "상법", "특허", "저작권"
        ],
        DocumentDomain.TECHNICAL: [
            "시스템", "알고리즘", "프로그램", "데이터", "네트워크", "서버",
            "system", "algorithm", "program", "data", "network", "server",
            "API", "프로토콜", "아키텍처", "모듈", "인터페이스", "구현",
            "반도체", "회로", "전자", "hardware", "software"
        ],
        DocumentDomain.ACADEMIC: [
            "연구", "논문", "실험", "결과", "가설", "분석", "방법론",
            "research", "paper", "experiment", "result", "hypothesis",
            "학술", "학위", "교육과정", "학점", "교과목", "수업"
        ],
        DocumentDomain.FINANCIAL: [
            "재무", "회계", "투자", "수익", "비용", "자산", "부채",
            "financial", "accounting", "investment", "revenue", "cost",
            "주식", "채권", "펀드", "금리", "환율", "세금",
            "재무제표", "손익", "현금흐름", "대차대조표", "포트폴리오"
        ],
        DocumentDomain.REGULATORY: [
            "규제", "인증", "표준", "준수", "감사", "검토", "승인",
            "regulatory", "certification", "standard", "compliance",
            "ISO", "FDA", "CE", "인허가", "심사", "요건"
        ]
    }
    
    @classmethod
    def classify(cls, text: str, title: str = "") -> Tuple[DocumentDomain, float]:
        """
        Classify document domain based on content.
        
        Args:
            text: Document text content
            title: Document title
        
        Returns:
            Tuple of (domain, confidence)
        """
        combined_text = f"{title} {text}".lower()
        
        domain_scores: Dict[DocumentDomain, int] = defaultdict(int)
        
        for domain, keywords in cls.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    domain_scores[domain] += 1
        
        if not domain_scores:
            return DocumentDomain.GENERAL, 0.5
        
        # Find domain with highest score
        best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
        total_keywords = sum(domain_scores.values())
        best_score = domain_scores[best_domain]
        
        confidence = best_score / total_keywords if total_keywords > 0 else 0.5
        
        # Require minimum confidence
        if confidence < 0.3:
            return DocumentDomain.GENERAL, confidence
        
        return best_domain, min(confidence + 0.3, 1.0)  # Boost confidence
    
    @classmethod
    def classify_with_llm(cls, text: str, title: str = "") -> Tuple[DocumentDomain, float]:
        """Use LLM for more accurate domain classification."""
        prompt = f"""Classify the following document into one of these domains:
- medical: Healthcare, clinical, biomedical
- legal: Laws, regulations, contracts
- technical: Software, hardware, engineering
- academic: Research papers, education
- financial: Finance, accounting, investment
- regulatory: Standards, compliance, certification
- general: None of the above

Document Title: {title}
Document Excerpt: {text[:2000]}

Respond in JSON format:
{{
  "domain": "<domain_name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

JSON only:"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                return cls.classify(text, title)
            
            result = json.loads(response.text)
            domain = DocumentDomain.from_string(result.get("domain", "general"))
            confidence = result.get("confidence", 0.5)
            
            return domain, confidence
            
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return cls.classify(text, title)


class AnswerEvaluator:
    """Evaluates answer quality against expected answers."""
    
    @staticmethod
    def compute_similarity(actual: str, expected: str) -> float:
        """Compute text similarity using ngram overlap."""
        if not actual or not expected:
            return 0.0
        
        def get_ngrams(text: str, n: int = 3) -> set:
            text = text.lower()
            return {text[i:i+n] for i in range(len(text) - n + 1)}
        
        actual_ngrams = get_ngrams(actual)
        expected_ngrams = get_ngrams(expected)
        
        if not actual_ngrams or not expected_ngrams:
            return 0.0
        
        intersection = len(actual_ngrams & expected_ngrams)
        union = len(actual_ngrams | expected_ngrams)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_keyword_recall(actual: str, expected_keywords: List[str]) -> float:
        """Compute keyword recall from expected keywords."""
        if not expected_keywords:
            return 1.0
        
        actual_lower = actual.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in actual_lower)
        
        return found / len(expected_keywords)
    
    @classmethod
    def evaluate_with_llm(
        cls, 
        question: str,
        actual: str, 
        expected: str
    ) -> Tuple[bool, float, str]:
        """
        Use LLM to evaluate answer correctness.
        
        Returns:
            Tuple of (is_correct, partial_score, explanation)
        """
        prompt = f"""Evaluate if the actual answer correctly addresses the question compared to the expected answer.

Question: {question}

Expected Answer: {expected}

Actual Answer: {actual[:2000]}

Evaluate on these criteria:
1. Factual correctness: Does the actual answer contain correct information?
2. Completeness: Does it cover the key points of the expected answer?
3. Relevance: Is the information relevant to the question?

Respond in JSON format:
{{
  "is_correct": true/false,
  "partial_score": <0.0-1.0>,
  "explanation": "<brief explanation>"
}}

JSON only:"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                # Fallback to similarity
                similarity = cls.compute_similarity(actual, expected)
                return similarity >= 0.7, similarity, "LLM evaluation failed"
            
            result = json.loads(response.text)
            return (
                result.get("is_correct", False),
                result.get("partial_score", 0.0),
                result.get("explanation", "")
            )
            
        except Exception as e:
            similarity = cls.compute_similarity(actual, expected)
            return similarity >= 0.7, similarity, f"Fallback: {str(e)}"


class BenchmarkDataset:
    """Manages benchmark question datasets for different domains."""
    
    def __init__(self, dataset_dir: Optional[str] = None):
        self.dataset_dir = dataset_dir or os.path.join(
            Config.DATA_DIR, "benchmarks"
        )
        self.questions: Dict[str, List[BenchmarkQuestion]] = defaultdict(list)
    
    def load_dataset(self, domain: DocumentDomain) -> List[BenchmarkQuestion]:
        """Load benchmark questions for a specific domain."""
        if domain.value in self.questions:
            return self.questions[domain.value]
        
        dataset_path = os.path.join(
            self.dataset_dir, f"{domain.value}_benchmark.json"
        )
        
        if not os.path.exists(dataset_path):
            print(f"No benchmark dataset found for {domain.value}")
            return []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            questions = [
                BenchmarkQuestion.from_dict(q) 
                for q in data.get("questions", [])
            ]
            self.questions[domain.value] = questions
            return questions
            
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return []
    
    def save_dataset(
        self, 
        domain: DocumentDomain, 
        questions: List[BenchmarkQuestion]
    ) -> bool:
        """Save benchmark questions to file."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        dataset_path = os.path.join(
            self.dataset_dir, f"{domain.value}_benchmark.json"
        )
        
        try:
            data = {
                "domain": domain.value,
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "question_count": len(questions),
                "questions": [q.to_dict() for q in questions]
            }
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.questions[domain.value] = questions
            return True
            
        except Exception as e:
            print(f"Failed to save dataset: {e}")
            return False
    
    def add_question(
        self,
        domain: DocumentDomain,
        question: str,
        expected_answer: str,
        difficulty: str = "medium",
        **kwargs
    ) -> BenchmarkQuestion:
        """Add a new benchmark question."""
        q_id = hashlib.md5(f"{domain.value}:{question}".encode()).hexdigest()[:12]
        
        new_question = BenchmarkQuestion(
            id=q_id,
            question=question,
            expected_answer=expected_answer,
            domain=domain,
            difficulty=difficulty,
            **kwargs
        )
        
        self.questions[domain.value].append(new_question)
        return new_question
    
    def get_all_domains(self) -> List[DocumentDomain]:
        """Get list of domains with loaded questions."""
        return [
            DocumentDomain.from_string(d) 
            for d in self.questions.keys()
        ]


class DomainBenchmark:
    """
    Main benchmarking engine for multi-domain evaluation.
    
    Runs benchmark questions against TreeRAG and evaluates
    performance across different document domains.
    """
    
    def __init__(self, dataset: Optional[BenchmarkDataset] = None):
        self.dataset = dataset or BenchmarkDataset()
        self.results: Dict[str, List[BenchmarkReport]] = defaultdict(list)
    
    def run_benchmark(
        self,
        document_name: str,
        domain: DocumentDomain,
        questions: Optional[List[BenchmarkQuestion]] = None,
        use_reasoning: bool = False
    ) -> BenchmarkReport:
        """
        Run benchmark for a document.
        
        Args:
            document_name: Name of the indexed document
            domain: Document domain
            questions: Optional custom questions (uses dataset if None)
            use_reasoning: Whether to use reasoning graph
        
        Returns:
            BenchmarkReport with all results
        """
        from src.core.reasoner import TreeRAGReasoner
        from src.utils.hallucination_detector import HallucinationDetector
        
        if questions is None:
            questions = self.dataset.load_dataset(domain)
        
        if not questions:
            # Create sample questions if none exist
            questions = self._generate_sample_questions(document_name, domain)
        
        results: List[BenchmarkResult] = []
        hallucination_detector = HallucinationDetector()
        
        for q in questions:
            result = self._evaluate_question(
                document_name=document_name,
                question=q,
                hallucination_detector=hallucination_detector,
                use_reasoning=use_reasoning
            )
            results.append(result)
        
        # Compute aggregate metrics
        correct_count = sum(1 for r in results if r.is_correct)
        partial_avg = sum(r.partial_score for r in results) / len(results) if results else 0
        recall_avg = sum(r.section_recall for r in results) / len(results) if results else 0
        time_avg = sum(r.response_time_ms for r in results) / len(results) if results else 0
        hallucination_rate = sum(1 for r in results if r.has_hallucination) / len(results) if results else 0
        reasoning_rate = sum(1 for r in results if r.reasoning_path_found) / len(results) if results else 0
        
        report = BenchmarkReport(
            domain=domain,
            document_name=document_name,
            total_questions=len(questions),
            correct_count=correct_count,
            partial_score_avg=partial_avg,
            section_recall_avg=recall_avg,
            response_time_avg_ms=time_avg,
            hallucination_rate=hallucination_rate,
            reasoning_success_rate=reasoning_rate,
            results=results
        )
        
        self.results[document_name].append(report)
        return report
    
    def _evaluate_question(
        self,
        document_name: str,
        question: BenchmarkQuestion,
        hallucination_detector,
        use_reasoning: bool
    ) -> BenchmarkResult:
        """Evaluate a single benchmark question."""
        from src.core.reasoner import TreeRAGReasoner
        
        start_time = time.time()
        
        try:
            # Get answer from TreeRAG
            reasoner = TreeRAGReasoner()
            response = reasoner.answer_question(
                question=question.question,
                document_names=[document_name],
                max_depth=5,
                max_branches=3
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            actual_answer = response.get("answer", "")
            sections_found = [
                s.get("title", "") 
                for s in response.get("sources", [])
            ]
            
            # Check for hallucinations
            has_hallucination = hallucination_detector.detect(
                question=question.question,
                answer=actual_answer,
                sources=[s.get("summary", "") for s in response.get("sources", [])]
            ).get("has_hallucination", False)
            
            # Evaluate answer
            is_correct, partial_score, _ = AnswerEvaluator.evaluate_with_llm(
                question=question.question,
                actual=actual_answer,
                expected=question.expected_answer
            )
            
            # Compute section recall
            if question.expected_sections:
                found_set = set(s.lower() for s in sections_found)
                expected_set = set(s.lower() for s in question.expected_sections)
                section_recall = len(found_set & expected_set) / len(expected_set)
            else:
                section_recall = 1.0 if sections_found else 0.0
            
            return BenchmarkResult(
                question_id=question.id,
                actual_answer=actual_answer,
                is_correct=is_correct,
                partial_score=partial_score,
                sections_found=sections_found,
                section_recall=section_recall,
                response_time_ms=elapsed_ms,
                has_hallucination=has_hallucination,
                reasoning_path_found=use_reasoning and len(sections_found) > 1
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return BenchmarkResult(
                question_id=question.id,
                actual_answer="",
                is_correct=False,
                partial_score=0.0,
                sections_found=[],
                section_recall=0.0,
                response_time_ms=elapsed_ms,
                has_hallucination=False,
                error=str(e)
            )
    
    def _generate_sample_questions(
        self,
        document_name: str,
        domain: DocumentDomain
    ) -> List[BenchmarkQuestion]:
        """Generate sample benchmark questions for a document."""
        # Load document tree
        index_path = os.path.join(
            Config.INDEX_DIR, f"{document_name}_index.json"
        )
        
        if not os.path.exists(index_path):
            return []
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                tree = json.load(f)
            
            return self._generate_questions_from_tree(tree, domain)
            
        except Exception as e:
            print(f"Failed to generate questions: {e}")
            return []
    
    def _generate_questions_from_tree(
        self,
        tree: Dict[str, Any],
        domain: DocumentDomain
    ) -> List[BenchmarkQuestion]:
        """Generate questions from document structure."""
        prompt = f"""Generate 5 benchmark questions for evaluating a RAG system on this {domain.value} document.

Document Structure:
{self._format_tree_structure(tree, max_depth=3)}

For each question, provide:
1. A question that requires finding specific information
2. The expected answer (based on the document structure)
3. Difficulty level (easy/medium/hard)
4. Whether it requires multi-section reasoning

JSON format:
{{
  "questions": [
    {{
      "question": "...",
      "expected_answer": "...",
      "difficulty": "easy|medium|hard",
      "requires_reasoning": true/false,
      "expected_sections": ["section_title_1", "section_title_2"]
    }}
  ]
}}

JSON only:"""

        try:
            response = Config.CLIENT.models.generate_content(
                model=Config.MODEL_NAME,
                contents=prompt,
                config=Config.get_generation_config(response_mime_type="application/json")
            )
            
            if not response.text:
                return []
            
            result = json.loads(response.text)
            questions = []
            
            for i, q in enumerate(result.get("questions", [])):
                questions.append(BenchmarkQuestion(
                    id=f"gen_{domain.value}_{i}",
                    question=q.get("question", ""),
                    expected_answer=q.get("expected_answer", ""),
                    domain=domain,
                    difficulty=q.get("difficulty", "medium"),
                    requires_reasoning=q.get("requires_reasoning", False),
                    expected_sections=q.get("expected_sections", [])
                ))
            
            return questions
            
        except Exception as e:
            print(f"Question generation failed: {e}")
            return []
    
    def _format_tree_structure(
        self, 
        node: Dict[str, Any], 
        depth: int = 0,
        max_depth: int = 3
    ) -> str:
        """Format tree structure for prompt."""
        if depth >= max_depth:
            return ""
        
        indent = "  " * depth
        title = node.get("title", "Untitled")
        summary = node.get("summary", "")[:100]
        
        result = f"{indent}- {title}: {summary}\n"
        
        children = node.get("children", [])
        for child in children[:5]:  # Limit children
            result += self._format_tree_structure(child, depth + 1, max_depth)
        
        return result
    
    def compare_domains(
        self,
        document_name: str
    ) -> Dict[str, Any]:
        """
        Compare performance across all domains for a document.
        
        Returns:
            Comparison summary with rankings
        """
        if document_name not in self.results:
            return {"error": "No benchmark results found for this document"}
        
        reports = self.results[document_name]
        
        comparison = {
            "document_name": document_name,
            "domains_evaluated": len(reports),
            "domain_metrics": [],
            "rankings": {
                "by_accuracy": [],
                "by_response_time": [],
                "by_hallucination_rate": []
            }
        }
        
        for report in reports:
            comparison["domain_metrics"].append({
                "domain": report.domain.value,
                "accuracy": report.accuracy,
                "partial_score_avg": report.partial_score_avg,
                "response_time_avg_ms": report.response_time_avg_ms,
                "hallucination_rate": report.hallucination_rate
            })
        
        # Create rankings
        sorted_by_accuracy = sorted(
            comparison["domain_metrics"],
            key=lambda x: x["accuracy"],
            reverse=True
        )
        comparison["rankings"]["by_accuracy"] = [
            {"rank": i+1, "domain": m["domain"], "accuracy": m["accuracy"]}
            for i, m in enumerate(sorted_by_accuracy)
        ]
        
        sorted_by_time = sorted(
            comparison["domain_metrics"],
            key=lambda x: x["response_time_avg_ms"]
        )
        comparison["rankings"]["by_response_time"] = [
            {"rank": i+1, "domain": m["domain"], "time_ms": m["response_time_avg_ms"]}
            for i, m in enumerate(sorted_by_time)
        ]
        
        sorted_by_hallucination = sorted(
            comparison["domain_metrics"],
            key=lambda x: x["hallucination_rate"]
        )
        comparison["rankings"]["by_hallucination_rate"] = [
            {"rank": i+1, "domain": m["domain"], "rate": m["hallucination_rate"]}
            for i, m in enumerate(sorted_by_hallucination)
        ]
        
        return comparison
    
    def save_report(
        self,
        report: BenchmarkReport,
        output_dir: Optional[str] = None
    ) -> str:
        """Save benchmark report to file."""
        output_dir = output_dir or os.path.join(Config.DATA_DIR, "benchmark_reports")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{report.document_name}_{report.domain.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_historical_reports(
        self,
        document_name: Optional[str] = None,
        domain: Optional[DocumentDomain] = None
    ) -> List[BenchmarkReport]:
        """Load historical benchmark reports from storage."""
        reports_dir = os.path.join(Config.DATA_DIR, "benchmark_reports")
        
        if not os.path.exists(reports_dir):
            return []
        
        reports = []
        
        for filename in os.listdir(reports_dir):
            if not filename.endswith('.json'):
                continue
            
            # Filter by document name if specified
            if document_name and not filename.startswith(document_name):
                continue
            
            # Filter by domain if specified
            if domain and domain.value not in filename:
                continue
            
            try:
                filepath = os.path.join(reports_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                report = BenchmarkReport(
                    domain=DocumentDomain.from_string(data.get("domain", "")),
                    document_name=data.get("document_name", ""),
                    total_questions=data.get("total_questions", 0),
                    correct_count=data.get("correct_count", 0),
                    partial_score_avg=data.get("partial_score_avg", 0),
                    section_recall_avg=data.get("section_recall_avg", 0),
                    response_time_avg_ms=data.get("response_time_avg_ms", 0),
                    hallucination_rate=data.get("hallucination_rate", 0),
                    reasoning_success_rate=data.get("reasoning_success_rate", 0),
                    results=[],  # Don't load full results for listing
                    run_timestamp=data.get("run_timestamp", "")
                )
                reports.append(report)
                
            except Exception as e:
                print(f"Failed to load report {filename}: {e}")
        
        return sorted(reports, key=lambda r: r.run_timestamp, reverse=True)
