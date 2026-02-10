import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import math


@dataclass
class CompressedContext:
    original_count: int
    compressed_count: int
    contexts: List[Dict[str, Any]]
    merged_groups: List[List[str]]
    compression_ratio: float
    total_tokens_saved: int


@dataclass
class ContextChunk:
    node_id: str
    title: str
    content: str
    page_ref: Optional[str]
    relevance_score: float
    token_count: int


class ContextualCompressor:
    SIMILARITY_THRESHOLD = 0.7
    MAX_OUTPUT_TOKENS = 4000
    MIN_CHUNK_RELEVANCE = 0.2
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_output_tokens: int = 4000
    ):
        self.similarity_threshold = similarity_threshold
        self.max_output_tokens = max_output_tokens
    
    def compress(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> CompressedContext:
        if not contexts:
            return CompressedContext(
                original_count=0,
                compressed_count=0,
                contexts=[],
                merged_groups=[],
                compression_ratio=1.0,
                total_tokens_saved=0
            )
        
        chunks = self._to_chunks(contexts)
        original_tokens = sum(c.token_count for c in chunks)
        
        scored_chunks = self._score_relevance(chunks, query)
        
        filtered = [c for c in scored_chunks if c.relevance_score >= self.MIN_CHUNK_RELEVANCE]
        if not filtered:
            filtered = sorted(scored_chunks, key=lambda x: -x.relevance_score)[:3]
        
        merged, groups = self._merge_similar(filtered)
        
        final = self._truncate_to_limit(merged)
        
        final_tokens = sum(c.token_count for c in final)
        
        return CompressedContext(
            original_count=len(contexts),
            compressed_count=len(final),
            contexts=[self._chunk_to_dict(c) for c in final],
            merged_groups=groups,
            compression_ratio=len(final) / len(contexts) if contexts else 1.0,
            total_tokens_saved=original_tokens - final_tokens
        )
    
    def _to_chunks(self, contexts: List[Dict[str, Any]]) -> List[ContextChunk]:
        chunks = []
        for ctx in contexts:
            content = self._extract_content(ctx)
            chunks.append(ContextChunk(
                node_id=ctx.get("id", ""),
                title=ctx.get("title", ""),
                content=content,
                page_ref=ctx.get("page_ref"),
                relevance_score=0.0,
                token_count=self._estimate_tokens(content)
            ))
        return chunks
    
    def _extract_content(self, ctx: Dict[str, Any]) -> str:
        parts = []
        if ctx.get("title"):
            parts.append(ctx["title"])
        if ctx.get("summary"):
            parts.append(ctx["summary"])
        if ctx.get("content"):
            parts.append(ctx["content"])
        return " ".join(parts)
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 3
    
    def _score_relevance(
        self, 
        chunks: List[ContextChunk], 
        query: str
    ) -> List[ContextChunk]:
        query_terms = self._tokenize(query)
        query_vec = self._term_frequency(query_terms)
        
        for chunk in chunks:
            chunk_terms = self._tokenize(chunk.content)
            chunk_vec = self._term_frequency(chunk_terms)
            
            tfidf_score = self._cosine_similarity(query_vec, chunk_vec)
            
            keyword_overlap = self._keyword_overlap(query_terms, chunk_terms)
            
            title_match = 1.0 if any(
                term in chunk.title.lower() for term in query_terms
            ) else 0.0
            
            chunk.relevance_score = (
                0.5 * tfidf_score +
                0.3 * keyword_overlap +
                0.2 * title_match
            )
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        tokens = text.split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     '은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]
    
    def _term_frequency(self, tokens: List[str]) -> Dict[str, float]:
        counter = Counter(tokens)
        total = sum(counter.values()) or 1
        return {term: count / total for term, count in counter.items()}
    
    def _cosine_similarity(
        self, 
        vec1: Dict[str, float], 
        vec2: Dict[str, float]
    ) -> float:
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values())) or 1
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values())) or 1
        
        return dot_product / (norm1 * norm2)
    
    def _keyword_overlap(
        self, 
        query_terms: List[str], 
        chunk_terms: List[str]
    ) -> float:
        query_set = set(query_terms)
        chunk_set = set(chunk_terms)
        
        if not query_set:
            return 0.0
        
        return len(query_set & chunk_set) / len(query_set)
    
    def _merge_similar(
        self, 
        chunks: List[ContextChunk]
    ) -> Tuple[List[ContextChunk], List[List[str]]]:
        if len(chunks) <= 1:
            return chunks, []
        
        n = len(chunks)
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._content_similarity(chunks[i], chunks[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
        
        merged_indices: set = set()
        merge_groups: List[List[str]] = []
        result: List[ContextChunk] = []
        
        for i in range(n):
            if i in merged_indices:
                continue
            
            similar_group = [i]
            for j in range(i + 1, n):
                if j not in merged_indices and similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_group.append(j)
                    merged_indices.add(j)
            
            if len(similar_group) > 1:
                merged = self._merge_chunks([chunks[idx] for idx in similar_group])
                result.append(merged)
                merge_groups.append([chunks[idx].node_id for idx in similar_group])
            else:
                result.append(chunks[i])
        
        return result, merge_groups
    
    def _content_similarity(
        self, 
        chunk1: ContextChunk, 
        chunk2: ContextChunk
    ) -> float:
        tokens1 = set(self._tokenize(chunk1.content))
        tokens2 = set(self._tokenize(chunk2.content))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_chunks(self, chunks: List[ContextChunk]) -> ContextChunk:
        best = max(chunks, key=lambda c: c.relevance_score)
        
        all_content_parts = []
        for c in chunks:
            if c.content and c.content not in all_content_parts:
                all_content_parts.append(c.content)
        
        merged_content = " | ".join(all_content_parts[:3])
        
        page_refs = [c.page_ref for c in chunks if c.page_ref]
        merged_page_ref = ", ".join(sorted(set(page_refs))) if page_refs else None
        
        return ContextChunk(
            node_id=best.node_id + "_merged",
            title=best.title,
            content=merged_content,
            page_ref=merged_page_ref,
            relevance_score=max(c.relevance_score for c in chunks),
            token_count=self._estimate_tokens(merged_content)
        )
    
    def _truncate_to_limit(
        self, 
        chunks: List[ContextChunk]
    ) -> List[ContextChunk]:
        sorted_chunks = sorted(chunks, key=lambda c: -c.relevance_score)
        
        result = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if total_tokens + chunk.token_count <= self.max_output_tokens:
                result.append(chunk)
                total_tokens += chunk.token_count
            elif total_tokens == 0:
                result.append(chunk)
                break
        
        return result
    
    def _chunk_to_dict(self, chunk: ContextChunk) -> Dict[str, Any]:
        return {
            "id": chunk.node_id,
            "title": chunk.title,
            "summary": chunk.content,
            "page_ref": chunk.page_ref,
            "relevance_score": chunk.relevance_score
        }


def format_compressed_context(result: CompressedContext) -> str:
    lines = []
    
    for ctx in result.contexts:
        title = ctx.get("title", "Unknown")
        summary = ctx.get("summary", "")
        page_ref = ctx.get("page_ref", "")
        score = ctx.get("relevance_score", 0)
        
        lines.append(f"### {title}")
        if page_ref:
            lines.append(f"(Page: {page_ref})")
        lines.append(f"[Relevance: {score:.2f}]")
        lines.append(summary)
        lines.append("")
    
    return "\n".join(lines)
