#!/usr/bin/env python3
import asyncio
import json
import os
from typing import List, Dict
from src.core.reasoner import RegulatoryReasoner


def load_dataset(path: str) -> List[Dict]:
    """평가 데이터셋 로드"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def calculate_score(response: str, expected_answer: str) -> float:
    """답변 품질 점수 계산 (0.0 ~ 1.0)"""
    score = 0.0
    
    # 1. 예상 답변 포함 여부 (40%)
    if expected_answer.lower() in response.lower():
        score += 0.4
    
    # 2. 인용 존재 여부 (30%)
    if "[" in response and "p." in response:
        score += 0.3
    
    # 3. 출처 요약 존재 여부 (20%)
    if "📚" in response or "참조 페이지" in response:
        score += 0.2
    
    # 4. 충분한 설명 (10%)
    if len(response) > 100:
        score += 0.1
    
    return score


async def evaluate_prompt(dataset: List[Dict]) -> Dict:
    """현재 프롬프트 성능 평가"""
    print("🔍 프롬프트 성능 평가 시작...\n")
    
    scores = []
    details = []
    
    for i, task in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] 질문: {task['question']}")
        
        try:
            index_filenames = task["index_filename"].split(",")
            reasoner = RegulatoryReasoner(index_filenames)
            
            response = reasoner.query(task["question"])
            score = calculate_score(response, task["expected_answer"])
            
            scores.append(score)
            details.append({
                "question": task["question"],
                "expected": task["expected_answer"],
                "response": response[:200] + "..." if len(response) > 200 else response,
                "score": score
            })
            
            print(f"  점수: {score:.2f}")
            print(f"  답변: {response[:100]}...\n")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}\n")
            scores.append(0.0)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "average_score": avg_score,
        "total_questions": len(dataset),
        "scores": scores,
        "details": details
    }


def main():
    """메인 실행 함수"""
    eval_path = "data/eval_dataset.jsonl"
    
    if not os.path.exists(eval_path):
        print(f"❌ {eval_path} 파일을 찾을 수 없습니다.")
        return
    
    dataset = load_dataset(eval_path)
    print(f"✅ {len(dataset)}개 평가 샘플 로드됨\n")
    
    results = asyncio.run(evaluate_prompt(dataset))
    
    print("\n" + "="*60)
    print("📊 평가 결과 요약")
    print("="*60)
    print(f"평균 점수: {results['average_score']:.2f} / 1.00")
    print(f"평가 문제 수: {results['total_questions']}개")
    print(f"개별 점수: {[f'{s:.2f}' for s in results['scores']]}")
    print("="*60 + "\n")
    
    # 결과 저장
    output_path = "data/evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 평가 결과 저장: {output_path}")
    
    if results['average_score'] >= 0.7:
        print("✅ 프롬프트 품질: 우수")
    elif results['average_score'] >= 0.5:
        print("⚠️ 프롬프트 품질: 보통 (개선 필요)")
    else:
        print("❌ 프롬프트 품질: 낮음 (최적화 필수)")


if __name__ == "__main__":
    main()
