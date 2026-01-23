#!/usr/bin/env python3
import asyncio
import json
import os
from typing import List, Dict
from src.apo_agent import regulatory_agent, QueryTask, SYSTEM_PROMPT, COMPARISON_SECTION
import agentlightning as agl


def load_dataset(path: str) -> List[QueryTask]:
    """평가 데이터셋 로드"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


async def main():
    print("🚀 APO Training 시작...")
    
    eval_path = "data/eval_dataset.jsonl"
    if not os.path.exists(eval_path):
        print(f"❌ {eval_path} 파일을 찾을 수 없습니다.")
        return
    
    dataset = load_dataset(eval_path)
    print(f"✅ {len(dataset)}개 평가 샘플 로드됨\n")
    
    initial_prompts = {
        "system_prompt": SYSTEM_PROMPT,
        "comparison_section": COMPARISON_SECTION,
    }
    
    print("📋 초기 프롬프트:")
    print(f"  - system_prompt: {len(SYSTEM_PROMPT.template)} chars")
    print(f"  - comparison_section: {len(COMPARISON_SECTION.template)} chars\n")
    
    llm = agl.LLM(
        endpoint=os.getenv("OPENAI_BASE_URL", "http://localhost:8000"),
        model="gemini-2.5-flash",
        sampling_parameters={"temperature": 0.0},
    )
    
    apo_config = {
        "max_iterations": 5,
        "beam_size": 3,
        "batch_size": 2,
        "learning_rate": 0.1,
    }
    
    print("⚙️ APO 설정:")
    for key, value in apo_config.items():
        print(f"  - {key}: {value}")
    print()
    
    trainer = agl.Trainer(
        n_workers=1,
        initial_resources={"main_llm": llm},
    )
    
    apo_algorithm = agl.algorithm.APO(
        prompts=initial_prompts,
        max_iterations=apo_config["max_iterations"],
        beam_size=apo_config["beam_size"],
    )
    
    print("🎯 APO 학습 시작...\n")
    
    try:
        await trainer.fit(
            algorithm=apo_algorithm,
            rollout=regulatory_agent,
            dataset=dataset,
        )
        
        print("\n✅ APO 학습 완료!")
        
        optimized_prompts = apo_algorithm.get_best_prompts()
        
        output_dir = "data/optimized_prompts"
        os.makedirs(output_dir, exist_ok=True)
        
        for name, prompt in optimized_prompts.items():
            output_path = os.path.join(output_dir, f"{name}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt.template)
            print(f"💾 최적화된 프롬프트 저장: {output_path}")
        
        print("\n📊 최적화 결과:")
        print(f"  초기 평균 reward: {apo_algorithm.initial_reward:.3f}")
        print(f"  최종 평균 reward: {apo_algorithm.final_reward:.3f}")
        print(f"  개선율: {(apo_algorithm.final_reward - apo_algorithm.initial_reward) * 100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
