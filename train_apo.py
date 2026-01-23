#!/usr/bin/env python3
import asyncio
import json
import os
from typing import List
from openai import AsyncOpenAI
from src.apo_agent import regulatory_agent, QueryTask, SYSTEM_PROMPT_TEXT
import agentlightning as agl
from dotenv import load_dotenv

load_dotenv()


def load_dataset(eval_path: str) -> List[QueryTask]:
    """평가 데이터셋 로드"""
    tasks = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


async def main():
    print("🚀 APO Training 시작 (Agent Lightning 0.3.0)...")
    
    eval_path = "data/eval_dataset.jsonl"
    if not os.path.exists(eval_path):
        print(f"❌ {eval_path} 파일을 찾을 수 없습니다.")
        return
    
    dataset = load_dataset(eval_path)
    print(f"✅ {len(dataset)}개 평가 샘플 로드됨\n")
    
    # 데이터셋을 train/val로 분할 (3:2 비율)
    split_idx = int(len(dataset) * 0.6)
    train_dataset = dataset[:split_idx]
    val_dataset = dataset[split_idx:]
    print(f"📊 데이터 분할:")
    print(f"  - Train: {len(train_dataset)}개")
    print(f"  - Validation: {len(val_dataset)}개\n")
    
    # 초기 프롬프트 템플릿
    seed_prompt = agl.PromptTemplate(
        template=SYSTEM_PROMPT_TEXT,
        engine="f-string"
    )
    
    print("📋 초기 프롬프트:")
    print(f"  - 길이: {len(SYSTEM_PROMPT_TEXT)} chars")
    print(f"  - 엔진: f-string\n")
    
    # AsyncOpenAI 클라이언트 생성 (로컬 FastAPI의 Gemini 프록시 사용)
    api_key = os.getenv("GOOGLE_API_KEY", "dummy-key")
    base_url = "http://localhost:8000/api"  # FastAPI OpenAI 호환 엔드포인트
    
    openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # APO 알고리즘 설정 (고품질: beam_width=5, branch_factor=4, beam_rounds=10)
    apo_config = {
        "beam_width": 5,        # 각 라운드에서 유지할 최고 프롬프트 수
        "branch_factor": 4,     # 각 부모에서 생성할 자식 프롬프트 수
        "beam_rounds": 10,      # beam search 라운드 수
        "gradient_batch_size": 2,  # gradient 계산에 사용할 샘플 수
        "val_batch_size": 2,    # 검증에 사용할 배치 크기
    }
    
    print("⚙️ APO 설정 (고품질):")
    for key, value in apo_config.items():
        print(f"  - {key}: {value}")
    print()
    
    # APO 알고리즘 초기화
    apo_algorithm = agl.algorithm.APO(
        async_openai_client=openai_client,
        gradient_model="gemini-2.0-flash-exp",  # Gemini로 gradient 생성
        apply_edit_model="gemini-2.0-flash-exp",  # Gemini로 edit 적용
        beam_width=apo_config["beam_width"],
        branch_factor=apo_config["branch_factor"],
        beam_rounds=apo_config["beam_rounds"],
        gradient_batch_size=apo_config["gradient_batch_size"],
        val_batch_size=apo_config["val_batch_size"],
        run_initial_validation=True,
    )
    
    # APO 알고리즘에 초기 리소스 및 Store 설정
    store = agl.InMemoryLightningStore()
    apo_algorithm.set_store(store)
    apo_algorithm.set_initial_resources({"system_prompt": seed_prompt})
    
    print("🎯 APO 학습 시작...\n")
    print("✅ Gemini API 사용 (로컬 FastAPI 프록시 경유)")
    print("⏱️  예상 소요 시간: 30-60분 (전체 학습)\n")
    
    try:
        # APO 알고리즘 직접 실행
        await apo_algorithm.run(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        
        print("\n✅ APO 학습 완료!")
        
        # 최적화된 프롬프트 가져오기
        best_prompt = apo_algorithm.get_best_prompt()
        
        output_dir = "data/optimized_prompts"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "system_prompt_optimized.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(best_prompt.template)
        print(f"💾 최적화된 프롬프트 저장: {output_path}")
        
        print("\n📊 최적화 결과:")
        print(f"  최고 점수: {apo_algorithm._history_best_score:.3f}")
        print(f"  버전: {apo_algorithm._history_best_version}")
        
    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
