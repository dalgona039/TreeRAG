"""
OpenAI 호환 API 엔드포인트
Agent Lightning의 APO가 Gemini API를 사용할 수 있도록 변환
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os
import time


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


async def openai_to_gemini_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI 형식의 chat completion 요청을 Gemini API로 변환하여 처리
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # OpenAI 메시지를 Gemini 형식으로 변환
    gemini_contents = []
    system_instruction = None
    
    for msg in request.messages:
        if msg.role == "system":
            system_instruction = msg.content
        else:
            role = "user" if msg.role == "user" else "model"
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg.content)]
                )
            )
    
    # Gemini 모델 선택 (OpenAI 모델명을 Gemini로 매핑)
    gemini_model = "gemini-2.0-flash-exp"  # 기본 모델
    if "gpt-4" in request.model.lower():
        gemini_model = "gemini-2.0-flash-exp"
    elif "gpt-3.5" in request.model.lower():
        gemini_model = "gemini-1.5-flash"
    
    # Gemini API 호출
    config = types.GenerateContentConfig(
        temperature=request.temperature,
        max_output_tokens=request.max_tokens,
        system_instruction=system_instruction,
    )
    
    response = client.models.generate_content(
        model=gemini_model,
        contents=gemini_contents,
        config=config,
    )
    
    # Gemini 응답을 OpenAI 형식으로 변환
    assistant_message = response.text
    
    # 토큰 사용량 추정 (정확하지 않지만 근사치)
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(assistant_message.split())
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=gemini_model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=assistant_message
                ),
                finish_reason="stop"
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )
