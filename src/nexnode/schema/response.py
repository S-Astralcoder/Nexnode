import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ChatCompletionFunction(BaseModel):
    name: str
    arguments: str

    @property
    def parsed_arguments(self) -> dict[str, Any]:
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError as e:
            return {"101_error": e}


class ChatCompletionToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ChatCompletionFunction
    extra_content: dict | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatCompletionToolCall] | None = None
    extra_content: dict | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: dict | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict | None = None
    completion_tokens_details: dict | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None = None
    service_tier: str | None = None
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None
