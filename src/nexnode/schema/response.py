
import json
from typing import Literal

from pydantic import BaseModel


class ChatCompletionFunction(BaseModel):
    name : str
    arguments : str

    @property
    def model_data(self) -> dict:
        return json.loads(self.arguments)

class ChatCompletionToolCall(BaseModel):
    id : str
    type : Literal["function"]
    function : ChatCompletionFunction


class ChatCompletionMessage(BaseModel):
    role : Literal["assistant"]
    content : str | None = None
    refusal : str | None = None
    tool_calls : list[ChatCompletionToolCall] | None = None

class ChatCompletionChoice(BaseModel):
    index : int
    message : ChatCompletionMessage
    finish_reason : str | None = None
    logprobs : dict | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict | None = None
    completion_tokens_details: dict | None = None

class ChatCompletionResponse(BaseModel):
    id : str
    object : str
    created : int
    model : str
    system_fingerprint : str | None = None
    service_tier : str | None = None
    choices : list[ChatCompletionChoice]
    usage : ChatCompletionUsage | None = None

