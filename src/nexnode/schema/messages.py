from typing import Literal

from pydantic import BaseModel

from .response import ChatCompletionToolCall


class SystemMessage(BaseModel):
    role: Literal["system", "developer"] = "system"
    content: str


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | list[dict]
    name: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ChatCompletionToolCall] | None = None


class ToolCallResultMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
