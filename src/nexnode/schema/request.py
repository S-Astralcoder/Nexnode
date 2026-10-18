from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from .messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)


class ChatCompletionHeader(BaseModel):
    token: str = Field(min_length=1)
    content_type: Literal["application/json"] = "application/json"

    @model_serializer
    def serialize_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": self.content_type,
        }


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[
        SystemMessage | UserMessage | AssistantMessage | ToolCallResultMessage
    ] = Field(default_factory=list)
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
