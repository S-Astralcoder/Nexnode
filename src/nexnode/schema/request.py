from typing import Literal

from pydantic import BaseModel, Field, model_serializer

from .messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)


class ChatCompletionHeader(BaseModel):
    authorization: str
    content_type: Literal["application/json"] = "application/json"

    @model_serializer
    def model_serialization(self):
        return {
            "Authorization": self.authorization.strip()
            if self.authorization.strip().startswith("Bearer")
            else f"Bearer {self.authorization}",
            "Content-Type": self.content_type,
        }


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[
        SystemMessage | UserMessage | AssistantMessage | ToolCallResultMessage
    ] = Field(default_factory=list)
    tools: list | None = None
    tool_choice: str | dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
