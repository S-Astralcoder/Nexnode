from typing import Literal

from pydantic import BaseModel


class AgentConfig(BaseModel):
    base_url: Literal["https://generativelanguage.googleapis.com/v1beta/openai/"] | str # pyright: ignore[reportInvalidTypeForm]
    api_key: str
    model: str
    tool_choice: str | dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
    timeout: int = 10
