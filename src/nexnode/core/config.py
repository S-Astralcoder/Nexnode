from pydantic import BaseModel


class AgentConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    tool_choice: str | dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
    timeout: int = 10
