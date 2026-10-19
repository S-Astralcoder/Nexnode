from nexnode.core.config import AgentConfig


def test_agent_config_model_fields_contract():
    expected_fields = {
        "base_url",
        "api_key",
        "model",
        "tool_choice",
        "temperature",
        "top_p",
        "max_completion_tokens",
        "response_format",
        "seed",
        "timeout",
    }

    actual_fields = set(AgentConfig.model_fields.keys())
    assert actual_fields == expected_fields


def test_agent_config_default_values_contract():
    fields = AgentConfig.model_fields

    assert fields["base_url"].is_required() is True
    assert fields["api_key"].is_required() is True
    assert fields["model"].is_required() is True

    assert fields["tool_choice"].default is None
    assert fields["temperature"].default is None
    assert fields["top_p"].default is None
    assert fields["max_completion_tokens"].default is None
    assert fields["response_format"].default is None
    assert fields["seed"].default is None
    assert fields["timeout"].default == 10


def test_agent_config_optional_fields_populate_chat_completion_request():
    from nexnode.schema.messages import UserMessage
    from nexnode.schema.request import ChatCompletionRequest

    config = AgentConfig(
        base_url="https://api.openai.com/v1",
        api_key="secret-key",
        model="gpt-4o",
        tool_choice="auto",
        temperature=0.2,
        top_p=0.95,
        max_completion_tokens=256,
        response_format={"type": "json_object"},
        seed=42,
        timeout=15,
    )

    req = ChatCompletionRequest(
        tools=[],
        messages=[UserMessage(content="Hello")],
        **config.model_dump(),
    )

    dumped = req.model_dump(exclude_none=True)
    assert dumped["model"] == "gpt-4o"
    assert dumped["tool_choice"] == "auto"
    assert dumped["temperature"] == 0.2
    assert dumped["top_p"] == 0.95
    assert dumped["max_completion_tokens"] == 256
    assert dumped["response_format"] == {"type": "json_object"}
    assert dumped["seed"] == 42
