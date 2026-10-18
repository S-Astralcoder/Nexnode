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
