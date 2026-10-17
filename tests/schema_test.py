import pytest

from nexnode.schema.response import ChatCompletionFunction


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "101", "arguments": '{"hello" : "what"'},
    ],
)
def test_chat_completion_function_model_arguement_serialization_handling_for_invalid_arguments(
    payload: dict,
):
    tool_function = ChatCompletionFunction(**payload)
    assert "101_error" in tool_function.parsed_arguments
