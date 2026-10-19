import pytest

from nexnode.schema.response import ChatCompletionFunction


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "test_unclosed_brace", "arguments": '{"hello": "what"'},
        {"name": "test_unclosed_nested", "arguments": '{"nested": {"key": "val"'},
        {"name": "test_missing_value", "arguments": '{"key":'},
        {"name": "test_unquoted_key", "arguments": '{key: "val"}'},
        {"name": "test_single_quotes", "arguments": "{'key': 'value'}"},
        {"name": "test_trailing_comma_dict", "arguments": '{"key": "value",}'},
        {"name": "test_missing_colon", "arguments": '{"key" "value"}'},
        {"name": "test_empty_string", "arguments": ""},
        {"name": "test_whitespace_only", "arguments": "   \n\t  "},
        {"name": "test_invalid_escape", "arguments": '{"path": "\\a"}'},
        {"name": "test_raw_newline_in_string", "arguments": '{"text": "line1\nline2"}'},
        {"name": "test_truncated_unicode", "arguments": '{"char": "\\u123"}'},
        {"name": "test_multiple_root_objects", "arguments": '{"a": 1} {"b": 2}'},
        {"name": "test_plain_text_error", "arguments": "Internal Server Error: 500"},
        {
            "name": "test_html_content",
            "arguments": "<html><body>Bad Gateway</body></html>",
        },
        {
            "name": "test_python_literals",
            "arguments": "{'is_admin': True, 'count': None}",
        },
    ],
)
def test_chat_completion_function_model_arguement_serialization_handling_for_invalid_arguments(
    payload: dict,
):
    tool_function = ChatCompletionFunction(**payload)
    assert "101_error" in tool_function.parsed_arguments


def test_chat_completion_function_valid_arguments_parsing():
    valid_payload = {
        "name": "test_tool",
        "arguments": '{"query": "search term", "limit": 10}',
    }
    tool_function = ChatCompletionFunction(**valid_payload)
    parsed = tool_function.parsed_arguments

    assert parsed == {"query": "search term", "limit": 10}
    assert "101_error" not in parsed


def test_chat_completion_header_serialization():
    from nexnode.schema.request import ChatCompletionHeader

    header = ChatCompletionHeader(token="test_token_123")
    serialized = header.model_dump()
    assert serialized == {
        "Authorization": "Bearer test_token_123",
        "Content-Type": "application/json",
    }


def test_message_schemas_happy_path():
    from nexnode.schema.messages import (
        AssistantMessage,
        SystemMessage,
        ToolCallResultMessage,
        UserMessage,
    )
    from nexnode.schema.response import ChatCompletionFunction, ChatCompletionToolCall

    sys_msg = SystemMessage(content="You are helpful.")
    assert sys_msg.role == "system"
    assert sys_msg.content == "You are helpful."

    user_msg = UserMessage(content="Hello", name="Alice")
    assert user_msg.role == "user"
    assert user_msg.content == "Hello"
    assert user_msg.name == "Alice"

    tool_call = ChatCompletionToolCall(
        id="call_1",
        type="function",
        function=ChatCompletionFunction(name="test_tool", arguments='{"x": 1}'),
    )
    assistant_msg = AssistantMessage(content="Let me check.", tool_calls=[tool_call])
    assert assistant_msg.role == "assistant"
    assert len(assistant_msg.tool_calls) == 1 # pyright: ignore[reportArgumentType]
    assert assistant_msg.tool_calls[0].function.parsed_arguments == {"x": 1} # pyright: ignore[reportOptionalSubscript]

    tool_res = ToolCallResultMessage(tool_call_id="call_1", content="result_ok")
    assert tool_res.role == "tool"
    assert tool_res.tool_call_id == "call_1"
    assert tool_res.content == "result_ok"


def test_chat_completion_request_and_response_happy_path():
    from nexnode.schema.messages import UserMessage
    from nexnode.schema.request import (
        ChatCompletionRequest,
        FunctionDefinition,
        ToolDefinition,
    )
    from nexnode.schema.response import (
        ChatCompletionResponse,
    )

    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[UserMessage(content="Ping")],
        tools=[
            ToolDefinition(
                type="function",
                function=FunctionDefinition(
                    name="dummy",
                    description="dummy description",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ],
        temperature=0.7,
    )
    req_dict = req.model_dump(exclude_none=True)
    assert req_dict["model"] == "gpt-4o"
    assert len(req_dict["messages"]) == 1
    assert len(req_dict["tools"]) == 1

    resp_data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Pong",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    resp = ChatCompletionResponse(**resp_data)
    assert resp.id == "chatcmpl-123"
    assert resp.choices[0].message.content == "Pong"
    assert resp.usage.total_tokens == 15 # pyright: ignore[reportOptionalMemberAccess]


def test_chat_completion_response_with_minimal_optional_fields():
    from nexnode.schema.response import ChatCompletionResponse

    minimal_payload = {
        "id": "chatcmpl-min",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Minimal response",
                },
            }
        ],
    }

    resp = ChatCompletionResponse(**minimal_payload)
    assert resp.id == "chatcmpl-min"
    assert resp.system_fingerprint is None
    assert resp.service_tier is None
    assert resp.usage is None
    assert resp.choices[0].finish_reason is None
    assert resp.choices[0].logprobs is None
    assert resp.choices[0].message.refusal is None
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].message.extra_content is None


def test_user_message_multimodal_content_payload():
    from nexnode.schema.messages import UserMessage

    multimodal_parts = [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/chart.png"},
        },
    ]

    msg = UserMessage(content=multimodal_parts)
    assert msg.role == "user"
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[0]["type"] == "text"
    assert msg.content[1]["image_url"]["url"] == "https://example.com/chart.png"
