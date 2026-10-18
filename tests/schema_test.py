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
