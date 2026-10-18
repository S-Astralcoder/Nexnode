import pytest

from nexnode.core.memory import AgentMemory
from nexnode.exceptions import InvalidMessageSequenceError
from nexnode.schema.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)
from nexnode.schema.response import (
    ChatCompletionChoice,
    ChatCompletionFunction,
    ChatCompletionMessage,
    ChatCompletionResponse,
    ChatCompletionToolCall,
)


def test_invalid_message_sequence_triggers_error():
    memory = AgentMemory()

    with pytest.raises(InvalidMessageSequenceError):
        memory.add(AssistantMessage(content="Cannot start with AssistantMessage"))

    with pytest.raises(InvalidMessageSequenceError):
        memory.add(ToolCallResultMessage(tool_call_id="call_1", content="ok"))

    memory.add(UserMessage(content="Hello"))

    with pytest.raises(InvalidMessageSequenceError):
        memory.add(UserMessage(content="Consecutive user message not allowed"))

    with pytest.raises(InvalidMessageSequenceError):
        memory.add(SystemMessage(content="System message after user message"))


def test_chat_completion_response_is_converted_and_added():
    memory = AgentMemory()
    memory.add(UserMessage(content="What is the weather?"))

    response = ChatCompletionResponse(
        id="chatcmpl-1",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="The weather is sunny.",
                ),
                finish_reason="stop",
            )
        ],
    )

    memory.add(response)

    messages = memory.get_messages()
    assert len(messages) == 2
    assert isinstance(messages[-1], AssistantMessage)
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "The weather is sunny."


def test_no_important_data_loss_during_chat_completion_response_conversion():
    memory = AgentMemory()
    memory.add(UserMessage(content="Run tool"))

    tool_call = ChatCompletionToolCall(
        id="call_abc123",
        type="function",
        function=ChatCompletionFunction(
            name="WeatherTool__get_weather",
            arguments='{"city": "Paris"}',
        ),
    )

    response_choice_message = ChatCompletionMessage(
        role="assistant",
        content="I will check the weather for Paris.",
        tool_calls=[tool_call],
        refusal="Safety refusal reason if present",
        extra_content={"custom_provider_key": "custom_provider_value"},
    )

    response = ChatCompletionResponse(
        id="chatcmpl-2",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=response_choice_message,
                finish_reason="tool_calls",
            )
        ],
    )

    memory.add(response)

    converted = memory.get_messages()[-1]
    assert isinstance(converted, AssistantMessage)
    assert converted.content == response_choice_message.content
    assert converted.tool_calls == response_choice_message.tool_calls
    assert converted.refusal == response_choice_message.refusal
    assert converted.extra_content == response_choice_message.extra_content


def test_valid_message_sequence_insertion():
    memory = AgentMemory()

    sys_msg = SystemMessage(content="You are an assistant.")
    user_msg_1 = UserMessage(content="Search for Python info.")
    assistant_msg = AssistantMessage(
        content="Searching now.",
        tool_calls=[
            ChatCompletionToolCall(
                id="call_1",
                type="function",
                function=ChatCompletionFunction(name="Search__query", arguments='{"q": "python"}'),
            )
        ],
    )
    tool_msg = ToolCallResultMessage(tool_call_id="call_1", content='{"result": "Python 3.12"}')
    user_msg_2 = UserMessage(content="Thank you!")

    memory.add(sys_msg)
    memory.add(user_msg_1)
    memory.add(assistant_msg)
    memory.add(tool_msg)
    memory.add(user_msg_2)

    messages = memory.get_messages()
    assert len(messages) == 5
    assert messages[0] == sys_msg
    assert messages[1] == user_msg_1
    assert messages[2] == assistant_msg
    assert messages[3] == tool_msg
    assert messages[4] == user_msg_2


def test_adding_none_triggers_error():
    memory = AgentMemory()
    with pytest.raises(InvalidMessageSequenceError):
        memory.add(None)  # type: ignore[arg-type]

    memory.add(UserMessage(content="Hello"))
    with pytest.raises(InvalidMessageSequenceError):
        memory.add(None)  # type: ignore[arg-type]
