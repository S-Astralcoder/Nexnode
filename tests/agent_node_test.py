import pytest

from nexnode.core.agent import BaseAgentNode
from nexnode.core.config import AgentConfig
from nexnode.core.memory import AgentMemory
from nexnode.exceptions import (
    DuplicateToolClassError,
    EmptyMemoryError,
    InvalidMessageSequenceError,
    InvalidToolClassError,
    MaxIterationLimitError,
    ToolCallPermissionError,
    ToolCallResultPermissionError,
)
from nexnode.schema.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)
from nexnode.schema.response import (
    ChatCompletionFunction,
    ChatCompletionResponse,
    ChatCompletionToolCall,
)
from nexnode.tools.base import BaseTool, tool


def test_duplicate_tools_rejection():
    class MockAgent(BaseAgentNode):
        def __init__(self, config: AgentConfig, memory: AgentMemory, tools: list[BaseTool] | None = None) -> None:
            super().__init__(config, memory, tools)

    class MockTool(BaseTool):
        pass

    with pytest.raises(DuplicateToolClassError):
        MockAgent(config=AgentConfig(base_url="", api_key="", model=""), memory=AgentMemory(), tools=[MockTool(), MockTool()])


def test_valid_response_convertion_to_pydantic_model(monkeypatch : pytest.MonkeyPatch):
    class MockAgent(BaseAgentNode):
        def __init__(self, config: AgentConfig, memory: AgentMemory, tools: list[BaseTool] | None = None) -> None:
            super().__init__(config, memory, tools)

    mock_agent = MockAgent(config=AgentConfig(base_url="asdas", api_key="dasdasd", model="dasdasdas"), memory=AgentMemory())

    class MockResponse:
        def __init__(self) -> None:
            pass

        def json(self):
            return {
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o",
  "system_fingerprint": "fp_44702179b9",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": None
      },
      "logprobs": None,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  }
}


        def raise_for_status(self):
            pass

    def demo_post(*args, **kawrgs):
        return MockResponse()

    monkeypatch.setattr(mock_agent.client, "post", demo_post)


    response = mock_agent.send_prompt("hello")

    assert isinstance(response, ChatCompletionResponse)
    assert response.id == "chatcmpl-123"
    assert response.object == "chat.completion"
    assert response.created == 1677652288
    assert response.model == "gpt-4o"
    assert response.system_fingerprint == "fp_44702179b9"

    assert len(response.choices) == 1
    choice = response.choices[0]
    assert choice.index == 0
    assert choice.finish_reason == "stop"
    assert choice.logprobs is None

    message = choice.message
    assert message.role == "assistant"
    assert message.content == "Hello! How can I assist you today?"
    assert message.refusal is None
    assert message.tool_calls is None

    assert response.usage is not None
    assert response.usage.prompt_tokens == 9
    assert response.usage.completion_tokens == 12
    assert response.usage.total_tokens == 21
    assert response.usage.prompt_tokens_details == {"cached_tokens": 0}
    assert response.usage.completion_tokens_details == {
        "reasoning_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }


def test_direct_basetool_instance_triggers_invalid_tool_class_error():
    direct_base_tool = object.__new__(BaseTool)

    with pytest.raises(InvalidToolClassError):
        BaseAgentNode(
            config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
            memory=AgentMemory(),
            tools=[direct_base_tool],
        )


def test_tool_class_not_inheriting_from_basetool_triggers_error():
    class UnrelatedTool:
        pass

    with pytest.raises(InvalidToolClassError):
        BaseAgentNode(
            config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
            memory=AgentMemory(),
            tools=[UnrelatedTool()],  # type: ignore[list-item]
        )


def test_empty_memory_triggers_empty_memory_error():
    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
    )

    with pytest.raises(EmptyMemoryError):
        agent._chat_completion()


def test_chat_completion_with_invalid_message_sequence_triggers_error():
    memory = AgentMemory()
    memory.add(SystemMessage(content="You are a system prompt."))

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
    )

    with pytest.raises(InvalidMessageSequenceError):
        agent._chat_completion()


def test_all_added_tools_are_registered_properly():
    class ToolA(BaseTool):
        @tool()
        def action_a(self, val: int) -> int:
            """Action A docstring."""
            return val

    class ToolB(BaseTool):
        @tool()
        def action_b(self, text: str) -> str:
            """Action B docstring."""
            return text

    tool_a = ToolA()
    tool_b = ToolB()

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[tool_a, tool_b],
    )

    assert agent.tool_class_links == {"ToolA": tool_a, "ToolB": tool_b}
    assert len(agent.tools_schema) == 2

    schema_names = {t.function.name for t in agent.tools_schema}
    assert schema_names == {"ToolA__action_a", "ToolB__action_b"}


def test_max_iteration_limit_error_triggered(monkeypatch: pytest.MonkeyPatch):
    class LoopingTool(BaseTool):
        @tool()
        def ping(self) -> str:
            return "pong"

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[LoopingTool()],
    )

    class MockLoopResponse:
        def json(self):
            return {
                "id": "chatcmpl-loop",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_loop_1",
                                    "type": "function",
                                    "function": {
                                        "name": "LoopingTool__ping",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }

        def raise_for_status(self):
            pass

    monkeypatch.setattr(agent.client, "post", lambda *args, **kwargs: MockLoopResponse())

    with pytest.raises(MaxIterationLimitError):
        agent.send_prompt("Trigger infinite tool call loop", max_iteration=3)


def test_before_tool_execution_permission_denied():
    class RestrictedAgent(BaseAgentNode):
        def before_tool_execution(self, tool_call: ChatCompletionToolCall) -> None:
            raise ToolCallPermissionError("Tool execution is blocked by policy.")

    class Calculator(BaseTool):
        @tool()
        def add(self, a: int, b: int) -> int:
            return a + b

    agent = RestrictedAgent(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[Calculator()],
    )

    tool_call = ChatCompletionToolCall(
        id="call_calc_1",
        type="function",
        function=ChatCompletionFunction(
            name="Calculator__add",
            arguments='{"a": 2, "b": 3}',
        ),
    )

    result = agent._execute_tool_call(tool_call)
    assert isinstance(result, ToolCallResultMessage)
    assert result.tool_call_id == "call_calc_1"
    assert "Tool execution is blocked by policy." in result.content


def test_after_tool_execution_permission_denied():
    class RedactingAgent(BaseAgentNode):
        def after_tool_execution(self, tool_call_result: ToolCallResultMessage) -> None:
            raise ToolCallResultPermissionError("Tool result contains sensitive data and was blocked.")

    class SecureTool(BaseTool):
        @tool()
        def fetch_secret(self) -> str:
            return "secret_data"

    agent = RedactingAgent(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[SecureTool()],
    )

    tool_call = ChatCompletionToolCall(
        id="call_sec_1",
        type="function",
        function=ChatCompletionFunction(
            name="SecureTool__fetch_secret",
            arguments="{}",
        ),
    )

    result = agent._execute_tool_call(tool_call)
    assert isinstance(result, ToolCallResultMessage)
    assert result.tool_call_id == "call_sec_1"
    assert "Tool result contains sensitive data and was blocked." in result.content


def test_execute_missing_tool_call():
    class ExistingTool(BaseTool):
        @tool()
        def dummy(self) -> str:
            return "ok"

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[ExistingTool()],
    )

    missing_class_call = ChatCompletionToolCall(
        id="call_miss_1",
        type="function",
        function=ChatCompletionFunction(
            name="NonExistentClass__dummy",
            arguments="{}",
        ),
    )
    result_missing_class = agent._execute_tool_call(missing_class_call)
    assert "The given tool NonExistentClass__dummy doesn't exist." in result_missing_class.content

    missing_method_call = ChatCompletionToolCall(
        id="call_miss_2",
        type="function",
        function=ChatCompletionFunction(
            name="ExistingTool__not_a_method",
            arguments="{}",
        ),
    )
    result_missing_method = agent._execute_tool_call(missing_method_call)
    assert "The given tool ExistingTool__not_a_method doesn't exist." in result_missing_method.content


def test_send_prompt_successful_tool_execution_flow(monkeypatch: pytest.MonkeyPatch):
    class MathTool(BaseTool):
        @tool()
        def multiply(self, a: int, b: int) -> int:
            return a * b

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[MathTool()],
    )

    responses = [
        {
            "id": "chatcmpl-step-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mult_1",
                                "type": "function",
                                "function": {
                                    "name": "MathTool__multiply",
                                    "arguments": '{"a": 6, "b": 7}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
        {
            "id": "chatcmpl-step-2",
            "object": "chat.completion",
            "created": 1700000001,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The result of 6 multiplied by 7 is 42.",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    class MockSequenceResponse:
        def __init__(self, data: dict):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    call_index = 0

    def mock_post(*args, **kwargs):
        nonlocal call_index
        resp = MockSequenceResponse(responses[call_index])
        call_index += 1
        return resp

    monkeypatch.setattr(agent.client, "post", mock_post)

    final_response = agent.send_prompt("Multiply 6 and 7")
    assert final_response.choices[0].message.content == "The result of 6 multiplied by 7 is 42."

    messages = agent.memory.get_messages()
    assert len(messages) == 4
    assert isinstance(messages[0], UserMessage)
    assert messages[0].content == "Multiply 6 and 7"
    assert isinstance(messages[1], AssistantMessage)
    assert messages[1].tool_calls[0].id == "call_mult_1" # pyright: ignore[reportOptionalSubscript]
    assert isinstance(messages[2], ToolCallResultMessage)
    assert messages[2].tool_call_id == "call_mult_1"
    assert messages[2].content == "42"
    assert isinstance(messages[3], AssistantMessage)
    assert messages[3].content == "The result of 6 multiplied by 7 is 42."


def test_send_prompt_multiple_tools_in_single_turn(monkeypatch: pytest.MonkeyPatch):
    class Arithmetic(BaseTool):
        @tool()
        def add(self, a: int, b: int) -> int:
            return a + b

        @tool()
        def multiply(self, a: int, b: int) -> int:
            return a * b

    agent = BaseAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
        tools=[Arithmetic()],
    )

    parallel_tool_response = {
        "id": "chatcmpl-multi",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_add",
                            "type": "function",
                            "function": {
                                "name": "Arithmetic__add",
                                "arguments": '{"a": 10, "b": 20}',
                            },
                        },
                        {
                            "id": "call_mult",
                            "type": "function",
                            "function": {
                                "name": "Arithmetic__multiply",
                                "arguments": '{"a": 3, "b": 4}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    final_response_data = {
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sum is 30 and product is 12.",
                },
                "finish_reason": "stop",
            }
        ],
    }

    call_count = 0

    class MockResp:
        def __init__(self, data: dict):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def mock_post(*args, **kwargs):
        nonlocal call_count
        resp = MockResp(parallel_tool_response if call_count == 0 else final_response_data)
        call_count += 1
        return resp

    monkeypatch.setattr(agent.client, "post", mock_post)

    resp = agent.send_prompt("Compute 10+20 and 3*4")
    assert resp.choices[0].message.content == "Sum is 30 and product is 12."

    messages = agent.memory.get_messages()
    assert len(messages) == 5
    assert isinstance(messages[0], UserMessage)
    assert isinstance(messages[1], AssistantMessage)
    assert len(messages[1].tool_calls) == 2  # pyright: ignore[reportArgumentType]
    assert isinstance(messages[2], ToolCallResultMessage)
    assert messages[2].tool_call_id == "call_add"
    assert messages[2].content == "30"
    assert isinstance(messages[3], ToolCallResultMessage)
    assert messages[3].tool_call_id == "call_mult"
    assert messages[3].content == "12"
    assert isinstance(messages[4], AssistantMessage)
