import pytest

from nexnode.core.agent import SequentialAgentNode
from nexnode.core.config import AgentConfig
from nexnode.core.memory import AgentMemory
from nexnode.exceptions import (
    InvalidMessageSequenceError,
    NexnodeError,
    ToolCallPermissionError,
    ToolCallResultPermissionError,
)
from nexnode.schema.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)
from nexnode.schema.response import ChatCompletionResponse, ChatCompletionToolCall
from nexnode.tools.base import BaseTool, tool


class MockTool(BaseTool):
    @tool()
    def success_action(self, val: int) -> int:
        return val * 2

    @tool()
    def failing_action(self) -> float:
        return 1 / 0


def test_initial_internal_state():
    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
    )
    assert agent._internal_state == "AWAITING_INPUT"


def test_internal_state_direct_agent_text_response(monkeypatch: pytest.MonkeyPatch):
    memory = AgentMemory()
    memory.add(UserMessage(content="Hello"))

    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
    )

    mock_llm_response = {
        "id": "chatcmpl-text",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }

    class MockResponse:
        def json(self):
            return mock_llm_response

        def raise_for_status(self):
            pass

    monkeypatch.setattr(agent.client, "post", lambda *args, **kwargs: MockResponse())

    result = agent.run_step()
    assert isinstance(result, ChatCompletionResponse)
    assert agent._internal_state == "AGENT_RESPONSE"


def test_internal_state_transitions_full_tool_cycle(monkeypatch: pytest.MonkeyPatch):
    memory = AgentMemory()
    memory.add(UserMessage(content="Run math tool"))

    tool_instance = MockTool()
    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
        tools=[tool_instance],
    )

    llm_step1 = {
        "id": "chatcmpl-tools",
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
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "MockTool__success_action",
                                "arguments": '{"val": 5}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "MockTool__failing_action",
                                "arguments": "{}",
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    llm_step2 = {
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Calculations processed.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
    }

    responses = [llm_step1, llm_step2]
    call_idx = 0

    class MockResp:
        def __init__(self, data: dict):
            self._data = data

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def mock_post(*args, **kwargs):
        nonlocal call_idx
        resp = MockResp(responses[call_idx])
        call_idx += 1
        return resp

    monkeypatch.setattr(agent.client, "post", mock_post)

    # 1. First run_step: LLM returns tool calls -> AWAITING_TOOL_EXECUTION
    res1 = agent.run_step()
    assert isinstance(res1, ChatCompletionResponse)
    assert agent._internal_state == "AWAITING_TOOL_EXECUTION"
    assert len(agent._tool_execution_queue) == 2

    # 2. Second run_step: executes first tool (success) -> TOOL_EXECUTION_SUCCESSFUL_NEXT
    res2 = agent.run_step()
    assert isinstance(res2, ToolCallResultMessage)
    assert res2.tool_call_id == "call_1"
    assert agent._internal_state == "TOOL_EXECUTION_SUCCESSFUL_NEXT"
    assert len(agent._tool_execution_queue) == 1

    # 3. Third run_step: executes second tool (runtime error) -> TOOL_EXECUTION_ERROR_NEXT
    res3 = agent.run_step()
    assert isinstance(res3, ToolCallResultMessage)
    assert res3.tool_call_id == "call_2"
    assert agent._internal_state == "TOOL_EXECUTION_ERROR_NEXT"
    assert len(agent._tool_execution_queue) == 0

    # 4. Fourth run_step: queue is empty -> TOOL_EXECUTION_END, returns None
    res4 = agent.run_step()
    assert res4 is None
    assert agent._internal_state == "TOOL_EXECUTION_END"

    # 5. Fifth run_step: from TOOL_EXECUTION_END -> calls LLM for final response -> AGENT_RESPONSE
    res5 = agent.run_step()
    assert isinstance(res5, ChatCompletionResponse)
    assert agent._internal_state == "AGENT_RESPONSE"


def test_internal_state_missing_tool_error_next():
    memory = AgentMemory()
    memory.add(UserMessage(content="Query"))
    memory.add(AssistantMessage(content=None))

    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
        tools=[MockTool()],
    )

    missing_tool_call = ChatCompletionToolCall(
        id="call_unknown",
        type="function",
        function={
            "name": "NonExistentTool__action",
            "arguments": "{}",
        }, # pyright: ignore[reportArgumentType]
    )

    agent._internal_state = "AWAITING_TOOL_EXECUTION"
    agent._tool_execution_queue = [missing_tool_call]

    result = agent.run_step()
    assert isinstance(result, ToolCallResultMessage)
    assert "doesn't exist" in result.content
    assert agent._internal_state == "TOOL_EXECUTION_ERROR_NEXT"


def test_internal_state_before_tool_permission_denied():
    class PermissionDeniedAgent(SequentialAgentNode):
        def before_tool_execution(self, tool_call: ChatCompletionToolCall) -> None:
            raise ToolCallPermissionError("Execution rejected")

    memory = AgentMemory()
    memory.add(UserMessage(content="Query"))
    memory.add(AssistantMessage(content=None))

    agent = PermissionDeniedAgent(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
        tools=[MockTool()],
    )

    tool_call = ChatCompletionToolCall(
        id="call_perm_1",
        type="function",
        function={
            "name": "MockTool__success_action",
            "arguments": '{"val": 1}',
        }, # pyright: ignore[reportArgumentType]
    )

    agent._internal_state = "AWAITING_TOOL_EXECUTION"
    agent._tool_execution_queue = [tool_call]

    result = agent.run_step()
    assert isinstance(result, ToolCallResultMessage)
    assert agent._internal_state == "TOOL_EXECUTION_PERMISSION_DENIED"


def test_internal_state_after_tool_result_blocked():
    class ResultBlockedAgent(SequentialAgentNode):
        def after_tool_execution(self, tool_call_result: ToolCallResultMessage) -> None:
            raise ToolCallResultPermissionError("Result leaked secrets")

    memory = AgentMemory()
    memory.add(UserMessage(content="Query"))
    memory.add(AssistantMessage(content=None))

    agent = ResultBlockedAgent(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
        tools=[MockTool()],
    )

    tool_call = ChatCompletionToolCall(
        id="call_block_1",
        type="function",
        function={
            "name": "MockTool__success_action",
            "arguments": '{"val": 1}',
        }, # pyright: ignore[reportArgumentType]
    )

    agent._internal_state = "AWAITING_TOOL_EXECUTION"
    agent._tool_execution_queue = [tool_call]

    result = agent.run_step()
    assert isinstance(result, ToolCallResultMessage)
    assert agent._internal_state == "TOOL_EXECUTION_RESULT_BLOCKED"


def test_internal_state_resets_to_awaiting_input_on_invalid_sequence():
    memory = AgentMemory()
    memory.add(SystemMessage(content="System only"))

    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=memory,
    )
    agent._internal_state = "TOOL_EXECUTION_END"

    with pytest.raises(InvalidMessageSequenceError):
        agent.run_step()

    assert agent._internal_state == "AWAITING_INPUT"


def test_run_step_unexpected_internal_state_raises_nexnode_error():
    agent = SequentialAgentNode(
        config=AgentConfig(base_url="https://api.test", api_key="test", model="gpt-4o"),
        memory=AgentMemory(),
    )
    agent._internal_state = "CORRUPTED_STATE"  # type: ignore[assignment]

    with pytest.raises(NexnodeError):
        agent.run_step()
