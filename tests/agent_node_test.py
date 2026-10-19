import pytest

from nexnode.core.agent import BaseAgentNode
from nexnode.core.config import AgentConfig
from nexnode.core.memory import AgentMemory
from nexnode.exceptions import DuplicateToolClassError
from nexnode.tool.base import BaseTool


def test_duplicate_tools_rejection():
    class MockAgent(BaseAgentNode):
        def __init__(self, config: AgentConfig, memory: AgentMemory, tools: list[BaseTool] | None = None) -> None:
            super().__init__(config, memory, tools)

    class MockTool(BaseTool):
        pass

    with pytest.raises(DuplicateToolClassError):
        MockAgent(config=AgentConfig(base_url="", api_key="", model=""), memory=AgentMemory(), tools=[MockTool(), MockTool()])





