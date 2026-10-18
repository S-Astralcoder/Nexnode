import pytest

from nexnode.exceptions import ToolNameLengthExceededError
from nexnode.tool.base import BaseTool, tool



def test_reject_basetool_instance_creation():
    with pytest.raises(NotImplementedError):
        tool_base = BaseTool()


def test_max_function_name_limit():
    with pytest.raises(ToolNameLengthExceededError):
        class DummyTool(BaseTool):

            @tool()
            def abcdefddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd(self):
                pass

        dummy = DummyTool()

        dummy.get_tool_schemas()
