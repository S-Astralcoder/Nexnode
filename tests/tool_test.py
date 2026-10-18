import pytest

from nexnode.exceptions import ToolNameLengthExceededError
from nexnode.tool.base import BaseTool, tool


def test_creation_of_basetool_class_instance_not_allowed():
    with pytest.raises(NotImplementedError):
        BaseTool()


def test_no_crash_on_null_tools():
    class Dummy(BaseTool):
        pass

    Dummy().get_tool_schemas()


def test_tool_registry_matches_decorated_tools():

    class SampleTool(BaseTool):
        @tool()
        def tool_one(self, x: int) -> int:
            return x

        @tool()
        def tool_two(self, text: str) -> str:
            return text

        def regular_method(self) -> None:
            pass

    instance = SampleTool()
    expected_tools = {"tool_one", "tool_two"}

    assert set(instance.tool_registry.keys()) == expected_tools
    assert len(instance.tool_registry) == len(expected_tools)
    assert "regular_method" not in instance.tool_registry


def test_tool_name_length_exceeded():
    class DummyTool(BaseTool):
        @tool()
        def this_is_a_method_name_that_is_deliberately_long_enough_to_exceed_the_limit(self) -> None:
            pass

    instance = DummyTool()
    with pytest.raises(ToolNameLengthExceededError):
        instance.get_tool_schemas()


def test_generated_schema_matches_tool():
    class MathTool(BaseTool):
        @tool()
        def add(self, a: int, b: int = 10) -> int:
            """Add two numbers together."""
            return a + b

    schemas = MathTool().get_tool_schemas()
    assert len(schemas) == 1

    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "MathTool__add"
    assert schema["function"]["description"] == "Add two numbers together."

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert params["properties"]["a"]["type"] == "integer"
    assert "b" in params["properties"]
    assert params["properties"]["b"]["type"] == "integer"
    assert params["properties"]["b"].get("default") == 10
    assert params["required"] == ["a"]


def test_tool_schema_with_custom_typing():
    from enum import Enum
    from typing import Literal
    from pydantic import BaseModel, Field

    class RoleEnum(str, Enum):
        ADMIN = "admin"
        USER = "user"

    class Profile(BaseModel):
        bio: str
        age: int = Field(ge=0)

    class CustomTypeTool(BaseTool):
        @tool()
        def process_user(
            self,
            role: RoleEnum,
            mode: Literal["fast", "slow"],
            profile: Profile,
        ) -> dict:
            """Process a user profile."""
            return {}

    schemas = CustomTypeTool().get_tool_schemas()
    assert len(schemas) == 1

    schema = schemas[0]
    assert schema["function"]["name"] == "CustomTypeTool__process_user"
    assert schema["function"]["description"] == "Process a user profile."

    params = schema["function"]["parameters"]
    assert params["type"] == "object"

    props = params["properties"]
    assert "role" in props
    assert "mode" in props
    assert "profile" in props

    assert props["mode"]["enum"] == ["fast", "slow"]
    assert set(params["required"]) == {"role", "mode", "profile"}

    defs = params.get("$defs", {})
    assert "RoleEnum" in defs
    assert defs["RoleEnum"]["enum"] == ["admin", "user"]
    assert "Profile" in defs
