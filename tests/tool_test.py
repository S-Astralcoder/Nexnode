import pytest

from nexnode.exceptions import DuplicateToolError, ToolNameLengthExceededError
from nexnode.tools.base import BaseTool, tool


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
        def this_is_a_method_name_that_is_deliberately_long_enough_to_exceed_the_limit(
            self,
        ) -> None:
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


def test_execute_tool_error_handling():
    class ExecutionTool(BaseTool):
        @tool()
        def divide(self, a: int, b: int) -> float:
            return a / b

    tool_instance = ExecutionTool()

    is_err, result = tool_instance.execute_tool("divide", {"a": 10, "b": 2})
    assert is_err is False
    assert result == "5.0"

    is_err, err_msg = tool_instance.execute_tool("divide", {"a": 10, "b": 0})
    assert is_err is True
    assert "division by zero" in err_msg

    is_err, err_msg = tool_instance.execute_tool("non_existent_tool", {})
    assert is_err is True
    assert "non_existent_tool" in err_msg

    is_err, err_msg = tool_instance.execute_tool("divide", {"unexpected": 1})
    assert is_err is True
    assert "missing" in err_msg or "divide()" in err_msg

    is_err, err_msg = tool_instance.execute_tool(
        "divide", {"101_error": "malformed JSON input"}
    )
    assert is_err is True
    assert err_msg == '{"101_error": "malformed JSON input"}'


def test_basetool_direct_schema_methods_raise_not_implemented():
    mock_base = object.__new__(BaseTool)

    with pytest.raises(NotImplementedError):
        mock_base.get_tool_schemas()

    with pytest.raises(NotImplementedError):
        mock_base._build_function_schema(lambda: None)


def test_execute_tool_non_serializable_return():
    class UnserializableTool(BaseTool):
        @tool()
        def get_set(self) -> set:
            return {1, 2, 3}

    instance = UnserializableTool()
    is_err, err_msg = instance.execute_tool("get_set", {})
    assert is_err is True
    assert "not JSON serializable" in err_msg or "TypeError" in err_msg


def test_tool_schema_ignores_varargs_and_supports_unannotated_params():
    class FlexTool(BaseTool):
        @tool()
        def process(self, raw_input, count: int = 1, *args, **kwargs) -> str:
            return f"{raw_input}_{count}_{len(args)}_{len(kwargs)}"

    instance = FlexTool()
    schemas = instance.get_tool_schemas()
    assert len(schemas) == 1

    props = schemas[0]["function"]["parameters"]["properties"]
    required = schemas[0]["function"]["parameters"]["required"]

    assert "raw_input" in props
    assert "count" in props
    assert "args" not in props
    assert "kwargs" not in props
    assert required == ["raw_input"]

    is_err, result = instance.execute_tool(
        "process", {"raw_input": "hello", "count": 3}
    )
    assert is_err is False
    assert result == '"hello_3_0_0"'


def test_duplicate_tool_error_in_tool_registry(monkeypatch: pytest.MonkeyPatch):
    class DuplicatedToolClass(BaseTool):
        @tool()
        def sample(self) -> str:
            return "ok"

    monkeypatch.setattr(
        "nexnode.tools.base.dir",
        lambda obj: ["sample", "sample"],
        raising=False,
    )

    with pytest.raises(DuplicateToolError):
        DuplicatedToolClass()
