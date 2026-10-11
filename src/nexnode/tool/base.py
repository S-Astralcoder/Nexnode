import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import create_model

from nexnode.exceptions import DuplicateToolError, ToolNameLengthExceededError


def tool():
    def decorator(func: Callable[..., Any]):
        func._is_tool = True  # pyrefly: ignore[missing-attribute]
        return func

    return decorator


class BaseTool:
    def __init__(self) -> None:
        self.tool_registry: dict[str, Callable] = {}
        self.tool_registry = self._discover_tools()

    def _discover_tools(self) -> dict[str, Callable]:
        tool_registry: dict[str, Callable] = {}
        for method_name in dir(self):
            if method_name.startswith("_") or method_name in (
                "tool_links",
                "tool_registry",
            ):
                continue
            method = getattr(self, method_name, None)
            if callable(method) and getattr(method, "_is_tool", False):
                if method_name in tool_registry:
                    raise DuplicateToolError(
                        f"Tool '{method_name}' is already registered in '{type(self).__name__}'."
                    )
                tool_registry[method_name] = method
        return tool_registry

    def _build_function_schema(self, func: Callable) -> dict[str, Any]:
        if self.__class__ is BaseTool:
            raise NotImplementedError(
                "Cannot generate tool schema directly from 'BaseTool'. Subclass 'BaseTool' and define tools with '@tool'."
            )

        name = f"{type(self).__name__}__{func.__name__}"

        if len(name) > 64:
            raise ToolNameLengthExceededError(
                f"Combined tool name '{name}' exceeds the 64-character limit ({len(name)}/64). "
                f"Shorten method '{func.__name__}' on '{type(self).__name__}'."
            )
        description = inspect.getdoc(func) or ""
        signature = inspect.signature(func)
        field: dict[str, Any] = {}

        for param_name, param in signature.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            annotation = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            if param.default == inspect.Parameter.empty:
                field[param_name] = (annotation, ...)
            else:
                field[param_name] = (annotation, param.default)

        DynamicModel = create_model(
            f"{name}_arguments",
            **field,
        )

        parameter_schema = DynamicModel.model_json_schema()

        parameter_schema.pop("title", None)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameter_schema,
            },
        }

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self.__class__ is BaseTool:
            raise NotImplementedError(
                "Cannot generate tool schema directly from 'BaseTool'. Subclass 'BaseTool' and define tools with '@tool'."
            )
        tool_schemas: list[dict[str, Any]] = []
        for method in self.tool_registry.values():
            tool_schemas.append(self._build_function_schema(func=method))
        return tool_schemas

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            return json.dumps(self.tool_registry[tool_name](**arguments))
        except Exception as e:  # noqa: BLE001
            return f"{e}"
