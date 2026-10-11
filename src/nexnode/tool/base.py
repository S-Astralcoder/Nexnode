import inspect
from collections.abc import Callable
from typing import Any

from pydantic import create_model

from nexnode.exceptions import DuplicateToolName


class ToolBase:
    @staticmethod
    def tool():
        def decorator(func: Callable[..., Any]):
            func._is_tool = True
            return func

        return decorator

    def __init__(self) -> None:
        self.tool_links: dict[str, Callable] = self.__create_tool_link()

    def __create_tool_link(self):
        tool_links: dict[str, Callable] = {}
        for method_name in dir(self):
            method = getattr(self, method_name)
            if hasattr(method, "_is_tool") and method._is_tool:
                if tool_links.get(method_name, False):
                    raise DuplicateToolName(
                        f"The tool {method_name} is already registered in this tool cluster."
                    )
                tool_links.setdefault(method_name, method)
        return tool_links

    def __create_function_openapi_schema(self, func: Callable):
        name = func.__name__
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

    def get_tool_schema(self):
        tool_schema : list[dict] = []
        for method in self.tool_links.values():
            tool_schema.append(self.__create_function_openapi_schema(func=method))
        return tool_schema




