import httpx

from ..exceptions import (
    DuplicateToolClassError,
    EmptyMemoryError,
    InvalidMessageSequenceError,
    InvalidToolClassError,
)
from ..schema.messages import UserMessage
from ..schema.request import ChatCompletionHeader, ChatCompletionRequest, ToolDefinition
from ..schema.response import ChatCompletionResponse
from ..tool.base import BaseTool
from .config import AgentConfig
from .memory import AgentMemory


class BaseAgentNode:
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tools: list | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.tools = tools

        self.tool_class_links: dict = self.__build_tool_class_links()
        self.tools_schema: list[ToolDefinition] = self.__collect_tools_schema()

        self.client = httpx.Client(base_url=self.config.base_url)

    def __build_tool_class_links(self) -> dict:
        tool_class_links = {}
        if self.tools is None:
            return tool_class_links
        for tool_class in self.tools:
            class_name = type(tool_class).__name__

            if not isinstance(tool_class, BaseTool) or class_name == BaseTool.__name__:
                raise InvalidToolClassError(
                    f"The class {class_name} isn't derived from parent class BaseTool."
                )

            if class_name in self.tool_class_links:
                raise DuplicateToolClassError(
                    f"The class {class_name} can't be added more than once."
                )

            tool_class_links.setdefault(class_name, tool_class)
        return tool_class_links

    def __collect_tools_schema(self) -> list[ToolDefinition]:
        tools_schema_list: list[ToolDefinition] = []
        if self.tools is None:
            return tools_schema_list

        for tool_class in self.tools:
            tools_schema_list.extend(
                [
                    ToolDefinition(**tool_schema)
                    for tool_schema in tool_class.get_tool_schemas()
                ]
            )

        return tools_schema_list

    def before_tool_execution(self):
        pass

    def after_tool_execution(self):
        pass

    def _chat_completion(self) -> ChatCompletionResponse:
        if not self.memory.messages:
            raise EmptyMemoryError(
                "Cannot execute chat completion: agent memory is empty."
            )
        last_message = self.memory.messages[-1]

        if not isinstance(last_message, UserMessage):
            last_message_type = type(last_message).__name__
            raise InvalidMessageSequenceError(
                f"Cannot execute chat completion: the last message must be a 'UserMessage', "
                f"but found '{last_message_type}'."
            )

        header = ChatCompletionHeader(authorization=self.config.api_key).model_dump()

        payload = ChatCompletionRequest(
            **self.config.model_dump(),
            **self.memory.model_dump(),
            tools=self.tools_schema,
        ).model_dump()

        response = self.client.post(
            url="/v1/chat/completions", headers=header, json=payload
        )

        response.raise_for_status()

        return ChatCompletionResponse.model_validate_json(response.json())
    

    def send_prompt(self, prompt: str):
        user_message = UserMessage(content=prompt)
        self.memory.add(message=user_message)
