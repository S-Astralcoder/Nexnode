from typing import Literal

import httpx

from ..exceptions import (
    DuplicateToolClassError,
    EmptyMemoryError,
    InvalidMessageSequenceError,
    InvalidToolClassError,
    MaxIterationLimitError,
    NexnodeError,
    ToolCallPermissionError,
    ToolCallResultPermissionError,
)
from ..schema.messages import ToolCallResultMessage, UserMessage
from ..schema.request import ChatCompletionHeader, ChatCompletionRequest, ToolDefinition
from ..schema.response import ChatCompletionResponse, ChatCompletionToolCall
from ..tool.base import BaseTool
from .config import AgentConfig
from .memory import AgentMemory


class BaseAgentNode:
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.tools = tools

        self.tool_class_links: dict[str, BaseTool] = self.__build_tool_class_links()
        self.tools_schema: list[ToolDefinition] = self.__collect_tools_schema()

        self.client = httpx.Client(
            base_url=self.config.base_url, timeout=self.config.timeout
        )

    def __build_tool_class_links(self) -> dict[str, BaseTool]:
        tool_class_links = {}
        if self.tools is None:
            return tool_class_links
        for tool_class in self.tools:
            class_name = type(tool_class).__name__

            if not isinstance(tool_class, BaseTool) or class_name == BaseTool.__name__:
                raise InvalidToolClassError(
                    f"The class {class_name} isn't derived from parent class BaseTool."
                )

            if class_name in tool_class_links:
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

    def before_tool_execution(self, tool_call: ChatCompletionToolCall):
        pass

    def after_tool_execution(self, tool_call_result: ToolCallResultMessage):
        pass

    def _chat_completion(self) -> ChatCompletionResponse:
        if not self.memory.messages:
            raise EmptyMemoryError(
                "Cannot execute chat completion: agent memory is empty."
            )
        last_message = self.memory.messages[-1]

        if not isinstance(last_message, (UserMessage, ToolCallResultMessage)):
            last_message_type = type(last_message).__name__
            raise InvalidMessageSequenceError(
                f"Cannot execute chat completion: the last message must be a 'UserMessage', "
                f"but found '{last_message_type}'."
            )

        header = ChatCompletionHeader(authorization=self.config.api_key).model_dump()

        payload = ChatCompletionRequest(
            tools=self.tools_schema,
            **self.config.model_dump(),
            **self.memory.model_dump(),
        ).model_dump(exclude_none=True)

        response = self.client.post(
            url="/chat/completions", headers=header, json=payload
        )


        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())

    def __execute_tool_call(
        self, tool_call: ChatCompletionToolCall
    ) -> ToolCallResultMessage:
        function_class, function_name = tool_call.function.name.split("__")
        arguments = tool_call.function.parsed_arguments
        try:
            self.before_tool_execution(tool_call=tool_call)
        except ToolCallPermissionError as e:
            return ToolCallResultMessage(tool_call_id=tool_call.id, content=f"{e}")

        tool_call_instance = self.tool_class_links.get(function_class)

        if (
            tool_call_instance is None
            or tool_call_instance.tool_registry.get(function_name) is None
        ):
            return ToolCallResultMessage(
                tool_call_id=tool_call.id,
                content=f"The given tool {tool_call.function.name} doesn't exist.",
            )

        result = tool_call_instance.execute_tool(
            tool_name=function_name, arguments=arguments
        )

        tool_call_result = ToolCallResultMessage(
            tool_call_id=tool_call.id, content=result
        )
        try:
            self.after_tool_execution(tool_call_result=tool_call_result)
        except ToolCallResultPermissionError as e:
            return ToolCallResultMessage(tool_call_id=tool_call.id, content=f"{e}")

        return tool_call_result

    def send_prompt(
        self, prompt: str, max_iteration: int = 10
    ) -> ChatCompletionResponse:
        user_message = UserMessage(content=prompt)
        self.memory.add(message=user_message)
        for _ in range(max_iteration):
            response = self._chat_completion()
            self.memory.add(response)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call_result = self.__execute_tool_call(tool_call=tool_call)
                    self.memory.add(tool_call_result)
                continue
            break
        else:
            raise MaxIterationLimitError(
                f"Max iteration limit {max_iteration} reached."
            )
        return response


class SequentialAgentNode:
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tools: list | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.tools = tools

        self.tool_class_links: dict[str, BaseTool] = self.__build_tool_class_links()
        self.tools_schema: list[ToolDefinition] = self.__collect_tools_schema()

        self.client = httpx.Client(
            base_url=self.config.base_url, timeout=self.config.timeout
        )

        self.__internal_state: Literal[
            "AWAITING_INPUT",
            "AGENT_RESPONSE",
            "AWAITING_TOOL_EXECUTION",
            "TOOL_EXECUTION_SUCCESSFUL_NEXT",
            "TOOL_EXECUTION_ERROR_NEXT",
            "TOOL_EXECUTION_END",
        ] = "AWAITING_INPUT"  # if you don't know what your doing. For the love of good don't access or update the value.

        self.tool_execution_queue: list[ChatCompletionToolCall]

    def __build_tool_class_links(self) -> dict[str, BaseTool]:
        tool_class_links = {}
        if self.tools is None:
            return tool_class_links
        for tool_class in self.tools:
            class_name = type(tool_class).__name__

            if not isinstance(tool_class, BaseTool) or class_name == BaseTool.__name__:
                raise InvalidToolClassError(
                    f"The class {class_name} isn't derived from parent class BaseTool."
                )

            if class_name in tool_class_links:
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

    def __execute_tool_call(
        self, tool_call: ChatCompletionToolCall
    ) -> ToolCallResultMessage:
        function_class, function_name = tool_call.function.name.split("__")
        arguments = tool_call.function.parsed_arguments
        tool_call_instance = self.tool_class_links.get(function_class)
        if (
            tool_call_instance is None
            or tool_call_instance.tool_registry.get(function_name) is None
        ):
            self.__internal_state = "TOOL_EXECUTION_ERROR_NEXT"
            return ToolCallResultMessage(
                tool_call_id=tool_call.id,
                content=f"The given tool {tool_call.function.name} doesn't exist.",
            )

        result = tool_call_instance.execute_tool(
            tool_name=function_name, arguments=arguments
        )

        tool_call_result = ToolCallResultMessage(
            tool_call_id=tool_call.id, content=result
        )
        self.__internal_state = "TOOL_EXECUTION_SUCCESSFUL_NEXT"

        return tool_call_result

    def _chat_completion(self) -> ChatCompletionResponse:
        if not self.memory.messages:
            raise EmptyMemoryError(
                "Cannot execute chat completion: agent memory is empty."
            )
        last_message = self.memory.messages[-1]

        if not isinstance(last_message, (UserMessage, ToolCallResultMessage)):
            last_message_type = type(last_message).__name__
            raise InvalidMessageSequenceError(
                f"Cannot execute chat completion: the last message must be a 'UserMessage', "
                f"but found '{last_message_type}'."
            )

        header = ChatCompletionHeader(authorization=self.config.api_key).model_dump()

        payload = ChatCompletionRequest(
            tools=self.tools_schema,
            **self.config.model_dump(),
            **self.memory.model_dump(),
        ).model_dump(exclude_none=True)

        response = self.client.post(
            url="/chat/completions", headers=header, json=payload
        )

        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())

    def run_step(self) -> ChatCompletionResponse | ToolCallResultMessage | None:
        if self.__internal_state in (
            "AWAITING_INPUT",
            "TOOL_EXECUTION_END",
            "AGENT_RESPONSE",
        ):
            response = self._chat_completion()
            self.memory.add(response)
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                self.__internal_state = "AWAITING_TOOL_EXECUTION"
                self.tool_execution_queue = tool_calls
                return response
            else:
                self.__internal_state = "AGENT_RESPONSE"
                return response
        elif self.__internal_state in (
            "AWAITING_TOOL_EXECUTION",
            "TOOL_EXECUTION_SUCCESSFUL_NEXT",
            "TOOL_EXECUTION_ERROR_NEXT",
        ):
            if not self.tool_execution_queue:
                self.__internal_state = "TOOL_EXECUTION_END"
                return
            tool_call = self.tool_execution_queue.pop(0)
            tool_call_result = self.__execute_tool_call(tool_call=tool_call)
            self.memory.add(tool_call_result)
            return tool_call_result
        else:
            raise NexnodeError(
                "A Unexpected Error Occurred that isn't accounted for. Please report it."
            )


# user message -> llm response -> repeat
#                              -> tool_execution -> repeat execution -> final result -> repeat
