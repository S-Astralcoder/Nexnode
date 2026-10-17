from typing import Any, Literal

import httpx
from httpx._models import Response

from nexnode.core.memory import MessageType

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


class _BaseAgentNode:
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.tools = tools

        self.tool_class_links: dict[str, BaseTool] = self._build_tool_class_links()
        self.tools_schema: list[ToolDefinition] = self._collect_tools_schema()

        self.client = httpx.Client(
            base_url=self.config.base_url, timeout=self.config.timeout
        )

    def before_tool_execution(self, tool_call: ChatCompletionToolCall) -> None:
        pass

    def after_tool_execution(self, tool_call_result: ToolCallResultMessage) -> None:
        pass

    def _build_tool_class_links(self) -> dict[str, BaseTool]:
        tool_class_links: dict[str, BaseTool] = {}
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

    def _collect_tools_schema(self) -> list[ToolDefinition]:
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

    def _chat_completion(self) -> ChatCompletionResponse:
        if not self.memory.messages:
            raise EmptyMemoryError(
                "Cannot execute chat completion: agent memory is empty."
            )
        last_message: MessageType = self.memory.messages[-1]

        if not isinstance(last_message, (UserMessage, ToolCallResultMessage)):
            last_message_type: str = type(last_message).__name__
            raise InvalidMessageSequenceError(
                f"Cannot execute chat completion: the last message must be a 'UserMessage' or 'ToolCallResultMessage', "
                f"but found '{last_message_type}'."
            )

        header: dict[str, Any] = ChatCompletionHeader(
            authorization=self.config.api_key
        ).model_dump()

        payload: dict[str, Any] = ChatCompletionRequest(
            tools=self.tools_schema,
            **self.config.model_dump(),
            **self.memory.model_dump(),
        ).model_dump(exclude_none=True)

        response: Response = self.client.post(
            url="/chat/completions", headers=header, json=payload
        )

        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())


class BaseAgentNode(_BaseAgentNode):
    def _execute_tool_call(
        self, tool_call: ChatCompletionToolCall
    ) -> ToolCallResultMessage:
        function_class, function_name = (
            tool_call.function.name.split("__")
            if len(tool_call.function.name.split("__")) == 2
            else ["", tool_call.function.name]
        )
        arguments: dict[str, Any] = tool_call.function.parsed_arguments
        try:
            self.before_tool_execution(tool_call=tool_call)
        except ToolCallPermissionError as e:
            tool_call_result = ToolCallResultMessage(
                tool_call_id=tool_call.id, content=f"{e}"
            )
        else:
            tool_call_instance: BaseTool | None = self.tool_class_links.get(
                function_class
            )

            if (
                tool_call_instance is None
                or tool_call_instance.tool_registry.get(function_name) is None
            ):
                tool_call_result = ToolCallResultMessage(
                    tool_call_id=tool_call.id,
                    content=f"The given tool {tool_call.function.name} doesn't exist.",
                )
            else:
                _, result = tool_call_instance.execute_tool(
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
            response: ChatCompletionResponse = self._chat_completion()
            self.memory.add(response)
            tool_calls: list[ChatCompletionToolCall] | None = response.choices[
                0
            ].message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    tool_call_result: ToolCallResultMessage = self._execute_tool_call(
                        tool_call=tool_call
                    )
                    self.memory.add(tool_call_result)
                continue
            break
        else:
            raise MaxIterationLimitError(
                f"Max iteration limit {max_iteration} reached."
            )
        return response


class SequentialAgentNode(_BaseAgentNode):  # beta node design
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        tools: list[BaseTool] | None = None,
    ) -> None:
        super().__init__(config=config, memory=memory, tools=tools)

        self._internal_state: Literal[
            "AWAITING_INPUT",
            "AGENT_RESPONSE",
            "AWAITING_TOOL_EXECUTION",
            "TOOL_EXECUTION_SUCCESSFUL_NEXT",
            "TOOL_EXECUTION_ERROR_NEXT",
            "TOOL_EXECUTION_END",
            "TOOL_EXECUTION_PERMISSION_DENIED",
            "TOOL_EXECUTION_RESULT_BLOCKED",
        ] = "AWAITING_INPUT"  # if you don't know what your doing. For the love of good don't access or update the value.

        self._tool_execution_queue: list[
            ChatCompletionToolCall
        ]  # also same for this. don't modify, if you don't know what your doing.

    def _execute_tool_call(
        self, tool_call: ChatCompletionToolCall
    ) -> ToolCallResultMessage:
        function_class, function_name = (
            tool_call.function.name.split("__")
            if len(tool_call.function.name.split("__")) == 2
            else ["", tool_call.function.name]
        )
        arguments: dict[str, Any] = tool_call.function.parsed_arguments
        try:
            self.before_tool_execution(tool_call=tool_call)
        except ToolCallPermissionError as e:
            self._internal_state = "TOOL_EXECUTION_PERMISSION_DENIED"
            tool_call_result = ToolCallResultMessage(
                tool_call_id=tool_call.id, content=f"{e}"
            )
        else:
            tool_call_instance: BaseTool | None = self.tool_class_links.get(
                function_class
            )
            if (
                tool_call_instance is None
                or tool_call_instance.tool_registry.get(function_name) is None
            ):
                tool_call_result = ToolCallResultMessage(
                    tool_call_id=tool_call.id,
                    content=f"The given tool {tool_call.function.name} doesn't exist.",
                )
                self._internal_state = "TOOL_EXECUTION_ERROR_NEXT"
            else:
                error, result = tool_call_instance.execute_tool(
                    tool_name=function_name, arguments=arguments
                )

                tool_call_result = ToolCallResultMessage(
                    tool_call_id=tool_call.id, content=result
                )
                if error:
                    self._internal_state = "TOOL_EXECUTION_ERROR_NEXT"
                else:
                    self._internal_state = "TOOL_EXECUTION_SUCCESSFUL_NEXT"

        try:
            self.after_tool_execution(tool_call_result=tool_call_result)
        except ToolCallResultPermissionError as e:
            self._internal_state = "TOOL_EXECUTION_RESULT_BLOCKED"
            return ToolCallResultMessage(tool_call_id=tool_call.id, content=f"{e}")
        return tool_call_result

    def _chat_completion(self) -> ChatCompletionResponse:
        try:
            return super()._chat_completion()
        except InvalidMessageSequenceError as e:
            self._internal_state = "AWAITING_INPUT"
            raise InvalidMessageSequenceError(e)

    def run_step(self) -> ChatCompletionResponse | ToolCallResultMessage | None:
        if self._internal_state in (
            "AWAITING_INPUT",
            "TOOL_EXECUTION_END",
            "AGENT_RESPONSE",
        ):
            response: ChatCompletionResponse = self._chat_completion()
            self.memory.add(response)
            tool_calls: list[ChatCompletionToolCall] | None = response.choices[
                0
            ].message.tool_calls
            if tool_calls:
                self._internal_state = "AWAITING_TOOL_EXECUTION"
                self._tool_execution_queue = tool_calls
                return response
            else:
                self._internal_state = "AGENT_RESPONSE"
                return response
        elif self._internal_state in (
            "AWAITING_TOOL_EXECUTION",
            "TOOL_EXECUTION_SUCCESSFUL_NEXT",
            "TOOL_EXECUTION_ERROR_NEXT",
            "TOOL_EXECUTION_PERMISSION_DENIED",
            "TOOL_EXECUTION_RESULT_BLOCKED",
        ):
            if not self._tool_execution_queue:
                self._internal_state = "TOOL_EXECUTION_END"
                return
            tool_call: ChatCompletionToolCall = self._tool_execution_queue.pop(0)
            tool_call_result: ToolCallResultMessage = self._execute_tool_call(
                tool_call=tool_call
            )
            self.memory.add(tool_call_result)
            return tool_call_result
        else:
            raise NexnodeError(
                "A Unexpected Error Occurred that isn't accounted for. Please report it."
            )


# user message -> llm response -> repeat
#                              -> tool_execution -> repeat execution -> final result -> repeat
