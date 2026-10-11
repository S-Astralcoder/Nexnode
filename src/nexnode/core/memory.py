from typing import ClassVar

from pydantic import BaseModel, Field, model_serializer

from ..exceptions import InvalidMessageSequenceError
from ..schema.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallResultMessage,
    UserMessage,
)

MessageType = SystemMessage | UserMessage | AssistantMessage | ToolCallResultMessage


class AgentMemory(BaseModel):
    _ALLOWED_TRANSITIONS: ClassVar[
        dict[type[MessageType] | None, list[type[MessageType]]]
    ] = {
        None: [SystemMessage, UserMessage],
        SystemMessage: [SystemMessage, UserMessage],
        UserMessage: [AssistantMessage],
        AssistantMessage: [ToolCallResultMessage, UserMessage],
        ToolCallResultMessage: [ToolCallResultMessage, UserMessage],
    }
    messages: list[MessageType] = Field(default_factory=list)

    @property
    def agent_memory(self) -> list[MessageType]:
        """Backward compatibility property for messages."""
        return self.messages

    @model_serializer
    def serialize_messages(self) -> list[MessageType]:
        return self.messages

    def add(self, message: MessageType) -> None:
        last_message_type = type(self.messages[-1]) if self.messages else None
        allowed_types = self._ALLOWED_TRANSITIONS.get(last_message_type, [])

        if not any(isinstance(message, msg_type) for msg_type in allowed_types):
            current_type_name = type(message).__name__
            prev_type_name = (
                last_message_type.__name__
                if last_message_type
                else "None (empty memory)"
            )
            allowed_names = [t.__name__ for t in allowed_types]
            raise InvalidMessageSequenceError(
                f"Invalid message sequence: cannot append '{current_type_name}' after '{prev_type_name}'. "
                f"Allowed next message types: {allowed_names}."
            )
        self.messages.append(message)

    def get_messages(self) -> list[MessageType]:
        return self.messages
