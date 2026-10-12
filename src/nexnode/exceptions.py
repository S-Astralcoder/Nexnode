class NexnodeError(Exception):
    """Base exception class for all Nexnode errors."""


class DuplicateToolError(NexnodeError):
    """Raised when a tool with the same name is already registered."""


class ToolNameLengthExceededError(NexnodeError):
    """Raised when a combined tool name exceeds the allowed character limit."""


class InvalidMessageSequenceError(NexnodeError):
    """Raised when a message does not follow the allowed memory sequence order."""


class EmptyMemoryError(NexnodeError):
    """Raised when an operation requires messages but agent memory is empty."""


class InvalidToolClassError(NexnodeError):
    """Raised when an tool class isn't a subclass of base tool class."""


class DuplicateToolClassError(NexnodeError):
    """Raised when an tool class is relisted more than once."""


# This two exception will be automatically be caught and send to the agent as tool result


class ToolCallPermissionError(
    NexnodeError
):  # Special exception for before_tool_execution
    """Raised when an tool call by a agent is to be blocked or not allowed."""


class ToolCallResultPermissionError(
    NexnodeError
):  # Special exception for after_tool_execution
    """Raised when the result of the tool call shouldn't be accessed by the agent"""


class MaxIterationLimitError(NexnodeError):
    """Raised when max number of iterations is exceeded."""
