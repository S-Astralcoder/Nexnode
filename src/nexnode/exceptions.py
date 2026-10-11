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
