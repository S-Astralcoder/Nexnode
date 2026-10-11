class DuplicateToolName(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNameLimit(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
