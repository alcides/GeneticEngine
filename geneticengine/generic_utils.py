from typing import Generic, TypeVar


C = TypeVar("C")
KEY = ""

class GenericWrapper(Generic[C]):
    container: dict[str, C]

    def __init__(self, c: C):
        self.container = {KEY: c}

    def get(self) -> C:
        return self.container[KEY]
