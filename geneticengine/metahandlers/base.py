from typing import Any, Callable, Dict, Type, Protocol

from geneticengine.core.random.sources import Source

from geneticengine.core.grammar import Grammar


class MetaHandlerGenerator(Protocol):
    def generate(
        self,
        r: Source,
        g: Grammar,
        wrapper: Callable[[Any, str, int, Callable[[int], Any]], Any],
        rec: Any,
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        ...
