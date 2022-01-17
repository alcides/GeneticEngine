from typing import Any, Callable, Dict, Type, Protocol

from geneticengine.core.random.sources import Source

from geneticengine.core.grammar import Grammar


class MetaHandlerGenerator(Protocol):
    """MetaHandlers are type refinements. They override the generation procedure of the base type."""

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec: Callable[[int, Type], Any],
        depth: int,
        base_type,
        argname: str,
        context: Dict[str, Type],
    ):
        """
        Generates an instance of type base_type, according to some criterion.

        :param Source r: Random source for generation
        :param Grammar g: Grammar to follow in the generation
        :param Callable[[int, Type], Any] rec: The method to generate a new instance of type and maximum depth d
        :param int depth: the current depth budget
        :param Type base_type: The inner type being annotated
        :param str argname: The name of the field of the parent object which is being generated
        :param Dict[str, Type] context: The names and types of all fields in the parent object
        """
        ...
