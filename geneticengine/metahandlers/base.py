from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Protocol
from typing import Type

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source


class MetaHandlerGenerator(Protocol):
    """MetaHandlers are type refinements. They override the generation procedure of the base type."""

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
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


def is_metahandler(ty: type) -> bool:
    """
    Returns if type is a metahandler.
    AnnotatedType[int, IntRange(3,10)] is an example of a Metahandler.

    Verification is done using the __metadata__, which is the first argument of Annotated
    """
    return hasattr(ty, "__metadata__")
