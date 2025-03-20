from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource


T = TypeVar("T")


class SynthesisException(Exception):
    pass


class MetaHandlerGenerator(ABC):
    """MetaHandlers are type refinements.

    They override the generation procedure of the base type.
    """

    @abstractmethod
    def validate(self, Any) -> bool: ...

    @abstractmethod
    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ) -> Any:
        """Generates an instance of type base_type, according to some
        criterion.

        :param Source r: Random source for generation
        :param Grammar g: Grammar to follow in the generation :param
            Callable[[int, Type], Any] rec: The method to generate a new
            instance of type and maximum depth d
        :param int depth: the current depth budget
        :param Type base_type: The inner type being annotated
        :param str argname: The name of the field of the parent object
            which is being generated :param Dict[str, Type] context: The
            names and types of all fields in the parent object :param
            Dict[str, Type] dependent_values: The names and values of
            all previous fields in the parent object
        """
        ...

    def get_dependencies(self):
        return []
