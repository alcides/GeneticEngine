from dataclasses import dataclass
from typing import Annotated, Any, Callable


from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.random.sources import RandomSource


@dataclass
class Dependent(MetaHandlerGenerator):
    name: str
    callable: Callable[[Any], type]

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        t: Any = self.callable(dependent_values[self.name])
        v = rec(Annotated[base_type, t])
        return v

    def __hash__(self):
        return hash(self.__class__) + hash(self.name) + hash(id(self.callable))

    def get_dependencies(self):
        return self.name.split(",")
