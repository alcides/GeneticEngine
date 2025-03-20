from dataclasses import dataclass
from typing import Annotated, Any, Callable, Generator


from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.random.sources import RandomSource


@dataclass
class Dependent(MetaHandlerGenerator):
    name: str
    callable: Callable[[Any], type]

    def validate(self, v) -> bool:
        return True

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        values = [dependent_values[name] for name in self.get_dependencies()]
        t: Any = self.callable(*values)
        v = rec(Annotated[base_type, t])
        return v

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        values = [dependent_values[name][1] for name in self.get_dependencies()]
        t: Any = self.callable(*values)
        v = rec(Annotated[base_type, t],dependent_values)
        return v

    def __hash__(self):
        return hash(self.__class__) + hash(self.name) + hash(id(self.callable))

    def get_dependencies(self):
        return self.name.split(",")
