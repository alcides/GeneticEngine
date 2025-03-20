from dataclasses import dataclass
from typing import Annotated, Any, Callable, Generator


from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.random.sources import RandomSource


@dataclass
class Parent(MetaHandlerGenerator):
    """The Parent(variable_names, func) function allows you to access the first
    matched value of variables from other classes based on the specified
    variable_names.

    Example:
        Class A:
            value_small : Annotated[int, IntRange(-100,0)]
            value_big : Annotated[int, IntRange(0,100)]
            value : B
        Class B:
            x : Annotated[int, Parent('value_small,value_big', lambda small, big: IntRange(small, big))]
    In this example, the variable x is accessing variables of the Class A by parent Metahandler.
    You can specify multiple names at the same time by separating them with commas.
    If no matching variable is found, that value will be None.
    In func, you must ensure that None is handled or guarantee that the result will never be None.
    If there are multiples parent classes matched it will choose the closest one.
    """
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
        values = [
            next((layer[name] for layer in reversed(parent_values) if name in layer), None)
            for name in self.get_parents()
        ]
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
        values = [dependent_values[name][1] for name in self.get_parents()]
        t: Any = self.callable(*values)
        v = rec(Annotated[base_type, t],dependent_values)
        return v

    def __hash__(self):
        return hash(self.__class__) + hash(self.name)

    def get_parents(self):
        return self.name.split(",")

@dataclass
class Parents(MetaHandlerGenerator):
    """The Parents(variable_names, func) function allows you to access all the
    matched values of variables from other classes based on the specified
    variable_names, returning them in a list.

    Example:
        Class A:
            value_small : Annotated[int, IntRange(-100,0)]
            value_big : Annotated[int, IntRange(0,100)]
            value : B
        Class B:
            x : Annotated[int, Parents('value_small,value_big', lambda small, big: IntRange(small[0], big[0]))]
    In this example, the variable x is accessing variables of the Class A by parents Metahandler.
    You can specify multiple names at the same time by separating them with commas.
    If no matching variable is found, that value will be a empty list.
    In func, you must ensure that empty list is handled or guarantee that the result will never be a empty list.
    """
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
        values = [
            [layer[name] for layer in reversed(parent_values) if name in layer]
            for name in self.get_parents()
        ]
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
        values = [dependent_values[name][1] for name in self.get_parents()]
        t: Any = self.callable(*values)
        v = rec(Annotated[base_type, t],dependent_values)
        return v

    def __hash__(self):
        return hash(self.__class__) + hash(self.name)

    def get_parents(self):
        return self.name.split(",")
