from __future__ import annotations
import copy
from enum import Enum
from typing import Any, Callable, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.utils import get_generic_parameter, is_generic_list
from geneticengine.random.sources import RandomSource
from geneticengine.solutions.tree import GengyList
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator


T = TypeVar("T")


class VariationType(Enum):
    REPLACEMENT = 1
    INSERTION = 2
    DELETION = 3


class ListSizeBetween(MetaHandlerGenerator):
    """ListSizeBetween(a,b) restricts lists to be of length between a and b and
    implements a special list mutation.

    The list of options can be dynamically altered before the grammar extraction
        Set.__annotations__["set"] = Annotated[List[Type], ListSizeBetween(c,d)].
    The special list mutation entails three different alterations to the list in question: deletion of a random element;
        addition of a random element; and replacement of a random element.
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
    ):
        assert is_generic_list(base_type)
        inner_type = get_generic_parameter(base_type)
        size = random.randint(self.min, self.max)
        li = []
        for i in range(size):
            nv = rec(inner_type)
            li.append(nv)
        assert len(li) == size
        assert self.min <= len(li) <= self.max
        print("Hello", li)
        return GengyList(inner_type, li)

    def mutate(
        self,
        random: RandomSource,
        g: Grammar,
        random_node,
        base_type,
        current_node,
    ):
        options : list[VariationType] = []

        if len(current_node) > 0:
            options.append(VariationType.REPLACEMENT)
        if len(current_node) < self.max:
            options.append(VariationType.INSERTION)
        if len(current_node) > self.min:
            options.append(VariationType.DELETION)
        if options:
            # Prepare information
            depth = current_node.synthesis_context.depth
            element_type = base_type.__args__[0]
            current_node_cpy : list = copy.copy(current_node)

            # Apply mutations
            match random.choice(options):
                case VariationType.REPLACEMENT:
                    element_to_be_replaced = random.randint(0, len(current_node) - 1)
                    new_element = random_node(random, g, depth, element_type)
                    current_node_cpy[element_to_be_replaced] = new_element
                case VariationType.INSERTION:
                    new_element = random_node(random=random, grammar=g, max_depth=depth, starting_symbol=element_type)
                    current_node_cpy.append(new_element)
                case VariationType.DELETION:
                    pos = random.randint(0, len(current_node_cpy) - 1)
                    current_node_cpy.pop(pos)
            assert self.min <= len(current_node_cpy) <= self.max
            return GengyList(element_type, current_node_cpy)
        else:
            assert False

    def crossover(
        self,
        r: RandomSource,
        g: Grammar,
        options,
        arg,
        list_type,
        current_node,
    ):
        if not options or (len(current_node) < 2):
            assert self.min <= len(current_node) <= self.max
            return current_node
        n_elements_replaced = r.randint(1, len(current_node) - 1)
        big_enough_options = [getattr(o, arg) for o in options if len(getattr(o, arg)) >= n_elements_replaced]
        while not big_enough_options:
            if n_elements_replaced == 1:
                return GengyList(list_type, current_node)
            n_elements_replaced = r.randint(1, n_elements_replaced - 1)
            big_enough_options = [getattr(o, arg) for o in options if len(getattr(o, arg)) >= n_elements_replaced]
        option = big_enough_options[r.randint(0, len(big_enough_options) - 1)]

        # Always cut beginning as we do double crossovers,
        # first using one tree as the current node,
        # and then the second tree as current node.
        new_node = copy.deepcopy(option[0:n_elements_replaced]) + current_node[n_elements_replaced:]
        assert self.min <= len(new_node) <= self.max
        return GengyList(list_type, new_node)

    def __class_getitem__(cls, args):
        return ListSizeBetween(*args)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"


class ListSizeBetweenWithoutListOperations(MetaHandlerGenerator):
    """ListSizeBetweenWithoutListOperations(a,b) restricts lists to be of
    length between a and b.

    The list of options can be dynamically altered before the grammar extraction
        Set.__annotations__["set"] = Annotated[List[Type], ListSizeBetweenWithoutListOperations(c,d)]
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        assert is_generic_list(base_type)
        inner_type = get_generic_parameter(base_type)
        size = random.randint(self.min, self.max)
        li = []
        for i in range(size):
            nv = rec(inner_type)
            li.append(nv)
        return GengyList(inner_type, li)

    def __class_getitem__(cls, args):
        return ListSizeBetweenWithoutListOperations(*args)

    def __repr__(self):
        return f"ListSizeBetweenWithoutListOperations[{self.min}...{self.max}]"
