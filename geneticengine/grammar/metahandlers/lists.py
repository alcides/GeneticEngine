from __future__ import annotations
import copy
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.utils import get_generic_parameter, is_generic_list
from geneticengine.random.sources import RandomSource
from geneticengine.representations.tree.initializations import SynthesisDecider
from geneticengine.solutions.tree import GengyList, TreeNode
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

    def validate(self, v) -> bool:
        return self.min <= len(v) <= self.max

    def generate(
        self,
        random: RandomSource,
        grammar: Grammar,
        base_type: type,
        rec: Callable[[type[T]], T],
        dependent_values: dict[str, Any],
        parent_values: list[dict[str, Any]],
    ):
        assert is_generic_list(base_type)
        inner_type = get_generic_parameter(base_type)
        size = random.randint(self.min, self.max)
        li : list[Any] = []
        parent_values[-1][next(reversed(parent_values[-1]))] = li
        for i in range(size):
            nv = rec(inner_type)
            li.append(nv)
        assert len(li) == size
        assert self.min <= len(li) <= self.max
        return GengyList(inner_type, li)

    def mutate(
        self,
        random: RandomSource,
        g: Grammar,
        random_node,
        base_type,
        current_node,
    ):
        options: list[VariationType] = []
        assert isinstance(current_node, GengyList)
        if len(current_node) > 0:
            options.append(VariationType.REPLACEMENT)
        if len(current_node) < self.max:
            options.append(VariationType.INSERTION)
        if len(current_node) > self.min:
            options.append(VariationType.DELETION)
        if options:
            # Prepare information
            assert isinstance(current_node, TreeNode)
            depth = current_node.gengy_synthesis_context.depth
            element_type = base_type.__args__[0]
            assert isinstance(current_node, GengyList)
            current_node_cpy: list = copy.copy(list(current_node.gengy_init_values))
            decider: SynthesisDecider = current_node.gengy_global_synthesis_context.decider  # type:ignore
            # TODO: Dependent Types

            # Apply mutations
            match random.choice(options):
                case VariationType.REPLACEMENT:
                    element_to_be_replaced = random.randint(0, len(current_node_cpy) - 1)
                    new_element = random_node(random, g, depth, element_type)
                    current_node_cpy[element_to_be_replaced] = new_element
                case VariationType.INSERTION:
                    new_element = random_node(random=random, grammar=g, starting_symbol=element_type, decider=decider)
                    current_node_cpy.append(new_element)
                case VariationType.DELETION:
                    pos = random.randint(0, len(current_node_cpy) - 1)
                    current_node_cpy.pop(pos)
            assert self.min <= len(current_node_cpy) <= self.max
            return current_node.new_like(current_node_cpy)
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
        return current_node.new_like(new_node)

    def iterate(
        self,
        base_type: type,
        combine_lists: Callable[[list[type]], Generator[Any, Any, Any]],
        rec: Any,
        dependent_values: dict[str, Any],
    ):
        assert is_generic_list(base_type)
        inner_type = get_generic_parameter(base_type)
        for length in range(self.min, self.max + 1):
            for li in combine_lists([inner_type for _ in range(length)]):
                yield li

    def __class_getitem__(cls, args):
        return ListSizeBetween(*args)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"
