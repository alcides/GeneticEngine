from __future__ import annotations

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.tree.utils import GengyList
from geneticengine.core.utils import build_finalizers
from geneticengine.metahandlers.base import MetaHandlerGenerator


class ListSizeBetween(MetaHandlerGenerator):
    """
    ListSizeBetween(a,b) restricts lists to be of length between a and b and implements a special list mutation.
    The list of options can be dynamically altered before the grammar extraction (Set.__annotations__["set"] = Annotated[List[Type], ListSizeBetween(c,d)].
    The special list mutation entails three different alterations to the list in question: deletion of a random element; addition of a random element; and replacement of a random element.
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        ctx: dict[str, str],
    ):
        base_type = base_type.__args__[0]
        size = r.randint(self.min, self.max, str(base_type))
        fins = build_finalizers(lambda *x: rec(GengyList(base_type, x)), size)
        ident = ctx["_"]
        for i, fin in enumerate(fins):
            nctx = ctx.copy()
            nident = ident + "_" + str(i)
            nctx["_"] = nident
            new_symbol(base_type, fin, depth - 1, nident, nctx)

    def mutate(
        self,
        r: Source,
        g: Grammar,
        random_node,
        depth: int,
        base_type,
        method,
        current_node,
    ):
        mutation_method = r.randint(0, 2)
        if (mutation_method == 0) and (len(current_node) != self.min):  # del
            element_to_be_deleted = r.randint(0, len(current_node) - 1)
            current_node.remove(current_node[element_to_be_deleted])
            return current_node
        elif (mutation_method == 1) and (len(current_node) != self.max):  # add
            new_element = random_node(r, g, depth, base_type.__args__[0], method=method)
            current_node.append(new_element)
            return current_node
        else:  # replace
            element_to_be_replaced = r.randint(0, len(current_node) - 1)
            new_element = random_node(r, g, depth, base_type.__args__[0], method=method)
            current_node[element_to_be_replaced] = new_element
            return current_node

    def crossover(
        self,
        r: Source,
        g: Grammar,
        options,
        arg,
        list_type,
        current_node,
    ):
        if not options:
            return current_node
        crossover_method = r.randint(0, 1)
        n_elements_replaced = r.randint(1, len(current_node) - 1)
        big_enough_options = [
            getattr(o, arg)
            for o in options
            if len(getattr(o, arg)) >= n_elements_replaced
        ]
        while not big_enough_options:
            if n_elements_replaced == 1:
                return GengyList(list_type, current_node)
            n_elements_replaced = r.randint(1, n_elements_replaced - 1)
            big_enough_options = [
                getattr(o, arg)
                for o in options
                if len(getattr(o, arg)) >= n_elements_replaced
            ]
        option = big_enough_options[r.randint(0, len(big_enough_options) - 1)]

        if crossover_method == 0:  # cut beginning
            new_node = (
                option[0:n_elements_replaced] + current_node[n_elements_replaced:]
            )
            return GengyList(list_type, new_node)
        else:  # cut end
            new_node = (
                current_node[0:n_elements_replaced]
                + option[n_elements_replaced : len(current_node)]
            )
            return GengyList(list_type, new_node)

    def __class_getitem__(self, args):
        return ListSizeBetween(*args)

    def __repr__(self):
        return f"ListSizeBetween[{self.min}...{self.max}]"


class ListSizeBetweenWithoutListOperations(MetaHandlerGenerator):
    """
    ListSizeBetweenWithoutListOperations(a,b) restricts lists to be of length between a and b.
    The list of options can be dynamically altered before the grammar extraction (Set.__annotations__["set"] = Annotated[List[Type], ListSizeBetweenWithoutListOperations(c,d)].
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        ctx: dict[str, str],
    ):
        base_type = base_type.__args__[0]
        size = r.randint(self.min, self.max, str(base_type))
        fins = build_finalizers(lambda *x: rec(GengyList(base_type, x)), size)
        ident = ctx["_"]
        for i, fin in enumerate(fins):
            nctx = ctx.copy()
            nident = ident + "_" + str(i)
            nctx["_"] = nident
            new_symbol(base_type, fin, depth - 1, nident, nctx)

    def __class_getitem__(self, args):
        return ListSizeBetweenWithoutListOperations(*args)

    def __repr__(self):
        return f"ListSizeBetweenWithoutListOperations[{self.min}...{self.max}]"
