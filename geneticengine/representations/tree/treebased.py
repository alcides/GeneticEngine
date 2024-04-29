from __future__ import annotations

from typing import Any
from typing import TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import (
    BasicSynthesisDecider,
    GlobalSynthesisContext,
    LocalSynthesisContext,
    apply_constructor,
    create_node,
    wrap_result,
)
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.solutions.tree import GengyList, TreeNode
from geneticengine.grammar.utils import get_arguments
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import has_annotated_mutation
from geneticengine.exceptions import GeneticEngineError

T = TypeVar("T")


def random_node(
    random: RandomSource,
    grammar: Grammar,
    max_depth: int,
    starting_symbol: type[Any] | None = None,
):
    starting_symbol = starting_symbol if starting_symbol else grammar.starting_symbol
    return create_node(
        GlobalSynthesisContext(
            random=random,
            grammar=grammar,
            decider=BasicSynthesisDecider(random, grammar, max_depth=max_depth),
        ),  # TODO
        starting_symbol,
        context=LocalSynthesisContext(depth=0, nodes=0, expansions=0),
    )


def random_individual(
    r: RandomSource,
    g: Grammar,
    max_depth: int = 5,
) -> TreeNode:
    try:
        assert max_depth >= g.get_min_tree_depth()
    except AssertionError:
        if g.get_min_tree_depth() == 1000000:
            raise GeneticEngineError(
                f"""Grammar's minimal tree depth is {g.get_min_tree_depth()}, which is the default tree depth.
                 It's highly like that there are nodes of your grammar than cannot reach any terminal.""",
            )
        raise GeneticEngineError(
            f"""Cannot use complete grammar for individual creation. Max depth ({max_depth})
            is smaller than grammar's minimal tree depth ({g.get_min_tree_depth()}).""",
        )
    ind = random_node(r, g, max_depth, g.starting_symbol)
    assert isinstance(ind, TreeNode)
    return ind


def select_option(options: list[TreeNode], c: int) -> int:
    """Given a list of options and the selected node, it mutates the node in
    the right spot."""
    seen = 0
    for i, o in enumerate(options):
        if c < seen + o.gengy_weighted_nodes:
            return i
    assert False


def mutate(global_context: GlobalSynthesisContext, i: TreeNode, ty: type) -> TreeNode:
    """Generates all nodes that can be mutable in a program."""
    assert hasattr(i, "synthesis_context")
    node_to_mutate = global_context.decider.random_int(0, i.gengy_weighted_nodes + 1)
    if node_to_mutate == 0:
        # First, we start by deciding whether we should mutate the current node, or one of its children.
        return create_node(global_context, ty, i.synthesis_context)
    elif has_annotated_mutation(ty):
        # Secondly, if there is a custom mutation to apply, do it.
        return ty.__metadata__[0].mutate(  # type: ignore
            global_context.random,
            global_context.grammar,
            random_node,
            get_generic_parameter(ty),
            current_node=i,
        )
    else:
        # Otherwise, select a random field and change it.
        args = list(i.gengy_init_values)
        if not args:
            return i
        argtypes = get_arguments(type(i))
        assert len(args) == len(argtypes)

        index = select_option(args, node_to_mutate)
        args[index] = mutate(global_context, args[index], argtypes[index][1])
        v: Any
        if isinstance(i, GengyList):
            v = GengyList(i.typ, args)
        else:
            v = apply_constructor(type(i), args)
        assert isinstance(v, ty)
        return wrap_result(v, global_context, i.synthesis_context)


def tree_mutate(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    max_depth: int,
    target_type: type,
    depth_aware_mut: bool = False,
) -> Any:
    global_context: GlobalSynthesisContext = GlobalSynthesisContext(r, g, BasicSynthesisDecider(r, g, max_depth))

    new_tree = mutate(global_context, i, target_type)
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
    return relabeled_new_tree


def find_in_tree(ty: type, o: TreeNode):
    if hasattr(o, "gengy_types_this_way") and ty in o.gengy_types_this_way:
        vals = o.gengy_types_this_way[ty]
        return vals


def crossover(global_context: GlobalSynthesisContext, i: TreeNode, ty: type, other: TreeNode) -> TreeNode:
    """Generates all nodes that can be mutable in a program."""
    assert hasattr(i, "synthesis_context")
    node_to_mutate = global_context.decider.random_int(0, i.gengy_weighted_nodes + 1)
    if node_to_mutate == 0:
        # First, we start by deciding whether we should mutate the current node, or one of its children.
        options = find_in_tree(ty, other)
        if not options:
            return create_node(global_context, ty, i.synthesis_context)
        else:
            return global_context.decider.choose_options(options, i.synthesis_context)
    elif has_annotated_mutation(ty):
        # Secondly, if there is a custom mutation to apply, do it.
        return ty.__metadata__[0].crossover(  # type: ignore
            global_context.random,
            global_context.grammar,
            random_node,
            get_generic_parameter(ty),
            current_node=i,
            other_node=other,
        )
    else:
        # Otherwise, select a random field and change it.
        args = list(i.gengy_init_values)
        if not args:
            return i
        argtypes = get_arguments(type(i))
        assert len(args) == len(argtypes)

        index = select_option(args, node_to_mutate)
        args[index] = mutate(global_context, args[index], argtypes[index][1])
        v: Any
        if isinstance(i, GengyList):
            v = GengyList(i.typ, args)
        else:
            v = apply_constructor(type(i), args)
        return wrap_result(v, global_context, i.synthesis_context)


def tree_crossover(
    r: RandomSource,
    g: Grammar,
    p1: TreeNode,
    p2: TreeNode,
    max_depth: int,
    specific_type: type | None = None,
    depth_aware_co: bool = False,
) -> tuple[TreeNode, TreeNode]:
    """Given the two input trees [p1] and [p2], the grammar and the random
    source, this function returns two trees that are created by crossing over.

    [p1] and [p2].

    The first tree returned has [p1] as the base, and the second tree
    has [p2] as a base.
    """
    global_context: GlobalSynthesisContext = GlobalSynthesisContext(r, g, BasicSynthesisDecider(r, g, max_depth))
    return crossover(global_context, p1, g.starting_symbol, p2), crossover(global_context, p2, g.starting_symbol, p1)


class TreeBasedRepresentation(
    Representation[TreeNode, TreeNode],
    RepresentationWithMutation[TreeNode],
    RepresentationWithCrossover[TreeNode],
):
    """This class represents the tree representation of an individual.

    In this approach, the genotype and the phenotype are exactly the
    same.
    """

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
    ):
        self.grammar = grammar
        self.max_depth = max_depth

    def create_genotype(self, random: RandomSource, **kwargs) -> TreeNode:
        actual_depth = kwargs.get("depth", self.max_depth)
        return random_individual(random, self.grammar, max_depth=actual_depth)

    def genotype_to_phenotype(self, genotype: TreeNode) -> TreeNode:
        return genotype

    def mutate(self, random: RandomSource, internal: TreeNode, **kwargs) -> TreeNode:
        return tree_mutate(
            random,
            self.grammar,
            internal,
            max_depth=self.max_depth,
            target_type=self.grammar.starting_symbol,
        )

    def crossover(
        self,
        random: RandomSource,
        parent1: TreeNode,
        parent2: TreeNode,
        **kwargs,
    ) -> tuple[TreeNode, TreeNode]:
        return tree_crossover(random, self.grammar, parent1, parent2, self.max_depth)
