from __future__ import annotations

from typing import Any
from typing import TypeVar

from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.base import MetaHandlerGenerator
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import (
    GlobalSynthesisContext,
    LocalSynthesisContext,
    SynthesisDecider,
    apply_constructor,
    create_node,
    number_of_nodes,
    wrap_result,
)
from geneticengine.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.solutions.tree import GengyList, TreeNode
from geneticengine.grammar.utils import get_arguments, is_metahandler
from geneticengine.grammar.utils import get_generic_parameter
from geneticengine.grammar.utils import has_annotated_mutation

T = TypeVar("T")


def random_node(
    random: RandomSource,
    grammar: Grammar,
    starting_symbol: type[Any],
    decider: SynthesisDecider,
):
    assert isinstance(decider, SynthesisDecider)
    return create_node(
        GlobalSynthesisContext(
            random=random,
            grammar=grammar,
            decider=decider,
        ),
        starting_symbol,
        context=LocalSynthesisContext(depth=0, nodes=0, expansions=0, dependent_values={}, parent_values=[]),
    )


def random_tree(
    random: RandomSource,
    grammar: Grammar,
    decider: SynthesisDecider,
) -> TreeNode:
    ind = random_node(random, grammar, grammar.starting_symbol, decider=decider)
    assert isinstance(ind, grammar.starting_symbol)
    assert isinstance(ind, TreeNode)
    return ind


def get_weighted_nodes(e: Any) -> int:
    if hasattr(e, "gengy_weighted_nodes"):
        return e.gengy_weighted_nodes
    return 1


def find_in_tree(ty: type, o: TreeNode):
    if hasattr(o, "gengy_types_this_way") and ty in o.gengy_types_this_way:
        vals = o.gengy_types_this_way[ty]
        return vals


def mutate(
    global_context: GlobalSynthesisContext,
    i: TreeNode,
    ty: type,
    dependent_values: dict[str, Any] | None = None,
    source_material: list[TreeNode] | None = None,
) -> TreeNode:
    """Generates all nodes that can be mutable in a program."""
    if not hasattr(i, "synthesis_context"):
        node_to_mutate = 0
    else:
        node_to_mutate = global_context.decider.random_int(0, i.gengy_weighted_nodes + 1)

    # First, we start by deciding whether we should mutate the current node, or one of its children.
    if node_to_mutate == 0 or not hasattr(i, "gengy_synthesis_context"):
        if has_annotated_mutation(ty):
            # Secondly, if there is a custom mutation to apply, do it.
            assert hasattr(ty, "__metadata__")
            mh: MetaHandlerGenerator = ty.__metadata__[0]

            return mh.mutate(  # type: ignore
                global_context.random,
                global_context.grammar,
                random_node,
                get_generic_parameter(ty),
                current_node=i,
            )
        elif not hasattr(i, "gengy_synthesis_context"):
            return create_node(global_context, ty, LocalSynthesisContext(0, 0, 0, {},[]), dependent_values, [])
        else:
            options = []
            if source_material and source_material[0] is not None:
                options = find_in_tree(ty, source_material[0])  # TODO: add support for multiple material
            if not options:
                return create_node(global_context, ty, i.gengy_synthesis_context, dependent_values)
            else:
                return global_context.decider.choose_options(options, i.gengy_synthesis_context)
    else:
        # Otherwise, select a random field and change it.

        nargs = []
        dependent_values = {}
        ctx = i.gengy_synthesis_context
        nctx = LocalSynthesisContext(ctx.depth + 1, ctx.nodes + 1, ctx.expansions + 1, dependent_values={}, parent_values=[])

        seen = 0
        mutated: list[str] = []

        for arg, (parn, part) in zip(i.gengy_init_values, get_arguments(type(i))):

            should_mutate = False
            if seen < node_to_mutate <= seen + get_weighted_nodes(arg):
                should_mutate = True
            elif is_metahandler(part):
                mh: MetaHandlerGenerator = part.__metadata__[0]  # type: ignore
                dependencies = mh.get_dependencies()
                for m in mutated:
                    if m in dependencies:
                        should_mutate = True

            if should_mutate:
                mutated.append(parn)
                narg = mutate(global_context, arg, part, dependent_values=dependent_values)
            else:
                narg = arg
            seen += get_weighted_nodes(arg)
            nargs.append(narg)
            dependent_values[parn] = narg
            nctx.nodes += number_of_nodes(arg)
        v: Any
        if isinstance(i, GengyList):
            v = GengyList(i.typ, nargs)
        else:
            v = apply_constructor(type(i), nargs)

        return wrap_result(v, global_context, i.gengy_synthesis_context)


def tree_mutate(
    r: RandomSource,
    g: Grammar,
    i: TreeNode,
    target_type: type,
    decider: SynthesisDecider,
) -> Any:
    global_context: GlobalSynthesisContext = GlobalSynthesisContext(r, g, decider)

    new_tree = mutate(global_context, i, target_type, dependent_values={})
    relabeled_new_tree = relabel_nodes_of_trees(new_tree, g)
    return relabeled_new_tree


def tree_crossover(
    r: RandomSource,
    g: Grammar,
    p1: TreeNode,
    p2: TreeNode,
    decider: SynthesisDecider,
) -> tuple[TreeNode, TreeNode]:
    """Given the two input trees [p1] and [p2], the grammar and the random
    source, this function returns two trees that are created by crossing over.

    [p1] and [p2].

    The first tree returned has [p1] as the base, and the second tree
    has [p2] as a base.
    """
    global_context: GlobalSynthesisContext = GlobalSynthesisContext(r, g, decider)
    return mutate(global_context, p1, g.starting_symbol, source_material=[p2]), mutate(
        global_context,
        p2,
        g.starting_symbol,
        source_material=[p1],
    )


class TreeBasedRepresentation(
    Representation[TreeNode, TreeNode],
    RepresentationWithMutation[TreeNode],
    RepresentationWithCrossover[TreeNode],
):
    """This class represents the tree representation of an individual.

    In this approach, the genotype and the phenotype are exactly the
    same.
    """

    def __init__(self, grammar: Grammar, decider: SynthesisDecider):
        self.grammar = grammar
        self.decider = decider
        assert isinstance(decider, SynthesisDecider)

    def create_genotype(self, random: RandomSource, **kwargs) -> TreeNode:
        decider = kwargs.get("decider", self.decider)
        return random_tree(random, self.grammar, decider=decider)

    def genotype_to_phenotype(self, genotype: TreeNode) -> TreeNode:
        return genotype

    def mutate(self, random: RandomSource, genotype: TreeNode, **kwargs) -> TreeNode:
        decider = kwargs.get("decider", self.decider)
        return tree_mutate(
            random,
            self.grammar,
            genotype,
            decider=decider,
            target_type=self.grammar.starting_symbol,
        )

    def crossover(
        self,
        random: RandomSource,
        parent1: TreeNode,
        parent2: TreeNode,
        **kwargs,
    ) -> tuple[TreeNode, TreeNode]:
        decider = kwargs.get("decider", self.decider)
        return tree_crossover(random, self.grammar, parent1, parent2, decider=decider)
