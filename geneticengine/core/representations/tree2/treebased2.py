from __future__ import annotations
from abc import ABC, abstractmethod
import sys

from typing import Any
from typing import TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.utils import relabel_nodes
from geneticengine.core.representations.tree.utils import relabel_nodes_of_trees
from geneticengine.core.representations.tree2.initializations import apply_metahandler
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import get_arguments, is_generic_list
from geneticengine.core.utils import get_generic_parameter
from geneticengine.core.utils import is_abstract
from geneticengine.core.evaluators import Evaluator
from geneticengine.metahandlers.base import is_metahandler

T = TypeVar("T")


class SynthesisContext(ABC):
    @abstractmethod
    def select(self, random_source: Source, grammar: Grammar, symbol: type[Any], alternatives, depth: int) -> type[Any]:
        """Handles the selection of alternatives."""
        ...

    @abstractmethod
    def select_length(self, random_source: Source, grammar: Grammar, max_length: int, depth: int) -> int:
        """Handles the selection of length for lists."""
        # TODO: remove max_length
        ...

    @abstractmethod
    def get_synthesis_override(
        self,
        random_source: Source,
        grammar: Grammar,
        ty: type[Any],
        depth: int,
    ) -> Any | None:
        """Overrides the generation process, or returns None if not
        applicable."""
        return None


def random_node(
    random_source: Source,
    grammar: Grammar,
    synctx: SynthesisContext,
    symbol: type[Any],
    depth: int = 0,
):
    """Generates a random node using a tree-based representation."""
    node: Any
    override = synctx.get_synthesis_override(random_source, grammar, symbol, depth)
    if override is not None:
        node = override
    elif is_metahandler(symbol):
        h: dict[str, Any] = {"key": None}

        def save(v):
            h["key"] = v

        def wrapper_debug(type, finalizer, depth, key, root):
            x = random_node(random_source, grammar, synctx, type, depth)
            finalizer(x)

        apply_metahandler(
            random_source,
            grammar,
            save,
            wrapper_debug,
            depth,
            symbol,
            {"_": "key"},
        )  # TODO: arguments
        node = h["key"]
    elif is_generic_list(symbol):
        inner_type = get_generic_parameter(symbol)
        length: int = synctx.select_length(random_source, grammar, 100, depth)
        node = [random_node(random_source, grammar, synctx, inner_type, depth + 1) for _ in range(length)]
    else:
        alternatives = grammar.alternatives[symbol]
        selected_type = synctx.select(random_source, grammar, symbol, alternatives, depth)
        parameters = get_arguments(selected_type)
        # TODO: Consider randomizing the order of parameters.
        arguments = [random_node(random_source, grammar, synctx, p, depth + 1) for (_, p) in parameters]
        node = selected_type(*arguments)

    relabel_nodes_of_trees(node, grammar)
    return node


class MutationContext(SynthesisContext):
    @abstractmethod
    def should_replace_current_node(
        self,
        random_source: Source,
        grammar: Grammar,
        root: TreeNode,
        ty: type[Any],
        depth: int,
    ) -> bool:
        """Decides if the current node should be replaced."""
        ...

    @abstractmethod
    def select_branch(self, random_source: Source, grammar: Grammar, root: TreeNode, ty: type[Any], depth: int):
        """Decides which of the branches of a tree will be replaced."""
        ...

    @abstractmethod
    def get_mutation_override(
        self,
        random_source: Source,
        grammar: Grammar,
        ty: type,
        node: Any,
        depth: int,
    ) -> Any | None:
        ...


def mutate(
    random_source: Source,
    grammar: Grammar,
    mutctx: MutationContext,
    root: TreeNode,
    ty: type[Any],
    depth: int = 0,
):
    """Mutates a node."""
    if mutctx.should_replace_current_node(random_source, grammar, root, ty, depth):
        mut = mutctx.get_mutation_override(random_source, grammar, ty, root, depth)
        if mut:
            return mut
        new_node = random_node(random_source, grammar, mutctx, ty, depth)
    else:
        parameters = get_arguments(ty)
        index_to_be_explored = mutctx.select_branch(random_source, grammar, root, ty, depth)
        arguments = [
            getattr(root, name)
            if i != index_to_be_explored
            else mutate(random_source, grammar, mutctx, getattr(root, name), ty, depth + 1)
            for (i, (name, ty)) in enumerate(parameters)
        ]
        new_node = ty(*arguments)

    relabel_nodes_of_trees(new_node, grammar)
    return new_node


def find_in_tree(g: Grammar, ty: type, o: TreeNode, max_depth: int):
    is_abs = is_abstract(ty)
    if hasattr(o, "gengy_types_this_way"):
        for t in o.gengy_types_this_way:

            def is_valid(node):
                _, depth, _, _ = relabel_nodes(node, g)

                if is_abs and g.expansion_depthing:
                    depth += g.abstract_dist_to_t[ty][t]

                return depth <= max_depth

            if ty in t.__bases__:
                vals = o.gengy_types_this_way[t]
                if vals:
                    yield from filter(is_valid, vals)


def find_in_tree_exact(g: Grammar, ty: type, o: TreeNode, max_depth: int):
    if hasattr(o, "gengy_types_this_way") and ty in o.gengy_types_this_way:
        vals = o.gengy_types_this_way[ty]
        if vals:

            def is_valid(node):
                _, depth, _, _ = relabel_nodes(node, g)
                return depth <= max_depth

            yield from filter(is_valid, vals)


class BasicTreeBased2SynthesisContext(MutationContext):
    def select(self, random_source: Source, grammar: Grammar, symbol: type[Any], alternatives, depth: int) -> type[Any]:
        """Handles the selection of alternatives."""
        return random_source.choice(alternatives)

    def select_length(self, random_source: Source, grammar: Grammar, max_length: int, depth: int) -> int:
        """Handles the selection of length for lists."""
        return random_source.randint(0, max_length)

    def get_synthesis_override(
        self,
        random_source: Source,
        grammar: Grammar,
        ty: type[Any],
        depth: int,
    ) -> Any | None:
        """Overrides the generation process, or returns None if not
        applicable."""
        if ty is int:
            max_int = sys.maxsize
            min_int = -sys.maxsize
            val = random_source.normalvariate(0, 100)
            val = round(val)
            val = max(min(val, max_int), min_int)
            return val
        elif ty is bool:
            return random_source.random_bool()
        elif ty is float:
            max_float = sys.float_info.max
            min_float = -sys.float_info.max
            val = random_source.normalvariate(0, 100)
            valf = max(min(val, max_float), min_float)
            return valf
        else:
            return None

    def should_replace_current_node(
        self,
        random_source: Source,
        grammar: Grammar,
        root: TreeNode,
        ty: type[Any],
        depth: int,
    ) -> bool:
        """Decides if the current node should be replaced."""
        return random_source.randint(0, root.gengy_weighted_nodes) == 0

    def select_branch(self, random_source: Source, grammar: Grammar, root: TreeNode, ty: type[Any], depth: int):
        """Decides which of the branches of a tree will be replaced."""
        arguments = get_arguments(type(root))
        return random_source.randint(0, len(arguments))

    def get_mutation_override(
        self,
        random_source: Source,
        grammar: Grammar,
        ty: type,
        node: Any,
        depth: int,
    ) -> Any | None:
        if ty is int:
            return node + random_source.normalvariate(0, abs(node) * 0.1)
        else:
            return None


class DefaultTB2Mutation(MutationOperator[TreeNode]):
    """Selects a random node, and generates a new replacement."""

    mutation_context: MutationContext

    def __init__(self, mutation_context: MutationContext):
        """
        Args:
            mutctx (MutationContext): a mutation context.

        """
        self.mutation_context = mutation_context

    def mutate(
        self,
        genotype: TreeNode,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> TreeNode:
        assert isinstance(representation, TreeBased2Representation)
        return mutate(
            random_source,
            representation.grammar,
            self.mutation_context,
            genotype,
            representation.grammar.starting_symbol,
        )


class DefaultTB2Crossover(CrossoverOperator[TreeNode]):
    """Selects a sub-tree from one parent and replaces it with a compatible
    tree from the other parent."""

    mutation_context: MutationContext

    def __init__(self, mutation_context: MutationContext):
        """
        Args:
            mutctx (MutationContext): a mutation context.

        """
        self.mutation_context = mutation_context

    def crossover(
        self,
        g1: TreeNode,
        g2: TreeNode,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[TreeNode, TreeNode]:
        assert isinstance(representation, TreeBased2Representation)
        return (None, None)  # TODO


class TreeBased2Representation(Representation[TreeNode, TreeNode]):
    """This class represents the tree representation of an individual.

    In this approach, the genotype and the phenotype are exactly the
    same.
    """

    synctx: MutationContext

    def __init__(self, grammar: Grammar, max_depth: int):
        super().__init__(grammar, max_depth)
        self.synctx = BasicTreeBased2SynthesisContext()

    def create_individual(
        self,
        random_source: Source,
        max_depth: int,
        **kwargs,
    ) -> TreeNode:
        return random_node(random_source, self.grammar, self.synctx, self.grammar.starting_symbol, 0)

    def genotype_to_phenotype(self, genotype: TreeNode) -> TreeNode:
        return genotype

    def phenotype_to_genotype(self, phenotype: Any) -> TreeNode:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        return relabel_nodes_of_trees(
            phenotype,
            self.grammar,
        )

    def get_mutation(self) -> MutationOperator[TreeNode]:
        return DefaultTB2Mutation(self.synctx)

    def get_crossover(self) -> CrossoverOperator[TreeNode]:
        return DefaultTB2Crossover(self.synctx)
