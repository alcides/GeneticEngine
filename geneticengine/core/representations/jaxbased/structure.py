from typing import Any
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator, MutationOperator, Representation
from geneticengine.core.tree import TreeNode

from jax.tree_util import tree_flatten, tree_unflatten
from jax.tree_util import tree_structure


class SMTTreeBasedRepresentation(Representation[Any, TreeNode]):
    """This class represents the tree representation of an individual.

    In this approach, the genotype and the phenotype are exactly the
    same.
    """

    def __init__(self, grammar: Grammar, max_depth: int):
        super().__init__(grammar, max_depth)

    def create_individual(
        self,
        r: Source,
        depth: int | None = None,
        **kwargs,
    ) -> TreeNode:
        actual_depth = depth or self.max_depth

        stru = tree_structure(self.grammar.starting_symbol)
        
        return None

    def genotype_to_phenotype(self, genotype: Any) -> TreeNode:
        return tree_unflatten(genotype)

    def phenotype_to_genotype(self, phenotype: Any) -> TreeNode:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        return tree_flatten(phenotype)

    def get_mutation(self) -> MutationOperator[TreeNode]:
        assert False

    def get_crossover(self) -> CrossoverOperator[TreeNode]:
        assert False
