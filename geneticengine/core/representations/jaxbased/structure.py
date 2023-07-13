from typing import Any
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator, MutationOperator, Representation
from geneticengine.core.tree import TreeNode

from jax.tree_util import tree_flatten, tree_unflatten
from jax.tree_util import tree_structure
from jax import random



class JaxMutation(MutationOperator[Any]):

    def mutate(
        self,
        genotype: Any,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> TreeNode:
        assert isinstance(representation, JaxRepresentation)
        return genotype
    
class JaxCrossover(CrossoverOperator[Any]):
    def crossover(
        self,
        g1: Any,
        g2: Any,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[Any, Any]:
        assert isinstance(representation, JaxRepresentation)
        return (g1, g2)


class JaxRepresentation(Representation[Any, TreeNode]):
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
    ) -> Any:
        self.stru = tree_structure(self.grammar.starting_symbol)
        print(self.stru)
        key = random.PRNGKey(r.randint(0,100000))
        return random.uniform(key, shape=(1000,))

    def genotype_to_phenotype(self, genotype: Any) -> TreeNode:
        return tree_unflatten(self.stru, genotype)

    def phenotype_to_genotype(self, phenotype: Any) -> TreeNode:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        stru, tree = tree_flatten(phenotype)
        return tree

    def get_mutation(self) -> MutationOperator[TreeNode]:
        assert False

    def get_crossover(self) -> CrossoverOperator[TreeNode]:
        assert False
