from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import CrossoverOperator
from geneticengine.core.representations.api import MutationOperator
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.initializations import (
    InitializationMethodType,
)
from geneticengine.core.representations.tree.initializations import pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.tree import TreeNode
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.utils import get_arguments, is_terminal
from geneticengine.metahandlers.base import is_metahandler

MAX_VALUE = 10000000
GENE_LENGTH = 256


@dataclass
class Genotype:
    dna: list[int]


def random_individual(
    r: Source,
    g: Grammar,
    depth: int = 5,
    starting_symbol: Any = None,
) -> Genotype:
    return Genotype([r.randint(0, MAX_VALUE) for _ in range(GENE_LENGTH)])


def mutate(r: Source, g: Grammar, ind: Genotype, max_depth: int) -> Genotype:
    rindex = r.randint(0, 255)
    clone = [i for i in ind.dna]
    clone[rindex] = r.randint(0, 10000)
    return Genotype(clone)


def crossover(
    r: Source,
    g: Grammar,
    p1: Genotype,
    p2: Genotype,
    max_depth: int,
) -> tuple[Genotype, Genotype]:
    rindex = r.randint(0, 255)
    c1 = p1.dna[:rindex] + p2.dna[rindex:]
    c2 = p2.dna[:rindex] + p1.dna[rindex:]
    return (Genotype(c1), Genotype(c2))

def phenotype_to_genotype(
        g: Grammar,
        p: TreeNode,
        depth: int,
) -> Genotype:
    """
    A imperfect method that tries to reconstruct the genotype from the phenotype. It is not possible to reconstruct integers and floats, do to the way integers and floats are constructed using a normalvariate function. However, the tree structure and node types are preserved. Therefore, this method can be used in the initialization process of trees.
    """
    dna = [0] # Not sure why this number is necessary yet, but if add, this works. The number value doesn't seem to affect the create tree.
    non_terminals = g.non_terminals

    def filter_choices(possible_choices: list[type], depth):
        valid_productions = [vp for vp in possible_choices if g.get_distance_to_terminal(vp) <= depth]
        return valid_productions

    def find_choices_super(t: TreeNode, ttype, depth: int):
        choices = []
        supers = type(t).__mro__
        ttype_index = supers.index(ttype)
        i = 1
        while i <= ttype_index:
            choice = filter_choices(g.alternatives[ttype], depth).index(supers[ttype_index - i])
            choices.append(choice)
            ttype = g.alternatives[ttype][choice]
            i += 1
        return choices
    
    def randint_inverse(min: int, max: int, v: int) -> int:
        assert v >= min
        assert v <= max
        return v - min

    def random_float_inverse(min: float, max: float, v: float) -> int:
        k = round((max - min) / (v - min))
        return randint_inverse(1, MAX_VALUE, k)
    
    def normalvariate_inverse(mean: float, sigma: float, v: float):
        z0 = (v - mean)/sigma
        u1 = 0.5
        z1 = math.sqrt(-2.0 * math.log(u1))
        u2 = math.acos(z0/z1)/(2 * math.pi)
        return random_float_inverse(0.0, 1.0, u1), random_float_inverse(0.0, 1.0, u2)
    
    def apply_inverse_metahandler(
        g: Grammar,
        depth: int,
        rec,
        base_type,
        instance,
    ) -> Any:
        """This method applies the inverse metahandler to use a custom generator for things
        of a given type.

        As an example, AnnotatedType[int, IntRange(3,10)] will use the
        IntRange.generate(r, recursive_generator). The generator is the
        annotation on the type ("__metadata__").
        """
        metahandler = base_type.__metadata__[0]
        return metahandler.inverse_generate(
            g,
            depth,
            rec,
            base_type,
            instance,
        ) 
    
    def reconstruct_genotype(t: TreeNode, starting_symbol, depth: int, dna: list[int]):
        if type(t) not in [int, float, str, bool, list]:
            dna += find_choices_super(t, starting_symbol, depth)
        if is_metahandler(starting_symbol):
            # reconstruct metahandler?
            x = apply_inverse_metahandler(
                    g,
                    depth,
                    reconstruct_genotype,
                    starting_symbol,
                    t,
                )
            dna.append(x)
        else:
            if is_terminal(type(t), non_terminals) and (not isinstance(t, list)):
                if isinstance(t, int) or isinstance(t, float):
                    x1, x2 = normalvariate_inverse(0, 100, t)
                    dna.append(x1)
                    dna.append(x2)
                if isinstance(t, bool):
                    if t:
                        dna.append(0)
                    else:
                        dna.append(1)
            else:
                if isinstance(t, list):
                    if depth > 1:
                        x = randint_inverse(1, depth, len(t))
                        dna.append(x)                
                    children = [(type(obj), obj) for obj in t]
                else:
                    if not hasattr(t, "gengy_init_values"):
                        breakpoint()
                    children = [(typ[1], t.gengy_init_values[idx]) for idx, typ in enumerate(get_arguments(t))]
                for t, child in children:
                    dna = reconstruct_genotype(child, t, depth - 1, dna)
        return dna
        
    dna = reconstruct_genotype(p, g.starting_symbol, depth, dna)
    print(dna)
    
    return Genotype(dna)




@dataclass
class ListWrapper(Source):
    dna: list[int]
    index: int = 0

    def randint(self, min: int, max: int, prod: str = "") -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float, prod: str = "") -> float:
        k = self.randint(1, MAX_VALUE, prod)
        return 1 * (max - min) / k + min
    


def create_tree(
    g: Grammar,
    ind: Genotype,
    depth: int,
    initialization_mode: InitializationMethodType = pi_grow_method,
) -> TreeNode:
    rand: Source = ListWrapper(ind.dna)
    return random_node(rand, g, depth, g.starting_symbol, initialization_mode)


class DefaultGEMutation(MutationOperator[Genotype]):
    """Chooses a position in the list, and mutates it."""

    def mutate(
        self,
        genotype: Genotype,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> Genotype:
        return mutate(
            random_source,
            representation.grammar,
            genotype,
            representation.max_depth,
        )


class DefaultGECrossover(CrossoverOperator[Genotype]):
    """One-point crossover between the lists."""

    def crossover(
        self,
        g1: Genotype,
        g2: Genotype,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        index_in_population: int,
        generation: int,
    ) -> tuple[Genotype, Genotype]:
        return crossover(random_source, representation.grammar, g1, g2, representation.max_depth)


class GrammaticalEvolutionRepresentation(Representation[Genotype, TreeNode]):
    """This representation uses a list of integers to guide the generation of
    trees in the phenotype."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
        initialization_mode: InitializationMethodType = pi_grow_method,
    ):
        """
        Args:
            grammar (Grammar): The grammar to use in the mapping
            max_depth (int): the maximum depth when performing the mapping
            initialization_mode (InitializationMethodType): method to create individuals in the mapping
                (e.g., pi_grow, full, grow)
        """
        super().__init__(grammar, max_depth)
        self.initialization_mode = initialization_mode

    def create_individual(
        self,
        r: Source,
        depth: int | None = None,
        **kwargs,
    ) -> Genotype:
        actual_depth = depth or self.max_depth
        return random_individual(r, self.grammar, depth=actual_depth)

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree(
            self.grammar,
            genotype,
            self.max_depth,
            self.initialization_mode,
        )

    def phenotype_to_genotype(self, phenotype: Any) -> Genotype:
        """Takes an existing program and adapts it to be used in the right
        representation."""
        raise NotImplementedError(
            "Reconstruction of genotype not supported in this representation.",
        )

    def get_mutation(self) -> MutationOperator[Genotype]:
        return DefaultGEMutation()

    def get_crossover(self) -> CrossoverOperator[Genotype]:
        return DefaultGECrossover()
