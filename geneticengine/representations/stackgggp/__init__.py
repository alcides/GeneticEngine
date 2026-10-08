"""This module relies on much of the GE implementation.

The only difference is the genotype to phenotype mapping, which uses
stacks.
"""

from dataclasses import dataclass
import sys
from typing import Any, get_args
from geneticengine.exceptions import GeneticEngineError
from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource

from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.representations.tree.initializations import apply_constructor
from geneticengine.solutions.tree import TreeNode
from geneticengine.representations.linear_mutation import LinearGenomeMutation, PointMutation
from geneticengine.grammar.utils import (
    get_arguments,
    get_generic_parameter,
    get_generic_parameters,
    is_abstract,
    is_generic_list,
    is_union,
    is_metahandler,
)


@dataclass
class Genotype:
    dna: list[int]


@dataclass
class ListWrapper(RandomSource):
    dna: list[int]
    index: int = 0
    # Total codon reads (does not wrap); used to detect one full genome pass.
    consumed: int = 0

    def randint(self, min: int, max: int) -> int:
        if not self.dna:
            raise IndexError("empty stack genome")
        self.consumed += 1
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float) -> float:
        b = self.randint(1, 10)
        e = self.randint(1, 10)
        k = pow(b, e)
        v = 1 * (max - min) / k + min
        return v

    def completed_one_pass(self) -> bool:
        """Whether at least one full pass over ``dna`` has been consumed."""
        return bool(self.dna) and self.consumed >= len(self.dna)


def add_to_stacks(stacks: dict[type, list[Any]], t: type, v: Any):
    if t not in stacks:
        stacks[t] = []
    stacks[t].append(v)


def find_element_that_meets_mh(stack, metahandler):
    for index, el in enumerate(stack):
        if metahandler.validate(el):
            return index
    raise IndexError


def create_tree_using_stacks(g: Grammar, r: ListWrapper, failures_limit=100):
    """Map a linear genome to a tree via typed stacks.

    Translation continues until **both**:

    1. the start-symbol stack holds at least one value, and
    2. the genome has been consumed at least once (``len(dna)`` codon reads),

    ``failures_limit`` applies only until the first start-symbol value appears;
    after that, mapping continues until one full genome pass. The phenotype is
    the last start-symbol value observed when both conditions first hold.
    """
    all_stack_types = g.get_all_mentioned_symbols()

    stacks: dict[type, list[Any]] = {k: [] for k in all_stack_types}

    failures = 0
    last_good: Any | None = None

    while True:
        if last_good is not None and r.completed_one_pass():
            return last_good
        if last_good is None and (failures >= failures_limit or r.completed_one_pass()):
            break
        try:
            weights = g.get_weights()
            target_type: type[Any] = r.choice_weighted(
                list(all_stack_types),
                [weights.get(x, 1) for x in all_stack_types],
            )
            # print("..........")
            # print(target_type, "|", stacks)
            if is_abstract(target_type):
                concrete = r.choice(g.alternatives[target_type])
                if stacks[concrete]:
                    v = stacks[concrete].pop(0)
                    add_to_stacks(stacks, target_type, v)
                else:
                    failures += 1

            elif target_type is int:
                add_to_stacks(stacks, int, r.randint(-10000, 10000))
            elif target_type is float:
                add_to_stacks(stacks, float, r.random_float(-100.0, 100.0))
            elif target_type is bool:
                add_to_stacks(stacks, bool, r.random_bool())
            elif target_type is tuple:
                args = []
                for inner_type in get_generic_parameters(target_type):
                    ret = stacks[inner_type].pop(0)
                    args.append(ret)
                v = tuple(args)
                add_to_stacks(stacks, target_type, v)
            elif is_generic_list(target_type):
                inner_type = get_generic_parameters(target_type)[0]
                length = r.randint(0, len(stacks[inner_type]))
                ret = stacks[inner_type][:length]
                stacks[inner_type] = stacks[inner_type][length:]
                add_to_stacks(stacks, target_type, ret)
            elif is_union(target_type):
                alternatives = get_generic_parameters(target_type)
                ty = r.choice(alternatives)
                ret = stacks[ty].pop()
                add_to_stacks(stacks, target_type, ret)
            elif target_type in g.alternatives:
                compatible_productions = g.alternatives[target_type]
                alt = r.choice(compatible_productions)
                ret = stacks[alt].pop()
                add_to_stacks(stacks, target_type, ret)
            else:
                args = []
                for _, argt in get_arguments(target_type):
                    if argt in stacks:
                        arg = stacks[argt].pop()
                    elif is_metahandler(argt):
                        metahandler = get_args(argt)[1]
                        base_type = get_generic_parameter(argt)
                        index = find_element_that_meets_mh(stacks[base_type], metahandler)
                        arg = stacks[base_type].pop(index)
                    else:
                        raise IndexError()
                    args.append(arg)
                v = apply_constructor(target_type, args)
                add_to_stacks(stacks, target_type, v)
        except IndexError:
            # Only count mapping failures before the first start-symbol value;
            # afterwards keep consuming the genome until one full pass is done.
            if last_good is None:
                failures += 1
        if stacks[g.starting_symbol]:
            last_good = stacks[g.starting_symbol][-1]
    raise GeneticEngineError("Stack genome not enough.")


class StackBasedGGGPRepresentation(
    Representation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    """This representation uses a list of integers to guide the generation of
    trees in the phenotype.

    Mapping stops once the start-symbol stack is non-empty **and** the genome
    has been read at least once; the last start-symbol value is returned.
    """

    def __init__(
        self,
        grammar: Grammar,
        gene_length: int = 1024,
        failures_limit: int = 100,
        mutation: LinearGenomeMutation[int] | None = None,
    ):
        """
        Args:
            grammar: Grammar used for stack-based mapping
            gene_length: Initial genome length for new individuals
            failures_limit: Max mapping failures before giving up
            mutation: Mutation strategy for the integer gene list. Defaults to
                :class:`~geneticengine.representations.linear_mutation.PointMutation`. Pass
                :class:`~geneticengine.representations.linear_mutation.UMAD` for
                addition/deletion mutation.
        """
        self.grammar = grammar
        self.gene_length = gene_length
        self.failures_limit = failures_limit
        self.mutation: LinearGenomeMutation[int] = mutation if mutation is not None else PointMutation()

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype(dna=[random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree_using_stacks(self.grammar, ListWrapper(genotype.dna), failures_limit=self.failures_limit)

    def _random_gene(self, random: RandomSource) -> int:
        return random.randint(0, sys.maxsize)

    def mutate(self, random: RandomSource, genotype: Genotype, **kwargs) -> Genotype:
        dna = self.mutation.mutate(
            list(genotype.dna),
            random,
            gene_factory=lambda: self._random_gene(random),
        )
        return Genotype(dna)

    def crossover(
        self,
        random: RandomSource,
        parent1: Genotype,
        parent2: Genotype,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        limit = min(len(parent1.dna), len(parent2.dna))
        if limit <= 0:
            return (Genotype(list(parent1.dna)), Genotype(list(parent2.dna)))
        rindex = random.randint(0, limit - 1)
        c1 = parent1.dna[:rindex] + parent2.dna[rindex:]
        c2 = parent2.dna[:rindex] + parent1.dna[rindex:]
        return (Genotype(c1), Genotype(c2))
