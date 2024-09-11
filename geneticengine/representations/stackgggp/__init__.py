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

    def randint(self, min: int, max: int) -> int:
        self.index = (self.index + 1) % len(self.dna)
        v = self.dna[self.index]
        return v % (max - min + 1) + min

    def random_float(self, min: float, max: float) -> float:
        b = self.randint(1, 10)
        e = self.randint(1, 10)
        k = pow(b, e)
        v = 1 * (max - min) / k + min
        return v


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
    all_stack_types = g.get_all_mentioned_symbols()

    stacks: dict[type, list[Any]] = {k: [] for k in all_stack_types}

    failures = 0

    while not stacks[g.starting_symbol] and failures < failures_limit:
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
            failures += 1
    if stacks[g.starting_symbol]:
        return stacks[g.starting_symbol][0]
    else:
        raise GeneticEngineError("Stack genome not enough.")


class StackBasedGGGPRepresentation(
    Representation[Genotype, TreeNode],
    RepresentationWithMutation[Genotype],
    RepresentationWithCrossover[Genotype],
):
    """This representation uses a list of integers to guide the generation of
    trees in the phenotype."""

    def __init__(
        self,
        grammar: Grammar,
        gene_length: int = 1024,
        failures_limit: int = 100,
    ):
        self.grammar = grammar
        self.gene_length = gene_length
        self.failures_limit = failures_limit

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype(dna=[random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree_using_stacks(self.grammar, ListWrapper(genotype.dna), failures_limit=self.failures_limit)

    def mutate(self, random: RandomSource, genotype: Genotype, **kwargs) -> Genotype:
        rindex = random.randint(0, self.gene_length - 1)
        clone = [i for i in genotype.dna]
        clone[rindex] = random.randint(0, 10000)
        return Genotype(clone)

    def crossover(
        self,
        random: RandomSource,
        parent1: Genotype,
        parent2: Genotype,
        **kwargs,
    ) -> tuple[Genotype, Genotype]:
        rindex = random.randint(0, 255)

        c1 = parent1.dna[:rindex] + parent2.dna[rindex:]
        c2 = parent2.dna[:rindex] + parent1.dna[rindex:]
        return (Genotype(c1), Genotype(c2))
