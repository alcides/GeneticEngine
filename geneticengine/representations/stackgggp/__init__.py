"""This module relies on much of the GE implementation.

The only difference is the genotype to phenotype mapping, which uses
stacks.
"""

import copy
from dataclasses import dataclass
import sys
from typing import Any
from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource

from geneticengine.representations.api import (
    RepresentationWithCrossover,
    RepresentationWithMutation,
    Representation,
)
from geneticengine.solutions.tree import TreeNode
from geneticengine.grammar.utils import (
    get_arguments,
    get_generic_parameter,
    get_generic_parameters,
    is_abstract,
    is_generic_list,
)
from geneticengine.grammar.metahandlers.base import is_metahandler


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
        k = self.randint(1, sys.maxsize)
        return 1 * (max - min) / k + min


def add_to_stacks(stacks: dict[type, list[Any]], t: type, v: Any):
    if t not in stacks:
        stacks[t] = []
    stacks[t].append(v)


def create_tree_using_stacks(g: Grammar, r: ListWrapper, max_depth: int = 10):
    a, b, c = g.get_all_symbols()
    all_stack_types = list(a) + list(b) + list(c)

    stacks: dict[type, list[Any]] = {k: [] for k in all_stack_types}

    failures = 0

    while not stacks[g.starting_symbol]:
        weights = g.get_weights()
        target_type: type[Any] = r.choice_weighted(all_stack_types, [weights[x] for x in all_stack_types])
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
        elif is_generic_list(target_type):
            inner_type: type = get_generic_parameters(target_type)[0]
            length = r.randint(0, len(stacks[inner_type]))
            ret = stacks[inner_type][:length]
            stacks[inner_type] = stacks[inner_type][length:]
            add_to_stacks(stacks, target_type, ret)
        else:

            def get_element_of_type(argt):
                nonlocal failures
                if argt in stacks:
                    if stacks[argt]:
                        return stacks[argt].pop()
                    else:
                        failures += 1
                        raise IndexError(f"No value for {argt} yet.")
                elif is_metahandler(argt):
                    metahandler = argt.__metadata__[0]
                    base_type = get_generic_parameter(argt)

                    def new_symbol(rule, receiver, budget, id, ctx):
                        receiver(stacks[rule].pop(0))

                    ret = None

                    def receiver(v):
                        nonlocal ret
                        ret = v

                    metahandler.generate(
                        r,
                        g,
                        receiver,
                        new_symbol,
                        0,
                        base_type,
                        {"_": str(argt)},
                    )
                    return ret
                elif is_generic_list(argt):
                    length = r.randint(0, 10)
                    ty = get_generic_parameter(argt)
                    list_prototype = []
                    for _ in range(length):
                        list_prototype.append(stacks[ty].pop())
                    return list_prototype
                else:
                    assert False, f"Type {argt} not supported in StackBasedGGGP"

            oldstacks = copy.deepcopy(stacks)
            try:
                kwargs = {argn: get_element_of_type(argt) for argn, argt in get_arguments(target_type)}
                add_to_stacks(stacks, target_type, target_type(**kwargs))
            except IndexError:
                stacks = oldstacks
                failures += 1

    return stacks[g.starting_symbol][0]


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
        max_depth: int,
        gene_length: int = 256,
    ):
        self.grammar = grammar
        self.max_depth = max_depth
        self.gene_length = gene_length

    def create_genotype(self, random: RandomSource, **kwargs) -> Genotype:
        return Genotype(dna=[random.randint(0, sys.maxsize) for _ in range(self.gene_length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree_using_stacks(
            self.grammar,
            ListWrapper(genotype.dna),
            self.max_depth,
        )

    def mutate(self, random: RandomSource, internal: Genotype, **kwargs) -> Genotype:
        rindex = random.randint(0, 255)
        clone = [i for i in internal.dna]
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
