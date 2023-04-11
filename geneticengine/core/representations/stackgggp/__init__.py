"""This module relies on much of the GE implementation.

The only difference is the genotype to phenotype mapping, which uses
stacks.
"""

import copy
from typing import Any, Optional
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source

from geneticengine.core.representations.api import CrossoverOperator, MutationOperator, Representation
from geneticengine.core.representations.grammatical_evolution.ge import (
    DefaultGECrossover,
    DefaultGEMutation,
    Genotype,
    ListWrapper,
)
from geneticengine.core.tree import TreeNode
from geneticengine.core.utils import (
    get_arguments,
    get_generic_parameter,
    get_generic_parameters,
    is_abstract,
    is_generic_list,
)
from geneticengine.metahandlers.base import is_metahandler


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


class StackBasedGGGPRepresentation(Representation[Genotype, TreeNode]):
    """This representation uses a list of integers to guide the generation of
    trees in the phenotype."""

    def __init__(
        self,
        grammar: Grammar,
        max_depth: int,
    ):
        super().__init__(grammar, max_depth)

    def create_individual(
        self,
        r: Source,
        depth: Optional[int] = None,
        **kwargs,
    ) -> Genotype:
        length = kwargs.get("length", 1000)
        return Genotype(dna=[r.randint(0, 10000000) for _ in range(length)])

    def genotype_to_phenotype(self, genotype: Genotype) -> TreeNode:
        return create_tree_using_stacks(
            self.grammar,
            ListWrapper(genotype.dna),
            self.max_depth,
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
