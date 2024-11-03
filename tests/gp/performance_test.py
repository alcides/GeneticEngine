import pytest
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.problems import SingleObjectiveProblem

from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.common import GenericPopulationInitializer
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DynamicStructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.representations.grammatical_evolution.structured_ge import (
    StructuredGrammaticalEvolutionRepresentation,
)
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.representations.tree.initializations import (
    FullDecider,
    MaxDepthDecider,
    PositionIndependentGrowDecider,
)
from geneticengine.representations.tree.operators import (
    FullInitializer,
    GrowInitializer,
    PositionIndependentGrowInitializer,
    RampedHalfAndHalfInitializer,
)
from geneticengine.representations.tree.treebased import (
    TreeBasedRepresentation,
    random_tree,
)

from geneticengine.solutions.individual import Individual
from utils.benchmark_grammars_test import Root, grammar


def force_generation_of_population(pop: list[Individual]):
    for ind in pop:
        ind.get_phenotype()


representation_list = [
    lambda r, g, d: TreeBasedRepresentation(g, MaxDepthDecider(r, g, d)),
    lambda r, g, d: GrammaticalEvolutionRepresentation(g, MaxDepthDecider(r, g, d)),
    lambda r, g, d: StructuredGrammaticalEvolutionRepresentation(g, MaxDepthDecider(r, g, d)),
    lambda r, g, d: DynamicStructuredGrammaticalEvolutionRepresentation(g, d),
    lambda r, g, d: StackBasedGGGPRepresentation(g),
]


@pytest.mark.parametrize(
    "d",
    [
        lambda r, g, d: MaxDepthDecider(r, g, d),
        lambda r, g, d: FullDecider(r, g, d),
        lambda r, g, d: PositionIndependentGrowDecider(r, g, d),
    ],
    ids=["MaxDepth", "Full", "PIGrow"],
)
@pytest.mark.benchmark(group="initializers", disable_gc=True, warmup=True, warmup_iterations=10, min_rounds=500)
def test_bench_initialization(benchmark, d):
    r = NativeRandomSource(seed=1)

    def population_initialization():
        n = random_tree(r, grammar, decider=d(r, grammar, 20))
        return n

    n = benchmark(population_initialization)
    assert isinstance(n, Root)


@pytest.mark.parametrize(
    "representation,initializer_builder",
    [
        (representation_list[0], lambda d: FullInitializer(d)),
        (representation_list[0], lambda d: GrowInitializer()),
        (representation_list[0], lambda d: PositionIndependentGrowInitializer(d)),
        (representation_list[0], lambda d: RampedHalfAndHalfInitializer(d)),
        (representation_list[1], lambda d: GenericPopulationInitializer()),
        (representation_list[2], lambda d: GenericPopulationInitializer()),
        (representation_list[3], lambda d: GenericPopulationInitializer()),
        (representation_list[4], lambda d: GenericPopulationInitializer()),
    ],
    ids=["Tree-Full", "Tree-Grow", "Tree-PIGrow", "Tree-HandH", "GE", "SGE", "DSGE", "StackGP"],
)
@pytest.mark.benchmark(group="initializers_class", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_initialization_class(benchmark, representation, initializer_builder):
    r = NativeRandomSource(seed=1)
    p = SingleObjectiveProblem(lambda x: 3)
    target_depth = 15
    target_size = 100

    def population_initialization():
        repr = representation(r, grammar, target_depth)
        initializer = initializer_builder(target_depth)
        population = list(initializer.initialize(problem=p, representation=repr, random=r, target_size=target_size))
        force_generation_of_population(population)
        return len(population)

    n = benchmark(population_initialization)
    assert n > 0


@pytest.mark.parametrize("representation", representation_list, ids=["TreeBased", "GE", "SGE", "DSGE", "StackGP"])
@pytest.mark.benchmark(group="mutation", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_mutation(benchmark, representation):
    r = NativeRandomSource(seed=1)
    target_depth = 20
    target_size = 100
    number_of_iterations = 100

    repr = representation(r, grammar, target_depth)

    gs = GenericMutationStep()

    def mutation():
        p = SingleObjectiveProblem(lambda x: 3)
        population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
        for _ in range(number_of_iterations):
            population = list(gs.apply(p, SequentialEvaluator(), repr, r, population, target_size, 0))
            force_generation_of_population(population)
        return len(population)

    n = benchmark(mutation)
    assert n > 0


@pytest.mark.parametrize("representation", representation_list, ids=["TreeBased", "GE", "SGE", "DSGE", "StackGP"])
@pytest.mark.benchmark(group="crossover", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_crossover(benchmark, representation):
    r = NativeRandomSource(seed=1)
    target_depth = 20
    target_size = 100
    number_of_iterations = 100

    repr = representation(r, grammar, target_depth)

    gs = GenericCrossoverStep()

    def mutation():
        p = SingleObjectiveProblem(lambda x: 3)
        population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
        for _ in range(number_of_iterations):
            population = list(gs.apply(p, SequentialEvaluator(), repr, r, population, target_size, 0))
            force_generation_of_population(population)
        return len(population)

    n = benchmark(mutation)
    assert n > 0
