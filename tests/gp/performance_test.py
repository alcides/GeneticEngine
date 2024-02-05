import pytest
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.evaluation import SequentialEvaluator
from geneticengine.problems import SingleObjectiveProblem

from geneticengine.random.sources import RandomSource
from geneticengine.representations.common import GenericPopulationInitializer
from geneticengine.representations.grammatical_evolution.dynamic_structured_ge import (
    DefaultDSGECrossover,
    DynamicStructuredGrammaticalEvolutionRepresentation,
    DefaultDSGEMutation,
)
from geneticengine.representations.grammatical_evolution.ge import (
    DefaultGECrossover,
    GrammaticalEvolutionRepresentation,
    DefaultGEMutation,
)
from geneticengine.representations.grammatical_evolution.structured_ge import (
    DefaultSGECrossover,
    StructuredGrammaticalEvolutionRepresentation,
    DefaultSGEMutation,
)
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.representations.tree.initializations import full_method, grow_method, pi_grow_method
from geneticengine.representations.tree.operators import (
    FullInitializer,
    GrowInitializer,
    PositionIndependentGrowInitializer,
    RampedHalfAndHalfInitializer,
    RampedInitializer,
)
from geneticengine.representations.tree.treebased import (
    DefaultTBCrossover,
    DefaultTBMutation,
    TreeBasedRepresentation,
    random_node,
)
from geneticengine.representations.tree_smt.operators import SMTGrowInitializer
from geneticengine.representations.tree_smt.treebased import (
    DefaultSMTTBCrossover,
    DefaultSMTTBMutation,
    SMTTreeBasedRepresentation,
)

from utils.benchmark_grammars_test import Root, grammar


@pytest.mark.parametrize(
    "fun",
    [
        full_method,
        grow_method,
        pi_grow_method,
    ],
)
@pytest.mark.benchmark(group="initializers", disable_gc=True, warmup=True, warmup_iterations=10, min_rounds=500)
def test_bench_initialization(benchmark, fun):
    r = RandomSource(seed=1)

    def population_initialization():
        n = random_node(r, grammar, 20, Root, method=fun)
        return n

    n = benchmark(population_initialization)
    assert isinstance(n, Root)


@pytest.mark.parametrize(
    "representation,initializer",
    [
        (TreeBasedRepresentation, FullInitializer),
        (TreeBasedRepresentation, GrowInitializer),
        (TreeBasedRepresentation, PositionIndependentGrowInitializer),
        (TreeBasedRepresentation, RampedInitializer),
        (TreeBasedRepresentation, RampedHalfAndHalfInitializer),
        (SMTTreeBasedRepresentation, SMTGrowInitializer),
        (GrammaticalEvolutionRepresentation, GenericPopulationInitializer),
        (StructuredGrammaticalEvolutionRepresentation, GenericPopulationInitializer),
        (DynamicStructuredGrammaticalEvolutionRepresentation, GenericPopulationInitializer),
        (StackBasedGGGPRepresentation, GenericPopulationInitializer),
    ],
)
@pytest.mark.benchmark(group="initializers_class", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_initialization_class(benchmark, representation, initializer):
    r = RandomSource(seed=1)
    p = SingleObjectiveProblem(lambda x: 3)
    target_depth = 20
    target_size = 100

    def population_initialization():

        repr = representation(grammar=grammar, max_depth=target_depth)

        population = initializer().initialize(p, repr, r, target_size)
        return len(population)

    n = benchmark(population_initialization)
    assert n > 0


@pytest.mark.parametrize(
    "representation,mut",
    [
        (TreeBasedRepresentation, DefaultTBMutation),
        (SMTTreeBasedRepresentation, DefaultSMTTBMutation),
        (GrammaticalEvolutionRepresentation, DefaultGEMutation),
        (StructuredGrammaticalEvolutionRepresentation, DefaultSGEMutation),
        (DynamicStructuredGrammaticalEvolutionRepresentation, DefaultDSGEMutation),
        (StackBasedGGGPRepresentation, DefaultGEMutation),
    ],
)
@pytest.mark.benchmark(group="mutation", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_mutation(benchmark, representation, mut):
    r = RandomSource(seed=1)
    target_depth = 20
    target_size = 100

    repr = representation(grammar=grammar, max_depth=target_depth)

    gs = GenericMutationStep(operator=mut())

    def mutation():
        p = SingleObjectiveProblem(lambda x: 3)
        population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
        for _ in range(100):
            population = gs.iterate(p, SequentialEvaluator(), repr, r, population, len(population), 0)
            for p in population:
                p.get_phenotype()

        return len(population)

    n = benchmark(mutation)
    assert n > 0


@pytest.mark.parametrize(
    "representation,xo",
    [
        (TreeBasedRepresentation, DefaultTBCrossover),
        (SMTTreeBasedRepresentation, DefaultSMTTBCrossover),
        (GrammaticalEvolutionRepresentation, DefaultGECrossover),
        (StructuredGrammaticalEvolutionRepresentation, DefaultSGECrossover),
        (DynamicStructuredGrammaticalEvolutionRepresentation, DefaultDSGECrossover),
        (StackBasedGGGPRepresentation, DefaultGECrossover),
    ],
)
@pytest.mark.benchmark(group="crossover", disable_gc=True, warmup=True, warmup_iterations=1, min_rounds=5)
def test_bench_crossover(benchmark, representation, xo):
    r = RandomSource(seed=1)
    target_depth = 20
    target_size = 100

    repr = representation(grammar=grammar, max_depth=target_depth)

    gs = GenericCrossoverStep(operator=xo())

    def mutation():
        p = SingleObjectiveProblem(lambda x: 3)
        population = GenericPopulationInitializer().initialize(p, repr, r, target_size)
        for _ in range(100):
            population = gs.iterate(p, SequentialEvaluator(), repr, r, population, len(population), 0)
            for p in population:
                p.get_phenotype()

        return len(population)

    n = benchmark(mutation)
    assert n > 0
