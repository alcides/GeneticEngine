import pytest
from geneticengine.core.problems import SingleObjectiveProblem

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.common import GenericPopulationWrapper
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import DynamicStructuredGrammaticalEvolutionRepresentation
from geneticengine.core.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.core.representations.grammatical_evolution.structured_ge import StructuredGrammaticalEvolutionRepresentation
from geneticengine.core.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.core.representations.tree.initializations import full_method, grow_method, pi_grow_method
from geneticengine.core.representations.tree.operators import FullInitializer, GrowInitializer, PositionIndependentGrowInitializer, RampedHalfAndHalfInitializer, RampedInitializer
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation, random_node

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
        (GrammaticalEvolutionRepresentation, GenericPopulationWrapper),
        (StructuredGrammaticalEvolutionRepresentation, GenericPopulationWrapper),
        (DynamicStructuredGrammaticalEvolutionRepresentation, GenericPopulationWrapper),
        (StackBasedGGGPRepresentation, GenericPopulationWrapper),
        
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
