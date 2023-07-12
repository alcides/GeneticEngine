import pytest

from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.initializations import full_method, grow_method, pi_grow_method
from geneticengine.core.representations.tree.treebased import random_node

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
