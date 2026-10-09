from __future__ import annotations

import math

from examples.benchmarks.metamorphic_testing import MetamorphicTestingBenchmark, Sine, Input


def test_sine_satisfies_periodic_metamorphic_relation():
    benchmark = MetamorphicTestingBenchmark()
    assert benchmark.fitness(Sine(Input())) < 1e-12


def test_non_periodic_program_is_penalized():
    benchmark = MetamorphicTestingBenchmark()
    assert benchmark.fitness(Input()) > 1.0
    assert math.isfinite(benchmark.fitness(Input()))
