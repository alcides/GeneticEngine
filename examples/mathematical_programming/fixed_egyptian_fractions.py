"""Genetic programming example for fixed-size Egyptian fractions."""

from examples.benchmarks.benchmark import example_run
from examples.benchmarks.mathematical import FixedEgyptianFractionBenchmark


if __name__ == "__main__":
    example_run(FixedEgyptianFractionBenchmark())
