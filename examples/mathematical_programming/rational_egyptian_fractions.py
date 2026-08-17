"""Genetic programming example for rational-target Egyptian fractions."""

from examples.benchmarks.benchmark import example_run
from examples.benchmarks.mathematical import RationalEgyptianFractionBenchmark


if __name__ == "__main__":
    example_run(RationalEgyptianFractionBenchmark())
