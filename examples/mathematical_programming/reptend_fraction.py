"""Genetic programming example for repeating-decimal rational numbers."""

from examples.benchmarks.benchmark import example_run
from examples.benchmarks.mathematical import ReptendFractionBenchmark


if __name__ == "__main__":
    example_run(ReptendFractionBenchmark())
