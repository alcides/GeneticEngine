"""Genetic programming example for the incremented reciprocal challenge."""

from examples.benchmarks.benchmark import example_run
from examples.benchmarks.mathematical import IncrementedReciprocalBenchmark


if __name__ == "__main__":
    example_run(IncrementedReciprocalBenchmark())
