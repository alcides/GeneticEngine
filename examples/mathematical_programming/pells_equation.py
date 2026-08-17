"""Genetic programming example for Pell's equation."""

from examples.benchmarks.benchmark import example_run
from examples.benchmarks.pell import PellsEquationBenchmark


if __name__ == "__main__":
    example_run(PellsEquationBenchmark())
