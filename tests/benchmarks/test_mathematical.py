from fractions import Fraction

from examples.benchmarks.mathematical import FixedEgyptianFractionBenchmark
from examples.benchmarks.mathematical import IncrementedReciprocalBenchmark
from examples.benchmarks.mathematical import RationalEgyptianFractionBenchmark
from examples.benchmarks.mathematical import ReptendFractionBenchmark
from examples.benchmarks.pell import PellsEquationBenchmark
from examples.benchmarks.pell import PellPair
from examples.benchmarks.pell import pell_fitness
from examples.benchmarks.mathematical import fixed_egyptian_fraction
from examples.benchmarks.mathematical import incremented_reciprocal
from examples.benchmarks.mathematical import reptend_fraction


def test_fixed_egyptian_fraction_is_valid():
    terms = fixed_egyptian_fraction(7)
    assert len(set(terms)) == 5
    assert Fraction(1, 7) == sum((Fraction(1, term) for term in terms), Fraction())


def test_reptend_example():
    numerator, denominator = reptend_fraction(539, 4762)
    assert Fraction(numerator, denominator) == Fraction(539 * 9999 + 4762, 1000 * 9999)


def test_incremented_reciprocal_examples():
    assert incremented_reciprocal(1) == (1, 9)
    assert incremented_reciprocal(2) == (11, 18)
    # The page prints 16/163, but 0.overline{253968} reduces to 16/63.
    assert incremented_reciprocal(7) == (16, 63)


def test_benchmarks_expose_zero_fitness_reference_targets():
    fixed = FixedEgyptianFractionBenchmark()
    candidate = type("C", (), {"evaluate": staticmethod(lambda _: fixed_egyptian_fraction(7))})()
    assert fixed.get_problem().evaluate(candidate).fitness_components == [0]
    reptend = ReptendFractionBenchmark()
    assert reptend.get_problem().target == [0]
    assert incremented_reciprocal(7) == (16, 63)


def test_all_benchmark_grammars_are_constructible():
    for benchmark in (
        FixedEgyptianFractionBenchmark(),
        RationalEgyptianFractionBenchmark(),
        ReptendFractionBenchmark(),
        IncrementedReciprocalBenchmark(),
        PellsEquationBenchmark(),
    ):
        assert benchmark.get_grammar() is not None


def test_pells_equation_minimizes_the_smallest_positive_solution():
    assert pell_fitness(PellPair(3, 2)) == 5
    assert pell_fitness(PellPair(1, 1)) > pell_fitness(PellPair(3, 2))
    assert PellsEquationBenchmark().get_problem().target == [5]
