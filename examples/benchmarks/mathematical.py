"""Benchmarks based on Vaguery's mathematical programming challenges.

The scoring functions use :class:`fractions.Fraction` deliberately.  These
problems are about exact rational arithmetic and floating-point fitness would
make correct programs appear incorrect for sufficiently large inputs.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import Problem, SingleObjectiveProblem


class IntegerExpression(ABC):
    def evaluate(self, values: tuple[int, ...]) -> int:
        raise NotImplementedError


@dataclass
class IntegerLiteral(IntegerExpression):
    value: Annotated[int, IntRange(-100, 100)]

    def evaluate(self, values: tuple[int, ...]) -> int:
        return self.value


@dataclass
class InputInteger(IntegerExpression):
    index: Annotated[int, IntRange(0, 1)]

    def evaluate(self, values: tuple[int, ...]) -> int:
        return values[self.index]


@dataclass
class Add(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> int:
        return self.left.evaluate(values) + self.right.evaluate(values)


@dataclass
class Subtract(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> int:
        return self.left.evaluate(values) - self.right.evaluate(values)


@dataclass
class Multiply(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> int:
        return self.left.evaluate(values) * self.right.evaluate(values)


@dataclass
class Quotient(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> int:
        denominator = self.right.evaluate(values)
        return 0 if denominator == 0 else self.left.evaluate(values) // denominator


@dataclass
class Remainder(IntegerExpression):
    left: IntegerExpression
    right: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> int:
        denominator = self.right.evaluate(values)
        return 0 if denominator == 0 else self.left.evaluate(values) % denominator


@dataclass
class FiveUnitFractions:
    terms: tuple[IntegerExpression, IntegerExpression, IntegerExpression, IntegerExpression, IntegerExpression]

    def evaluate(self, values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(term.evaluate(values) for term in self.terms)


@dataclass
class IntegerPair:
    numerator: IntegerExpression
    denominator: IntegerExpression

    def evaluate(self, values: tuple[int, ...]) -> tuple[int, int]:
        return self.numerator.evaluate(values), self.denominator.evaluate(values)


def fixed_egyptian_fraction(k: int) -> tuple[int, ...]:
    """Return five distinct nonzero denominators summing to ``1/k``."""
    return 2 * k, 3 * k, 6 * k, 7 * k, -7 * k


def rational_egyptian_fraction(numerator: int, denominator: int) -> tuple[int, ...]:
    """Find five signed unit fractions for a small rational target.

    The finite benchmark domain is intentionally small; exhaustive search is
    used for the reference oracle so the benchmark remains independently
    checkable and does not encode a particular synthesis strategy.
    """
    target = Fraction(numerator, denominator)
    forbidden = {numerator, denominator}
    candidates = [i for i in range(-200, 201) if i and i not in forbidden]
    for a in candidates:
        for b in candidates:
            if b == a:
                continue
            remainder = target - Fraction(1, a) - Fraction(1, b)
            if remainder.numerator not in (-1, 1):
                continue
            c = remainder.denominator * remainder.numerator
            if c in forbidden or c in {a, b} or c not in candidates:
                continue
            d = next((x for x in candidates if x not in {a, b, c} and -x in candidates), None)
            if d is not None and -d not in {a, b, c, d}:
                return a, b, c, d, -d
    raise ValueError(f"no five-term representation found for {numerator}/{denominator}")


def reptend_fraction(prefix: int, repetend: int) -> tuple[int, int]:
    prefix_digits = len(str(prefix))
    repetend_digits = len(str(repetend))
    cycle = 10**repetend_digits - 1
    return prefix * cycle + repetend, 10**prefix_digits * cycle


def incremented_reciprocal(n: int) -> tuple[int, int]:
    """Increment every digit of the repeating decimal expansion of ``1/n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    remainder = 1 % n
    seen: dict[int, int] = {}
    digits: list[int] = []
    while remainder and remainder not in seen:
        seen[remainder] = len(digits)
        remainder *= 10
        digits.append(remainder // n)
        remainder %= n
    cycle_start = seen.get(remainder, len(digits))
    nonrepeating = digits[:cycle_start]
    repeating = digits[cycle_start:] or [0]
    shifted_nonrepeating = [((digit + 1) % 10) for digit in nonrepeating]
    shifted_repeating = [((digit + 1) % 10) for digit in repeating]
    scale = 10 ** len(shifted_nonrepeating)
    prefix = 0
    for digit in shifted_nonrepeating:
        prefix = 10 * prefix + digit
    cycle_value = 0
    for digit in shifted_repeating:
        cycle_value = 10 * cycle_value + digit
    denominator = scale * (10 ** len(shifted_repeating) - 1)
    value = Fraction(prefix, scale) + Fraction(cycle_value, denominator)
    return value.numerator, value.denominator


def _unit_fraction_error(candidate: tuple[int, ...], target: Fraction, forbidden: set[int]) -> float:
    if len(candidate) != 5 or any(value == 0 for value in candidate):
        return 1_000_000.0
    error = abs(sum((Fraction(1, value) for value in candidate), Fraction()) - target)
    penalty = 0 if len(set(candidate)) == 5 and not forbidden.intersection(candidate) else 1
    return float(error) + penalty


def _pair_error(candidate: tuple[int, int], target: Fraction) -> float:
    numerator, denominator = candidate
    if denominator == 0:
        return 1_000_000.0
    return float(abs(Fraction(numerator, denominator) - target))


class _IntegerExpressionBenchmark(Benchmark):
    expression_nodes = [IntegerLiteral, InputInteger, Add, Subtract, Multiply, Quotient, Remainder]

    def get_grammar(self) -> Grammar:
        return extract_grammar(self.expression_nodes + [self.root_type], self.root_type)


class FixedEgyptianFractionBenchmark(_IntegerExpressionBenchmark):
    root_type = FiveUnitFractions

    def __init__(self) -> None:
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=0,
            fitness_function=lambda candidate: _unit_fraction_error(
                candidate.evaluate((7, 0)), Fraction(1, 7), {7}
            ),
        )

    def get_problem(self) -> Problem:
        return self.problem


class RationalEgyptianFractionBenchmark(_IntegerExpressionBenchmark):
    root_type = FiveUnitFractions

    def __init__(self) -> None:
        target = Fraction(5, 12)
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=0,
            fitness_function=lambda candidate: _unit_fraction_error(
                candidate.evaluate((5, 12)), target, {5, 12}
            ),
        )

    def get_problem(self) -> Problem:
        return self.problem


class ReptendFractionBenchmark(_IntegerExpressionBenchmark):
    root_type = IntegerPair

    def __init__(self) -> None:
        target = Fraction(*reptend_fraction(539, 4762))
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=0,
            fitness_function=lambda candidate: _pair_error(candidate.evaluate((539, 4762)), target),
        )

    def get_problem(self) -> Problem:
        return self.problem


class IncrementedReciprocalBenchmark(_IntegerExpressionBenchmark):
    root_type = IntegerPair

    def __init__(self) -> None:
        target = Fraction(*incremented_reciprocal(7))
        self.problem = SingleObjectiveProblem(
            minimize=True,
            target=0,
            fitness_function=lambda candidate: _pair_error(candidate.evaluate((7, 0)), target),
        )

    def get_problem(self) -> Problem:
        return self.problem


BENCHMARKS = [
    FixedEgyptianFractionBenchmark,
    RationalEgyptianFractionBenchmark,
    ReptendFractionBenchmark,
    IncrementedReciprocalBenchmark,
]
