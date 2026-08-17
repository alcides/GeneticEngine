"""Solve the repeating-decimal rational-number challenge."""

from fractions import Fraction

from examples.benchmarks.mathematical import reptend_fraction


def solve(prefix: int = 539, repetend: int = 4762) -> tuple[int, int]:
    result = reptend_fraction(prefix, repetend)
    assert Fraction(*result) == Fraction(prefix * 9999 + repetend, 1000 * 9999)
    return result


if __name__ == "__main__":
    print(solve())
