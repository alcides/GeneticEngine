"""Solve the generalized Egyptian-fraction challenge for 5/12."""

from fractions import Fraction

from examples.benchmarks.mathematical import rational_egyptian_fraction


def solve(numerator: int = 5, denominator: int = 12) -> tuple[int, ...]:
    terms = rational_egyptian_fraction(numerator, denominator)
    assert Fraction(numerator, denominator) == sum((Fraction(1, term) for term in terms), Fraction())
    assert len(set(terms)) == 5
    assert numerator not in terms and denominator not in terms
    return terms


if __name__ == "__main__":
    print(solve())
