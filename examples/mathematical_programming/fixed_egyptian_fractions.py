"""Solve the fixed-size Egyptian-fraction challenge for k = 7."""

from fractions import Fraction

from examples.benchmarks.mathematical import fixed_egyptian_fraction


def solve(k: int = 7) -> tuple[int, ...]:
    terms = fixed_egyptian_fraction(k)
    assert Fraction(1, k) == sum((Fraction(1, term) for term in terms), Fraction())
    assert len(set(terms)) == 5
    return terms


if __name__ == "__main__":
    print(solve())
