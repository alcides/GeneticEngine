"""Solve the digitwise-modulo reciprocal challenge."""

from fractions import Fraction

from examples.benchmarks.mathematical import incremented_reciprocal


def solve(n: int = 7) -> tuple[int, int]:
    result = incremented_reciprocal(n)
    assert Fraction(*result) == Fraction(16, 63)
    return result


if __name__ == "__main__":
    print(solve())
