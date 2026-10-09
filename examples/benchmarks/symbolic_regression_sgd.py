"""Fit the numeric constants of a symbolic-regression candidate with SGD.

Run with `uv run --extra sgd python -m examples.benchmarks.symbolic_regression_sgd`.
"""

import numpy as np
import sympy

from geml.constant_optimization import fit_constants


if __name__ == "__main__":
    x = sympy.Symbol("x")
    samples = np.linspace(-2.0, 2.0, 128)
    targets = 3.0 * samples + 1.0
    candidate = sympy.Float(0.5) * x + sympy.Float(0.5)

    fitted = fit_constants(candidate, {"x": samples}, targets, learning_rate=0.05, steps=500)
    print(f"Candidate: {candidate}")
    print(f"Fitted:    {fitted}")
