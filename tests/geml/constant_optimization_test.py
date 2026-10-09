import numpy as np
import pytest
import sympy

pytest.importorskip("jax")
pytest.importorskip("equinox")
pytest.importorskip("sympy2jax")

from geml.constant_optimization import fit_constants


def test_fit_constants_recovers_linear_coefficients():
    x = sympy.Symbol("x")
    samples = np.linspace(-2.0, 2.0, 64)
    targets = 3.0 * samples + 1.0

    fitted = fit_constants(
        sympy.Float(0.5) * x + sympy.Float(0.5),
        {"x": samples},
        targets,
        learning_rate=0.05,
        steps=500,
    )

    error = np.mean((np.asarray([float(fitted.subs(x, value)) for value in samples]) - targets) ** 2)
    assert error < 1e-4


@pytest.mark.parametrize("learning_rate,steps", [(0, 1), (-0.1, 1), (0.1, -1)])
def test_fit_constants_rejects_invalid_training_parameters(learning_rate, steps):
    with pytest.raises(ValueError):
        fit_constants(sympy.Symbol("x"), {"x": [1.0]}, [1.0], learning_rate=learning_rate, steps=steps)
