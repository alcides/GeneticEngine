"""Optional gradient-based fitting of constants in symbolic expressions."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import sympy


def fit_constants(
    expression: sympy.Expr,
    inputs: Mapping[str, Any],
    targets: Any,
    *,
    learning_rate: float = 0.01,
    steps: int = 100,
) -> sympy.Expr:
    """Fit numeric constants in a SymPy expression with stochastic gradient descent.

    `inputs` maps SymPy symbol names to arrays. All samples are used on each update,
    so this performs full-batch gradient descent. Integer and rational constants are
    converted to floating-point leaves before optimization.

    The optional `sgd` dependencies are required to call this function.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import sympy2jax
    except ImportError as error:
        raise ImportError("fit_constants requires the optional dependencies; install GeneticEngine[sgd]") from error

    numeric_constants = expression.atoms(sympy.Integer, sympy.Rational)
    float_expression = expression.xreplace({value: sympy.Float(value) for value in numeric_constants})
    module = sympy2jax.SymbolicModule([float_expression])
    jax_inputs = {name: jnp.asarray(values) for name, values in inputs.items()}
    jax_targets = jnp.asarray(np.asarray(targets))

    def loss_fn(model):
        predictions = model(**jax_inputs)[0]
        return jnp.mean(jnp.square(predictions - jax_targets))

    for _ in range(steps):
        _, gradients = eqx.filter_value_and_grad(loss_fn)(module)
        updates = jax.tree.map(lambda gradient: -learning_rate * gradient if gradient is not None else None, gradients)
        module = eqx.apply_updates(module, updates)

    return module.sympy()[0]
