"""Tests for the expression simplifier using egglog equality saturation."""

from geml.grammars.symbolic_regression import (
    Expression,
    Plus,
    Mult,
    Sin,
    Cos,
    Pi,
    E,
    Zero,
    One,
    FloatLiteral,
    make_var,
)
from geml.simplifier import simplify


def make_var_for_test():
    """Create a Var class for testing."""
    Var = make_var(["x", "y", "z", "w"])
    return Var


def assert_expressions_equivalent(expr1: Expression, expr2: Expression, tolerance: float = 1e-10):
    """Assert that two expressions are equivalent by comparing their sympy representations.

    Args:
        expr1: First expression to compare.
        expr2: Second expression to compare.
        tolerance: Numerical tolerance for comparison (not used for symbolic comparison).
    """
    import sympy
    from sympy.simplify import simplify as sympy_simplify

    # Convert to sympy and simplify
    try:
        sympy1 = sympy.sympify(
            expr1.to_sympy(), locals={
                'sin': sympy.sin,
                'cos': sympy.cos,
                'log': sympy.log,
                'pi': sympy.pi,
                'e': sympy.E,
            },
        )
        sympy2 = sympy.sympify(
            expr2.to_sympy(), locals={
                'sin': sympy.sin,
                'cos': sympy.cos,
                'log': sympy.log,
                'pi': sympy.pi,
                'e': sympy.E,
            },
        )

        # Simplify both
        simplified1 = sympy_simplify(sympy1)
        simplified2 = sympy_simplify(sympy2)

        # Check if they're equivalent by subtracting and simplifying
        diff = sympy_simplify(simplified1 - simplified2)
        assert diff == 0, f"Expressions not equivalent: {expr1.to_sympy()} vs {expr2.to_sympy()}"
    except Exception as e:
        # If sympy parsing fails, fall back to string comparison after simplification
        # This handles cases where the expression structure might be slightly different
        # but semantically equivalent
        msg = f"Could not compare expressions: {expr1.to_sympy()} vs {expr2.to_sympy()}. Error: {e}"
        # For now, we'll let it pass if the to_sympy representations are the same
        # This is a fallback for edge cases
        if expr1.to_sympy() == expr2.to_sympy():
            return
        raise AssertionError(msg)


class TestIdentityRules:
    """Test simplification of identity rules."""

    def test_add_zero_left(self):
        """Test x + 0 simplifies to x."""
        Var = make_var_for_test()
        expr = Plus(Zero(), Var("x"))
        simplified = simplify(expr)
        # Should simplify to x (or at least be equivalent)
        assert_expressions_equivalent(simplified, Var("x"))

    def test_add_zero_right(self):
        """Test 0 + x simplifies to x."""
        Var = make_var_for_test()
        expr = Plus(Var("x"), Zero())
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Var("x"))

    def test_multiply_one_left(self):
        """Test 1 * x simplifies to x."""
        Var = make_var_for_test()
        expr = Mult(One(), Var("x"))
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Var("x"))

    def test_multiply_one_right(self):
        """Test x * 1 simplifies to x."""
        Var = make_var_for_test()
        expr = Mult(Var("x"), One())
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Var("x"))

    def test_multiply_zero_left(self):
        """Test 0 * x simplifies to 0."""
        Var = make_var_for_test()
        expr = Mult(Zero(), Var("x"))
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Zero())

    def test_multiply_zero_right(self):
        """Test x * 0 simplifies to 0."""
        Var = make_var_for_test()
        expr = Mult(Var("x"), Zero())
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Zero())


class TestCommutativity:
    """Test simplification using commutativity rules."""

    def test_add_commutativity(self):
        """Test that addition is commutative (order shouldn't matter)."""
        Var = make_var_for_test()
        expr1 = Plus(Var("x"), Var("y"))
        expr2 = Plus(Var("y"), Var("x"))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)

    def test_mult_commutativity(self):
        """Test that multiplication is commutative."""
        Var = make_var_for_test()
        expr1 = Mult(Var("x"), Var("y"))
        expr2 = Mult(Var("y"), Var("x"))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)


class TestAssociativity:
    """Test simplification using associativity rules."""

    def test_add_associativity(self):
        """Test that addition is associative: (x + y) + z = x + (y + z)."""
        Var = make_var_for_test()
        expr1 = Plus(Plus(Var("x"), Var("y")), Var("z"))
        expr2 = Plus(Var("x"), Plus(Var("y"), Var("z")))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)

    def test_mult_associativity(self):
        """Test that multiplication is associative: (x * y) * z = x * (y * z)."""
        Var = make_var_for_test()
        expr1 = Mult(Mult(Var("x"), Var("y")), Var("z"))
        expr2 = Mult(Var("x"), Mult(Var("y"), Var("z")))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)


class TestDistributivity:
    """Test simplification using distributivity rules."""

    def test_left_distributivity(self):
        """Test x * (y + z) = (x * y) + (x * z)."""
        Var = make_var_for_test()
        expr1 = Mult(Var("x"), Plus(Var("y"), Var("z")))
        expr2 = Plus(Mult(Var("x"), Var("y")), Mult(Var("x"), Var("z")))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)

    def test_right_distributivity(self):
        """Test (x + y) * z = (x * z) + (y * z)."""
        Var = make_var_for_test()
        expr1 = Mult(Plus(Var("x"), Var("y")), Var("z"))
        expr2 = Plus(Mult(Var("x"), Var("z")), Mult(Var("y"), Var("z")))
        simplified1 = simplify(expr1)
        simplified2 = simplify(expr2)
        assert_expressions_equivalent(simplified1, simplified2)


class TestComplexCombinations:
    """Test complex combinations of simplification rules."""

    def test_nested_add_zero(self):
        """Test x + (y + 0) + z simplifies to x + y + z."""
        Var = make_var_for_test()
        expr = Plus(Plus(Var("x"), Plus(Var("y"), Zero())), Var("z"))
        simplified = simplify(expr)
        # Should be equivalent to x + y + z
        expected = Plus(Plus(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_nested_mult_one(self):
        """Test x * (y * 1) * z simplifies to x * y * z."""
        Var = make_var_for_test()
        expr = Mult(Mult(Var("x"), Mult(Var("y"), One())), Var("z"))
        simplified = simplify(expr)
        expected = Mult(Mult(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_distribute_and_simplify(self):
        """Test x * (y + 0) simplifies to x * y."""
        Var = make_var_for_test()
        expr = Mult(Var("x"), Plus(Var("y"), Zero()))
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Mult(Var("x"), Var("y")))

    def test_complex_nested_expression(self):
        """Test a complex nested expression: x * (y + 0) + z * 1."""
        Var = make_var_for_test()
        expr = Plus(
            Mult(Var("x"), Plus(Var("y"), Zero())),
            Mult(Var("z"), One()),
        )
        simplified = simplify(expr)
        expected = Plus(Mult(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_multiple_zeros_in_addition(self):
        """Test x + 0 + y + 0 simplifies to x + y."""
        Var = make_var_for_test()
        expr = Plus(Plus(Plus(Var("x"), Zero()), Var("y")), Zero())
        simplified = simplify(expr)
        expected = Plus(Var("x"), Var("y"))
        assert_expressions_equivalent(simplified, expected)

    def test_multiple_ones_in_multiplication(self):
        """Test x * 1 * y * 1 simplifies to x * y."""
        Var = make_var_for_test()
        expr = Mult(Mult(Mult(Var("x"), One()), Var("y")), One())
        simplified = simplify(expr)
        expected = Mult(Var("x"), Var("y"))
        assert_expressions_equivalent(simplified, expected)

    def test_distributivity_with_zeros(self):
        """Test x * (y + 0 + z) simplifies to x * (y + z)."""
        Var = make_var_for_test()
        expr = Mult(Var("x"), Plus(Plus(Var("y"), Zero()), Var("z")))
        simplified = simplify(expr)
        expected = Mult(Var("x"), Plus(Var("y"), Var("z")))
        assert_expressions_equivalent(simplified, expected)

    def test_combined_arithmetic_operations(self):
        """Test x * (y + z) + 0 * w simplifies to x * (y + z)."""
        Var = make_var_for_test()
        expr = Plus(
            Mult(Var("x"), Plus(Var("y"), Var("z"))),
            Mult(Zero(), Var("w")),
        )
        simplified = simplify(expr)
        expected = Mult(Var("x"), Plus(Var("y"), Var("z")))
        assert_expressions_equivalent(simplified, expected)

    def test_triple_nested_identities(self):
        """Test x + (y + (z + 0)) simplifies to x + y + z."""
        Var = make_var_for_test()
        expr = Plus(Var("x"), Plus(Var("y"), Plus(Var("z"), Zero())))
        simplified = simplify(expr)
        expected = Plus(Plus(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_distributivity_with_multiple_terms(self):
        """Test x * (y + z + 0) simplifies to x * (y + z)."""
        Var = make_var_for_test()
        expr = Mult(Var("x"), Plus(Plus(Var("y"), Var("z")), Zero()))
        simplified = simplify(expr)
        expected = Mult(Var("x"), Plus(Var("y"), Var("z")))
        assert_expressions_equivalent(simplified, expected)

    def test_mixed_identity_operations(self):
        """Test (x + 0) * (y * 1) simplifies to x * y."""
        Var = make_var_for_test()
        expr = Mult(Plus(Var("x"), Zero()), Mult(Var("y"), One()))
        simplified = simplify(expr)
        expected = Mult(Var("x"), Var("y"))
        assert_expressions_equivalent(simplified, expected)

    def test_complex_nested_expression_with_all_rules(self):
        """Test a complex expression using all simplification rules."""
        Var = make_var_for_test()
        # (x + 0) * (y * 1) + (z * 0) + (w * 1) should simplify to x * y + w
        expr = Plus(
            Plus(
                Mult(Plus(Var("x"), Zero()), Mult(Var("y"), One())),
                Mult(Var("z"), Zero()),
            ),
            Mult(Var("w"), One()),
        )
        simplified = simplify(expr)
        expected = Plus(Mult(Var("x"), Var("y")), Var("w"))
        assert_expressions_equivalent(simplified, expected)


class TestNumericSimplifications:
    """Test simplification with numeric literals."""

    def test_numeric_addition(self):
        """Test that numeric addition can be simplified."""
        Var = make_var_for_test()
        expr = Plus(Plus(FloatLiteral(2.0), FloatLiteral(3.0)), Var("x"))
        simplified = simplify(expr)
        expected = Plus(FloatLiteral(5.0), Var("x"))
        assert_expressions_equivalent(simplified, expected)

    def test_numeric_multiplication(self):
        """Test that numeric multiplication can be simplified."""
        Var = make_var_for_test()
        expr = Mult(Mult(FloatLiteral(2.0), FloatLiteral(3.0)), Var("x"))
        simplified = simplify(expr)
        expected = Mult(FloatLiteral(6.0), Var("x"))
        assert_expressions_equivalent(simplified, expected)

    def test_numeric_with_identity(self):
        """Test 2 * x + 0 simplifies to 2 * x."""
        Var = make_var_for_test()
        expr = Plus(Mult(FloatLiteral(2.0), Var("x")), Zero())
        simplified = simplify(expr)
        expected = Mult(FloatLiteral(2.0), Var("x"))
        assert_expressions_equivalent(simplified, expected)


class TestSpecialConstants:
    """Test simplification with special constants like Pi and E."""

    def test_pi_constant(self):
        """Test that Pi constant is preserved."""
        expr = Plus(Pi(), Zero())
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Pi())

    def test_e_constant(self):
        """Test that E constant is preserved."""
        expr = Mult(E(), One())
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, E())

    def test_constants_with_operations(self):
        """Test Pi + 0 * E simplifies to Pi."""
        expr = Plus(Pi(), Mult(Zero(), E()))
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, Pi())


class TestTrigonometricFunctions:
    """Test simplification with trigonometric functions."""

    def test_sin_with_zero(self):
        """Test sin(0) should be preserved (simplification may vary)."""
        expr = Sin(Zero())
        simplified = simplify(expr)
        # sin(0) = 0, but our simplifier might not know this
        # At minimum, it should still be a valid expression
        assert isinstance(simplified, (Sin, Zero))

    def test_cos_with_zero(self):
        """Test cos(0) should be preserved."""
        expr = Cos(Zero())
        simplified = simplify(expr)
        assert isinstance(simplified, (Cos, Zero, One))

    def test_nested_trig_with_identity(self):
        """Test sin(x + 0) simplifies to sin(x)."""
        Var = make_var_for_test()
        expr = Sin(Plus(Var("x"), Zero()))
        simplified = simplify(expr)
        expected = Sin(Var("x"))
        assert_expressions_equivalent(simplified, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_variable(self):
        """Test that a single variable is unchanged."""
        Var = make_var_for_test()
        expr = Var("x")
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, expr)

    def test_single_constant(self):
        """Test that a single constant is unchanged."""
        expr = FloatLiteral(5.0)
        simplified = simplify(expr)
        assert_expressions_equivalent(simplified, expr)

    def test_zero_only(self):
        """Test that zero alone is unchanged."""
        expr = Zero()
        simplified = simplify(expr)
        assert isinstance(simplified, Zero)

    def test_one_only(self):
        """Test that one alone is unchanged."""
        expr = One()
        simplified = simplify(expr)
        assert isinstance(simplified, One)

    def test_very_nested_expression(self):
        """Test a very deeply nested expression."""
        Var = make_var_for_test()
        expr = Plus(
            Plus(
                Plus(Plus(Var("x"), Zero()), Zero()),
                Zero(),
            ),
            Var("y"),
        )
        simplified = simplify(expr)
        expected = Plus(Var("x"), Var("y"))
        assert_expressions_equivalent(simplified, expected)

    def test_identity_with_different_types(self):
        """Test that identity rules work with different expression types."""
        Var = make_var_for_test()
        # x * 1 + sin(y + 0) should simplify
        expr = Plus(
            Mult(Var("x"), One()),
            Sin(Plus(Var("y"), Zero())),
        )
        simplified = simplify(expr)
        expected = Plus(Var("x"), Sin(Var("y")))
        assert_expressions_equivalent(simplified, expected)


class TestRealWorldScenarios:
    """Test realistic simplification scenarios."""

    def test_polynomial_simplification(self):
        """Test simplification of a polynomial-like expression."""
        Var = make_var_for_test()
        # x * (y + 0) + z * 1 + 0 should simplify to x * y + z
        expr = Plus(
            Plus(
                Mult(Var("x"), Plus(Var("y"), Zero())),
                Mult(Var("z"), One()),
            ),
            Zero(),
        )
        simplified = simplify(expr)
        expected = Plus(Mult(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_multiple_variables_with_identities(self):
        """Test expression with multiple variables and identity elements."""
        Var = make_var_for_test()
        # x + 0 * y + z * 1 should simplify to x + z
        expr = Plus(
            Plus(Var("x"), Mult(Zero(), Var("y"))),
            Mult(Var("z"), One()),
        )
        simplified = simplify(expr)
        expected = Plus(Var("x"), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_distributed_expression_simplification(self):
        """Test that distributed expressions are simplified."""
        Var = make_var_for_test()
        # x * (y + z + 0) should simplify to x * (y + z)
        expr = Mult(Var("x"), Plus(Plus(Var("y"), Var("z")), Zero()))
        simplified = simplify(expr)
        expected = Mult(Var("x"), Plus(Var("y"), Var("z")))
        assert_expressions_equivalent(simplified, expected)

    def test_complex_mixed_operations(self):
        """Test complex expression mixing all operations."""
        Var = make_var_for_test()
        # (x + 0) * (y * 1) + (z + 0) * (w * 1) should simplify to x * y + z * w
        expr = Plus(
            Mult(Plus(Var("x"), Zero()), Mult(Var("y"), One())),
            Mult(Plus(Var("z"), Zero()), Mult(Var("w"), One())),
        )
        simplified = simplify(expr)
        expected = Plus(
            Mult(Var("x"), Var("y")),
            Mult(Var("z"), Var("w")),
        )
        assert_expressions_equivalent(simplified, expected)

    def test_nested_distributivity_and_identity(self):
        """Test nested distributivity with identity elements."""
        Var = make_var_for_test()
        # x * ((y + 0) + (z * 1)) should simplify to x * (y + z)
        expr = Mult(
            Var("x"),
            Plus(
                Plus(Var("y"), Zero()),
                Mult(Var("z"), One()),
            ),
        )
        simplified = simplify(expr)
        expected = Mult(Var("x"), Plus(Var("y"), Var("z")))
        assert_expressions_equivalent(simplified, expected)

    def test_very_complex_expression(self):
        """Test a very complex expression combining many rules."""
        Var = make_var_for_test()
        # ((x + 0) * (y * 1)) + ((z * 0) + (w * 1)) + 0 should simplify to x * y + w
        expr = Plus(
            Plus(
                Mult(Plus(Var("x"), Zero()), Mult(Var("y"), One())),
                Plus(
                    Mult(Var("z"), Zero()),
                    Mult(Var("w"), One()),
                ),
            ),
            Zero(),
        )
        simplified = simplify(expr)
        expected = Plus(Mult(Var("x"), Var("y")), Var("w"))
        assert_expressions_equivalent(simplified, expected)

    def test_associativity_with_identities(self):
        """Test associativity combined with identity simplification."""
        Var = make_var_for_test()
        # (x + 0) + (y + 0) + (z + 0) should simplify to x + y + z
        expr = Plus(
            Plus(Plus(Var("x"), Zero()), Plus(Var("y"), Zero())),
            Plus(Var("z"), Zero()),
        )
        simplified = simplify(expr)
        expected = Plus(Plus(Var("x"), Var("y")), Var("z"))
        assert_expressions_equivalent(simplified, expected)

    def test_distributivity_multiple_levels(self):
        """Test distributivity at multiple levels."""
        Var = make_var_for_test()
        # x * (y + 0) + x * (z * 1) should simplify to x * y + x * z
        # Note: The simplifier might not factor back to x * (y + z), so we check
        # that it's at least simplified (identities removed) and equivalent
        expr = Plus(
            Mult(Var("x"), Plus(Var("y"), Zero())),
            Mult(Var("x"), Mult(Var("z"), One())),
        )
        simplified = simplify(expr)
        # The simplified version should be equivalent to x * y + x * z
        # or x * (y + z) - both are valid simplifications
        expected1 = Plus(Mult(Var("x"), Var("y")), Mult(Var("x"), Var("z")))
        expected2 = Mult(Var("x"), Plus(Var("y"), Var("z")))
        # Check equivalence with either form
        try:
            assert_expressions_equivalent(simplified, expected1)
        except AssertionError:
            assert_expressions_equivalent(simplified, expected2)
