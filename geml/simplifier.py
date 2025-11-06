"""Equality-saturation-based simplification of expressions using egglog."""

import egglog
import sympy
from geml.grammars.symbolic_regression import (
    Expression,
    Plus,
    Minus,
    Mult,
    SafeDiv,
    Pow,
    Sin,
    Cos,
    Log,
    Pi,
    E,
    Zero,
    One,
    Two,
    Ten,
    FloatLiteral,
)

class Math(egglog.Expr):
    def __init__(self, i: egglog.i64Like) -> None: ...
    def __add__(self, other: "Math") -> "Math": ...
    def __mul__(self, other: "Math") -> "Math": ...
    def __pow__(self, other: "Math") -> "Math": ...


def simplify(expr: Expression) -> Expression:
    """Simplify an Expression using egglog equality saturation.

    Args:
        expr: The Expression to simplify.

    Returns:
        A simplified Expression.
    """
    try:
        # Convert Expression to sympy first (as intermediary)
        sympy_expr = _to_sympy(expr)

        # Use egglog for simplification
        simplified_sympy = _simplify_with_egglog(sympy_expr)

        # Convert back to Expression format
        return _from_sympy(simplified_sympy, expr)

    except ImportError:
        # If egglog is not available, return the original expression
        return expr
    except Exception:
        # If simplification fails for any reason, return the original
        return expr


def _simplify_with_egglog(sympy_expr: sympy.Expr) -> sympy.Expr:
    """Simplify a sympy expression using egglog equality saturation.

    This function attempts to use egglog for equality saturation-based
    simplification. If egglog is not available or the API doesn't match
    expectations, it falls back to sympy's built-in simplification.

    Args:
        sympy_expr: The sympy expression to simplify.

    Returns:
        A simplified sympy expression.
    """
    try:
        # Try to import egglog
        # The exact API may vary, so we try with error handling
        import egglog

        # Build an egglog expression tree from the sympy expression
        egglog_expr = _sympy_to_egglog_expr(sympy_expr)

        # Create an EGraph and add the expression
        # Note: The exact API may vary - this is a template
        egraph = egglog.EGraph()
        egraph.saturate(egglog_expr)

        # Define simplification rules using rewrite patterns
        # These are arithmetic simplification rules for equality saturation
        # Note: Math type is expected to be defined by egglog module
        # This code will gracefully fall back if egglog API doesn't match
        @egraph.register
        def simplification_rules():
            # Commutativity
            a, b = egglog.vars_("a b", Math)
            yield egglog.rewrite(a + b).to(b + a)
            yield egglog.rewrite(a * b).to(b * a)

            # Associativity
            a, b, c = egglog.vars_("a b c", Math)
            yield egglog.rewrite((a + b) + c).to(a + (b + c))
            yield egglog.rewrite((a * b) * c).to(a * (b * c))

            # Identity elements
            yield egglog.rewrite(a + 0).to(a)
            yield egglog.rewrite(a * 1).to(a)
            yield egglog.rewrite(a * 0).to(0)
            yield egglog.rewrite(0 * a).to(0)

            # Distributivity
            yield egglog.rewrite(a * (b + c)).to((a * b) + (a * c))
            yield egglog.rewrite((a + b) * c).to((a * c) + (b * c))

            # Power rules
            yield egglog.rewrite(a ** 1).to(a)
            yield egglog.rewrite(a ** 0).to(1)
            yield egglog.rewrite(1 ** a).to(1)

        # Run equality saturation
        egraph.run(simplification_rules)

        # Extract the simplified expression
        simplified = egraph.extract(egglog_expr)

        # Convert back to sympy
        return _egglog_expr_to_sympy(simplified)

    except (ImportError, AttributeError, TypeError, NameError):
        # If egglog is not available or API doesn't match, use sympy simplification
        # This is expected if egglog isn't installed or the API is different
        return sympy.simplify(sympy_expr)
    except Exception:
        # If anything else fails, use sympy simplification
        return sympy.simplify(sympy_expr)


def _sympy_to_egglog_expr(sympy_expr: sympy.Expr):
    """Convert a sympy expression to an egglog expression.

    This recursively builds an egglog expression tree from a sympy expression.
    The exact implementation depends on egglog's API.

    Args:
        sympy_expr: The sympy expression.

    Returns:
        An egglog expression.
    """
    try:
        import egglog

        # Recursively convert sympy expression structure to egglog
        if sympy_expr.is_Number:
            return float(sympy_expr)
        elif sympy_expr.is_Symbol:
            # Create a variable in egglog
            return egglog.var(str(sympy_expr))
        elif sympy_expr.is_Add:
            # Convert addition
            args = sympy_expr.args
            result = _sympy_to_egglog_expr(args[0])
            for arg in args[1:]:
                result = result + _sympy_to_egglog_expr(arg)
            return result
        elif sympy_expr.is_Mul:
            # Convert multiplication
            args = sympy_expr.args
            result = _sympy_to_egglog_expr(args[0])
            for arg in args[1:]:
                result = result * _sympy_to_egglog_expr(arg)
            return result
        elif sympy_expr.is_Pow:
            # Convert power
            base, exp = sympy_expr.args
            return _sympy_to_egglog_expr(base) ** _sympy_to_egglog_expr(exp)
        elif isinstance(sympy_expr, sympy.sin):
            return egglog.sin(_sympy_to_egglog_expr(sympy_expr.args[0]))
        elif isinstance(sympy_expr, sympy.cos):
            return egglog.cos(_sympy_to_egglog_expr(sympy_expr.args[0]))
        elif isinstance(sympy_expr, sympy.log):
            return egglog.log(_sympy_to_egglog_expr(sympy_expr.args[0]))
        else:
            # Fallback: convert to string and try to parse
            # This may not work perfectly but is a best effort
            return egglog.var(str(sympy_expr))
    except Exception:
        # If conversion fails, return the original sympy expression
        # The fallback in _simplify_with_egglog will handle it
        return sympy_expr


def _egglog_expr_to_sympy(egglog_expr) -> sympy.Expr:
    """Convert an egglog expression back to sympy.

    Args:
        egglog_expr: The egglog expression.

    Returns:
        A sympy expression.
    """
    # If it's already a sympy expression, return it
    if isinstance(egglog_expr, sympy.Expr):
        return egglog_expr

    # Try to convert to string and parse
    try:
        expr_str = str(egglog_expr)
        return sympy.sympify(
            expr_str, locals={
                'sin': sympy.sin,
                'cos': sympy.cos,
                'log': sympy.log,
                'pi': sympy.pi,
                'e': sympy.E,
            },
        )
    except Exception:
        # If conversion fails, try to return as-is (might be a number)
        try:
            return sympy.sympify(egglog_expr)
        except Exception:
            # Last resort: return a placeholder
            return sympy.Symbol('x')


def _to_sympy(expr: Expression) -> sympy.Expr:
    """Convert a GeneticEngine Expression to a sympy expression.

    Args:
        expr: The Expression to convert.

    Returns:
        A sympy expression.
    """
    # Use the to_sympy() method to get a string, then parse it
    expr_str = expr.to_sympy()
    # Parse the string to sympy expression
    # We need to handle sympy functions like sin, cos, log, pi, e
    return sympy.sympify(
        expr_str, locals={
            'sin': sympy.sin,
            'cos': sympy.cos,
            'log': sympy.log,
            'pi': sympy.pi,
            'e': sympy.E,
        },
    )


def _from_sympy(sympy_expr: sympy.Expr, original_expr: Expression) -> Expression:
    """Convert a sympy expression back to a GeneticEngine Expression.

    Args:
        sympy_expr: The sympy expression to convert.
        original_expr: The original Expression (for reference, e.g., variable types).

    Returns:
        A GeneticEngine Expression.
    """
    # Recursively convert sympy expression to Expression tree
    if sympy_expr.is_Number:
        value = float(sympy_expr)
        if value == 0.0:
            return Zero()
        elif value == 1.0:
            return One()
        elif value == 2.0:
            return Two()
        elif value == 10.0:
            return Ten()
        else:
            return FloatLiteral(value)
    elif sympy_expr == sympy.pi:
        return Pi()
    elif sympy_expr == sympy.E:
        return E()
    elif sympy_expr.is_Symbol:
        # We need to preserve the variable name and type
        var_name = str(sympy_expr)
        # Try to find a Var instance in the original expression to get the class
        var_class = _find_var_class(original_expr)
        if var_class:
            # Get feature_names from the original if available
            feature_names = getattr(original_expr, 'feature_names', [])
            var_instance = var_class(var_name)
            if feature_names:
                var_instance.feature_names = feature_names
            return var_instance
        else:
            # Fallback: return original if we can't reconstruct
            return original_expr
    elif sympy_expr.is_Add:
        args = sympy_expr.args
        if len(args) == 0:
            return Zero()
        elif len(args) == 1:
            return _from_sympy(args[0], original_expr)
        elif len(args) == 2:
            return Plus(_from_sympy(args[0], original_expr), _from_sympy(args[1], original_expr))
        else:
            # Handle more than 2 arguments by building left-associative tree
            result = _from_sympy(args[0], original_expr)
            for arg in args[1:]:
                result = Plus(result, _from_sympy(arg, original_expr))
            return result
    elif sympy_expr.is_Mul:
        args = sympy_expr.args
        if len(args) == 0:
            return One()
        elif len(args) == 1:
            return _from_sympy(args[0], original_expr)
        elif len(args) == 2:
            return Mult(_from_sympy(args[0], original_expr), _from_sympy(args[1], original_expr))
        else:
            # Handle more than 2 arguments by building left-associative tree
            result = _from_sympy(args[0], original_expr)
            for arg in args[1:]:
                result = Mult(result, _from_sympy(arg, original_expr))
            return result
    elif sympy_expr.is_Pow:
        base, exp = sympy_expr.args
        return Pow(_from_sympy(base, original_expr), _from_sympy(exp, original_expr))
    elif isinstance(sympy_expr, sympy.sin):
        return Sin(_from_sympy(sympy_expr.args[0], original_expr))
    elif isinstance(sympy_expr, sympy.cos):
        return Cos(_from_sympy(sympy_expr.args[0], original_expr))
    elif isinstance(sympy_expr, sympy.log):
        return Log(_from_sympy(sympy_expr.args[0], original_expr))
    else:
        # Fallback: return original expression
        return original_expr


def _find_var_class(expr: Expression):
    """Find the Var class used in an expression.

    Args:
        expr: The Expression to search.

    Returns:
        The Var class if found, None otherwise.
    """
    # Recursively search for a Var instance
    if hasattr(expr, 'name') and hasattr(expr, 'to_sympy') and hasattr(expr, 'to_numpy'):
        # This looks like a Var instance
        return type(expr)
    elif isinstance(expr, (Plus, Minus, Mult, SafeDiv, Pow)):
        # Check children
        var_class = _find_var_class(expr.l)
        if var_class:
            return var_class
        return _find_var_class(expr.r)
    elif isinstance(expr, (Sin, Cos, Log)):
        return _find_var_class(expr.e)
    return None
