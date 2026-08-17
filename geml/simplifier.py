"""Equality-saturation-based simplification of expressions using egglog."""

from __future__ import annotations

import re
from typing import Any

import sympy
from egglog import EGraph, Expr, StringLike, f64Like, rewrite, ruleset, vars_

from geml.grammars.symbolic_regression import (
    Cos,
    E,
    Expression,
    FloatLiteral,
    Log,
    Minus,
    Mult,
    One,
    Pi,
    Plus,
    Pow,
    SafeDiv,
    Sin,
    Ten,
    Two,
    Zero,
)

_initial_egraph = EGraph()


class MathNum(Expr):
    def __init__(self, value: f64Like) -> None: ...

    @classmethod
    def var(cls, name: StringLike) -> MathNum: ...

    @classmethod
    def sin(cls, arg: MathNum) -> MathNum: ...

    @classmethod
    def cos(cls, arg: MathNum) -> MathNum: ...

    @classmethod
    def log(cls, arg: MathNum) -> MathNum: ...

    def __add__(self, other: MathNum) -> MathNum: ...

    def __sub__(self, other: MathNum) -> MathNum: ...

    def __mul__(self, other: MathNum) -> MathNum: ...

    def __truediv__(self, other: MathNum) -> MathNum: ...

    def __pow__(self, other: MathNum) -> MathNum: ...


def _build_identity_rules():
    a, = vars_("a", MathNum)
    zero = MathNum(0.0)
    one = MathNum(1.0)
    return ruleset(
        rewrite(a + zero).to(a),
        rewrite(zero + a).to(a),
        rewrite(a * one).to(a),
        rewrite(one * a).to(a),
        rewrite(a * zero).to(zero),
        rewrite(zero * a).to(zero),
    ) * 5


IDENTITY_RULES = _build_identity_rules()

_SYMPY_LOCALS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "log": sympy.log,
    "pi": sympy.pi,
    "e": sympy.E,
}

_EGGLOG_VAR_PATTERN = re.compile(r'MathNum\.var\("([^"]+)"\)')
_EGGLOG_NUM_PATTERN = re.compile(r"MathNum\(([^)]+)\)")
_EGGLOG_FUNC_PATTERNS = (
    (re.compile(r"MathNum\.sin\((.+)\)"), r"sin(\1)"),
    (re.compile(r"MathNum\.cos\((.+)\)"), r"cos(\1)"),
    (re.compile(r"MathNum\.log\((.+)\)"), r"log(\1)"),
)


def simplify(expr: Expression) -> Expression:
    """Simplify a symbolic-regression Expression using egglog and sympy."""
    try:
        egglog_expr = _to_egglog(expr)
        egraph = EGraph()
        egraph.register(egglog_expr)
        egraph.run(IDENTITY_RULES)
        saturated = egraph.extract(egglog_expr)
        sympy_expr = sympy.simplify(_egglog_to_sympy(saturated))
        return _from_sympy(sympy_expr, expr)
    except Exception:
        try:
            return _from_sympy(sympy.simplify(_to_sympy(expr)), expr)
        except Exception:
            return expr


def _to_egglog(expr: Expression) -> Any:
    if isinstance(expr, Zero):
        return MathNum(0.0)
    if isinstance(expr, One):
        return MathNum(1.0)
    if isinstance(expr, Two):
        return MathNum(2.0)
    if isinstance(expr, Ten):
        return MathNum(10.0)
    if isinstance(expr, FloatLiteral):
        return MathNum(float(expr.value))
    if isinstance(expr, Pi):
        return MathNum.var("pi")
    if isinstance(expr, E):
        return MathNum.var("e")
    if hasattr(expr, "name") and hasattr(expr, "to_numpy"):
        return MathNum.var(expr.name)
    if isinstance(expr, Plus):
        return _to_egglog(expr.l) + _to_egglog(expr.r)
    if isinstance(expr, Minus):
        return _to_egglog(expr.l) - _to_egglog(expr.r)
    if isinstance(expr, Mult):
        return _to_egglog(expr.l) * _to_egglog(expr.r)
    if isinstance(expr, SafeDiv):
        return _to_egglog(expr.l) / _to_egglog(expr.r)
    if isinstance(expr, Pow):
        return _to_egglog(expr.l) ** _to_egglog(expr.r)
    if isinstance(expr, Sin):
        return MathNum.sin(_to_egglog(expr.e))
    if isinstance(expr, Cos):
        return MathNum.cos(_to_egglog(expr.e))
    if isinstance(expr, Log):
        return MathNum.log(_to_egglog(expr.e))
    raise TypeError(f"Unsupported expression type: {type(expr)}")


def _egglog_to_sympy(egglog_expr: Any) -> sympy.Expr:
    expr_str = str(egglog_expr)
    expr_str = _EGGLOG_VAR_PATTERN.sub(r"\1", expr_str)
    expr_str = _EGGLOG_NUM_PATTERN.sub(r"\1", expr_str)
    for pattern, replacement in _EGGLOG_FUNC_PATTERNS:
        expr_str = pattern.sub(replacement, expr_str)
    return sympy.sympify(expr_str, locals=_SYMPY_LOCALS)


def _to_sympy(expr: Expression) -> sympy.Expr:
    return sympy.sympify(expr.to_sympy(), locals=_SYMPY_LOCALS)


def _from_sympy(sympy_expr: sympy.Expr, original_expr: Expression) -> Expression:
    if sympy_expr.is_Number:
        value = float(sympy_expr)
        if value == 0.0:
            return Zero()
        if value == 1.0:
            return One()
        if value == 2.0:
            return Two()
        if value == 10.0:
            return Ten()
        return FloatLiteral(value)
    if sympy_expr == sympy.pi:
        return Pi()
    if sympy_expr == sympy.E:
        return E()
    if sympy_expr.is_Symbol:
        var_name = str(sympy_expr)
        var_class = _find_var_class(original_expr)
        if var_class:
            var_instance = var_class(var_name)
            feature_names = getattr(original_expr, "feature_names", [])
            if feature_names:
                var_instance.feature_names = feature_names
            return var_instance
        return original_expr
    if sympy_expr.is_Add:
        args = sympy_expr.args
        if len(args) == 0:
            return Zero()
        if len(args) == 1:
            return _from_sympy(args[0], original_expr)
        result = _from_sympy(args[0], original_expr)
        for arg in args[1:]:
            result = Plus(result, _from_sympy(arg, original_expr))
        return result
    if sympy_expr.is_Mul:
        args = sympy_expr.args
        if len(args) == 0:
            return One()
        if len(args) == 1:
            return _from_sympy(args[0], original_expr)
        result = _from_sympy(args[0], original_expr)
        for arg in args[1:]:
            result = Mult(result, _from_sympy(arg, original_expr))
        return result
    if sympy_expr.is_Pow:
        base, exp = sympy_expr.args
        return Pow(_from_sympy(base, original_expr), _from_sympy(exp, original_expr))
    if isinstance(sympy_expr, sympy.sin):
        return Sin(_from_sympy(sympy_expr.args[0], original_expr))
    if isinstance(sympy_expr, sympy.cos):
        return Cos(_from_sympy(sympy_expr.args[0], original_expr))
    if isinstance(sympy_expr, sympy.log):
        return Log(_from_sympy(sympy_expr.args[0], original_expr))
    return original_expr


def _find_var_class(expr: Expression):
    if hasattr(expr, "name") and hasattr(expr, "to_sympy") and hasattr(expr, "to_numpy"):
        return type(expr)
    if isinstance(expr, (Plus, Minus, Mult, SafeDiv, Pow)):
        var_class = _find_var_class(expr.l)
        if var_class:
            return var_class
        return _find_var_class(expr.r)
    if isinstance(expr, (Sin, Cos, Log)):
        return _find_var_class(expr.e)
    return None
