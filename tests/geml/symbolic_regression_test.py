import numpy as np
from sklearn.metrics import r2_score
from geml.grammars.symbolic_regression import (
    Pi,
    Two,
    components,
    Zero,
    Minus,
    Plus,
    Mult,
    One,
    Expression,
    Pow,
    make_var,
)
from sympy.parsing.sympy_parser import parse_expr
from sympy.simplify import simplify

from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.problems import SingleObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


def forward_dataset_expr(e: Expression, dataset) -> float:
    return eval(e.to_numpy(), {"x": dataset, "np": np})


def test_symbolic_regression_sympy_simp():
    s = Plus(Zero(), Minus(One(), Mult(One(), One())))
    v = s.to_sympy()
    e = parse_expr(v)
    e2 = e.simplify()
    assert str(e2) == "0"


def test_symbolic_regression_gp() -> None:
    r = NativeRandomSource(seed=1)

    # Grammar setup
    vars = ["x"]
    Var = make_var(vars, 5000)
    g: Grammar = extract_grammar(components + [Var], Expression)

    # Dataset setup
    target_expression = Plus(
        Pi(),
        Pow(Var("x"), Two()),
    )
    dataset = np.arange(-1000.0, 1000.0, 1.0).reshape(2000, 1)
    y_true = forward_dataset_expr(target_expression, dataset)

    def fitness_function(x: Expression) -> float:
        symbolic = x.to_sympy()
        e = parse_expr(f"({target_expression}) - ({symbolic})")
        s = simplify(e)
        if s == 0:
            return 1.0
        elif "x" not in symbolic:
            return -100000000
        else:
            try:
                y_pred = forward_dataset_expr(x, dataset)
                return r2_score(y_true, y_pred)
            except ValueError as e:
                return -10000000

    p = SingleObjectiveProblem(fitness_function=fitness_function)
    gp = GeneticProgramming(
        representation=TreeBasedRepresentation(g, ProgressivelyTerminalDecider(r, g)),
        problem=p,
        random=r,
        budget=EvaluationBudget(100),
    )
    ind = gp.search()[0]
    assert ind.get_fitness(p).fitness_components[0] <= 1
