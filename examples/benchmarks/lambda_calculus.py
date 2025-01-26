from abc import ABC
from dataclasses import dataclass
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.problems import Problem, SingleObjectiveProblem
from geneticengine.grammar.metahandlers.ints import IntRange

import sys
sys.setrecursionlimit(1500)



class Expression(ABC):
    pass

@dataclass
class Var(Expression):
    debruijn_index: Annotated[int, IntRange(0, 100)]

    def __str__(self) -> str:
        return f"{self.debruijn_index}"

@dataclass
class Abstraction(Expression):
    body : Expression

    def __str__(self) -> str:
        return f"\\{self.body}"

@dataclass
class Application(Expression):
    fun: Expression
    arg: Expression

    def __str__(self) -> str:
        return f"({self.fun} {self.arg})"

def increase_free_var(e:Expression, binders:int, n:int) -> Expression:
    match e:
        case Application(fun, arg):
            return Application(increase_free_var(fun, binders, n), increase_free_var(arg, binders, n))
        case Abstraction(body):
            return Abstraction(increase_free_var(body, binders+1, n))
        case Var(a):
            if a > binders:
                return Var(a+n)
            else:
                return Var(a)
        case _:
            assert False

def substitute(e:Expression, binders:int, x:Expression) -> Expression:
    match e:
        case Application(fun, arg):
            return Application(substitute(fun, binders, x), substitute(arg, binders, x))
        case Abstraction(body):
            return Abstraction(substitute(body, binders+1, x))
        case Var(a):
            if a == binders:
               return increase_free_var(x, 0, binders-1)
            elif a > binders:
               return Var(a-1)
            else:
               return Var(a)
        case _:
            assert False

def beta_reduction(e:Expression) -> Expression:
    match e:
        case Application(Abstraction(body), arg):
            return substitute(body, 1, arg)
        case Application(fun, arg):
            return Application(beta_reduction(fun), beta_reduction(arg))
        case Abstraction(body):
            return Abstraction(beta_reduction(body))
        case Var(index):
            return Var(index)
        case _:
            assert False

def reduction(e:Expression) -> Expression:
    """Fixpoint reduction"""
    current : Expression | None = None
    next = e
    while next != current:
        current = next
        next = beta_reduction(current)
    return next

# Test Reduction
r2 = Abstraction(Abstraction(Var(1)))
r1 = Abstraction(Abstraction(Var(2)))
r3 = Application(Application(r2 ,r1), r2)
assert r2 == reduction(r3)


def church(n:int):
    """Returns the Church numeral for the natural number n."""
    v : Expression = Var(1) # Zero
    for _ in range(1, n+1):
        v = Application(Var(2), v)
    return Abstraction(Abstraction(v))

plus_solution = Abstraction( #m
                Abstraction( #n
                Abstraction( #f
                Abstraction( #x
                    Application(
                        Application(Var(4), Var(2)), # m f
                        Application(Application( Var(3), Var(2)), Var(1)),
                    ), # n f x
                ),
                ),
                ),
)


def test_plus(expr:Expression, a1:int, a2:int) -> bool:
    tree = Application(Application(expr, church(a1)), church(a2))
    v = reduction(tree)
    return v == church(a1 + a2)


def test_numbers(expr:Expression) -> int:
    return sum([
        int(not test_plus(expr, i, j)) for i in range(10) for j in range(10)
    ])

assert church(0) == Abstraction(Abstraction(Var(1)))
assert church(1) == Abstraction(Abstraction(Application(Var(2), Var(1))))
assert test_plus(plus_solution, 0, 0)
assert test_plus(plus_solution, 10, 8)

def max_var(e:Expression) -> int:
    match e:
        case Var(n):
            return n
        case Application(fun, arg):
            return max(max_var(fun), max_var(arg))
        case Abstraction(body):
            return max_var(body)-1
        case _:
            assert False

def fix_variables(e:Expression) -> Expression:
    n = max_var(e)
    r = e
    for _ in range(n):
        r = Abstraction(r)
    return r

class LambdaCalculusBenchmark(Benchmark):
    def __init__(self):
        self.setup_problem()
        self.setup_grammar()

    def setup_problem(self):

        # Problem
        def fitness_function(b: Expression) -> int:
            b2 = fix_variables(b)
            try:
                r = test_numbers(b2)
            except RecursionError:
                r = 1000
            return r

        self.problem = SingleObjectiveProblem(minimize=True, fitness_function=fitness_function, target=0)

    def setup_grammar(self):
        self.grammar = extract_grammar(
            [Abstraction, Application, Var],
            Expression,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":

    example_run(LambdaCalculusBenchmark())
