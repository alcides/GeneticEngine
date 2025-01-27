from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Annotated


from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import SingleObjectiveProblem, Problem


class Boolean(ABC):
    pass

class Float(ABC):
    pass

@dataclass
class BooleanLiteral(Boolean):
    val : bool

@dataclass
class FloatLiteral(Float):
    val : float

@dataclass
class FloatVar(Float):
    val : Annotated[int, IntRange(0, 2)]

@dataclass
class Not(Boolean):
    e: Boolean

@dataclass
class And(Boolean):
    a: Boolean
    b: Boolean

@dataclass
class Or(Boolean):
    a: Boolean
    b: Boolean

@dataclass
class LessThan(Boolean):
    a : Float
    b : Float

@dataclass
class GreaterThan(Boolean):
    a : Float
    b : Float

@dataclass
class Plus(Float):
    a : Float
    b : Float

@dataclass
class Minus(Float):
    a : Float

@dataclass
class Mul(Float):
    a : Float
    b : Float

@dataclass
class Div(Float):
    a : Float
    b : Float

@dataclass
class If(Float):
    c : Boolean
    a : Float
    b : Float


def eval(e:Float | Boolean, args:list[float]) -> float|bool:
    match e:
        case BooleanLiteral(v):
            return v
        case Not(b):
            return not eval(b, args)
        case And(ab, bb):
            return eval(ab, args) and eval(bb, args)
        case Or(ab, bb):
            return eval(ab, args) and eval(bb, args)
        case LessThan(f1, f2):
            return eval(f1, args) < eval(f2, args)
        case GreaterThan(f1, f2):
            return eval(f1, args) > eval(f2, args)
        case FloatLiteral(vf):
            return vf
        case FloatVar(index):
            return args[index]
        case Plus(f1, f2):
            return eval(f1, args) + eval(f2, args)
        case Minus(f):
            return -eval(f, args)
        case Mul(f1, f2):
            return eval(f1, args) * eval(f2, args)
        case Div(f1, f2):
            return eval(f1, args) / eval(f2, args)
        case If(b, f1, f2):
            if eval(b, args):
                return eval(f1, args)
            else:
                return eval(f2, args)
        case _:
            assert False

def true_median(ls):
    return sum(ls) / len(ls)

def eval_wrapper(expr:Float, args:list[float]) -> float:
    try:
        return eval(expr, args)
    except ZeroDivisionError:
        return 10000

vals_to_consider = [0.0, 0.5, 1.0, 2.0, -1.0, 3.0, 1000.4, -23.2]
def test_eval(expr:Float) -> float:
    return sum([
        abs(eval_wrapper(expr, [i, j, k]) - true_median([i, j, k]))
        for i in vals_to_consider
        for j in vals_to_consider
        for k in vals_to_consider
    ])

solution = Div(Plus(Plus(FloatVar(0), FloatVar(1)), FloatVar(2)), FloatLiteral(3))

class MedianBenchmark(Benchmark):
    def __init__(self):
        self.setup_problem()
        self.setup_grammar()

    def setup_problem(self):

        # Problem
        def fitness_function(b: Float) -> float:
            return test_eval(b)

        self.problem = SingleObjectiveProblem(minimize=True, fitness_function=fitness_function, target=0)

    def setup_grammar(self):
        self.grammar = extract_grammar(
            [FloatLiteral, BooleanLiteral, FloatVar, Not, And, Or, Plus, Minus, Mul, Div, If, LessThan, GreaterThan],
            Float,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(MedianBenchmark())
