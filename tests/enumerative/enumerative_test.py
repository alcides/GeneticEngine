from abc import ABC
from dataclasses import dataclass

from geneticengine.algorithms.enumerative import iterate_individuals
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.decorators import weight

class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class Branch1(Root):
    v1: Root
    v2: Root


@dataclass
class Branch2(Root):
    v1: Root
    v2: Root

def test_enumerative():
    g = extract_grammar([Leaf, Branch1, Branch2], Root)
    exp = [Leaf(), Branch1(Leaf(), Leaf()), Branch2(Leaf(), Leaf())]

    xs = []
    for x in iterate_individuals(g, Root):
        xs.append(x.instance)
        if len(xs) > 10:
            break

    for expected, real in zip(exp, xs):
        assert expected == real



class GridExpression(ABC):
    pass

class TupleExpression(ABC):
    pass

@dataclass
class TupleFromGrid(TupleExpression):
    grid: GridExpression

    def __str__(self):
        return f"tuple_of {self.grid}"

@dataclass
@weight(3)
class Crop(GridExpression):
    grid: GridExpression
    tuple1: TupleExpression
    tuple2: TupleExpression

    def __str__(self):
        return f"crop_of ({self.grid}) ({self.tuple1}) ({self.tuple2})"

@dataclass
@weight(1)
class Drop(GridExpression):
    grid: GridExpression

    def __str__(self):
        return f"drop_of {self.grid}"

@dataclass
class GridFromTuple(GridExpression):
    tuple: TupleExpression

    def __str__(self):
        return f"grid_of {self.tuple}"

class InputGrid(GridExpression):

    def __str__(self):
        return "input"

def depth(n:GridExpression | TupleExpression) -> int:
    match n:
        case InputGrid():
            return 1
        case Crop(grid, t1, t2):
            return 1 + max(depth(grid), depth(t1), depth(t2))
        case Drop(grid):
            return 1 + depth(grid)
        case GridFromTuple(tuple):
            return 1 + depth(tuple)
        case TupleFromGrid(grid):
            return 1 + depth(grid)
        case _:
            assert False, f"Unknown {n}"

def test_enumerative_order():
    g = extract_grammar([Crop, Drop, InputGrid, TupleFromGrid, GridFromTuple], GridExpression)
    previous = -1

    for i, x in enumerate(iterate_individuals(g, GridExpression)):
        tree = x.get_phenotype()
        d = depth(tree)
        print(",", d, tree)
        assert d >= previous
        previous = d
        if i > 100:
            break
