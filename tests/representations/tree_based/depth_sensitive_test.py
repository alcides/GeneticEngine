from dataclasses import dataclass

from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


@abstract
class Expr:
    pass


@dataclass
class Leaf(Expr):
    value: int


@dataclass
class Branch(Expr):
    left: Expr
    right: Expr


def test_depth_sensitive_operations_are_opt_in():
    random = NativeRandomSource(4)
    grammar = extract_grammar([Leaf, Branch], Expr)
    decider = MaxDepthDecider(random, grammar, max_depth=4)
    representation = TreeBasedRepresentation(grammar, decider, max_operation_depth=1)
    parent = representation.create_genotype(random)

    mutated = representation.mutate(random, parent)
    crossed, _ = representation.crossover(random, parent, parent)

    assert mutated
    assert crossed


def test_negative_operation_depth_is_rejected():
    random = NativeRandomSource(1)
    grammar = extract_grammar([Leaf, Branch], Expr)
    decider = MaxDepthDecider(random, grammar, max_depth=2)
    try:
        TreeBasedRepresentation(grammar, decider, max_operation_depth=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative operation depth should be rejected")
