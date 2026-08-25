from pathlib import Path
from typing import get_args

import pytest

from geneticengine.grammar import PonyGE2GrammarError, load_ponyge2_grammar, parse_ponyge2_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.representations.tree.treebased import random_node


GRAMMAR = """
# The first rule is the start symbol.
<expr> ::= <expr> " + " <expr> | <value>  # recursive alternative
<value> ::= GE_RANGE:dataset_n_vars | 'x'
"""


def test_parse_ponyge2_grammar_generates_a_geneticengine_grammar():
    parsed = parse_ponyge2_grammar(GRAMMAR, ge_constants={"dataset_n_vars": 3})

    assert parsed.starting_symbol is parsed.nonterminals["<expr>"]
    assert parsed.grammar.starting_symbol is parsed.starting_symbol
    assert len(parsed.productions["<expr>"]) == 2
    assert parsed.to_geneticengine() == (parsed.nodes, parsed.starting_symbol)

    range_production = parsed.productions["<value>"][0]
    range_type = next(iter(range_production.__annotations__.values()))
    assert get_args(range_type)[1].__class__ is IntRange
    assert (get_args(range_type)[1].min, get_args(range_type)[1].max) == (0, 2)

    literal_production = parsed.productions["<value>"][1]
    literal_type = next(iter(literal_production.__annotations__.values()))
    assert get_args(literal_type)[1].__class__ is VarRange
    assert get_args(literal_type)[1].options == ["x"]


def test_generated_grammar_can_generate_a_tree():
    parsed = parse_ponyge2_grammar("<expr> ::= <value> + <value>\n<value> ::= GE_RANGE:2")
    random = NativeRandomSource(seed=1)

    individual = random_node(
        random,
        parsed.grammar,
        parsed.starting_symbol,
        MaxDepthDecider(random, parsed.grammar, max_depth=3),
    )

    assert isinstance(individual, parsed.starting_symbol)


def test_load_ponyge2_grammar_from_file(tmp_path: Path):
    path = tmp_path / "grammar.bnf"
    path.write_text("<start> ::= 'hello'")

    parsed = load_ponyge2_grammar(path)

    assert parsed.starting_symbol.__name__ == "Start"


def test_named_ge_range_requires_a_constant():
    with pytest.raises(PonyGE2GrammarError, match="ge_constants"):
        parse_ponyge2_grammar("<value> ::= GE_RANGE:dataset_n_vars")


def test_unknown_ge_constant_is_rejected():
    with pytest.raises(PonyGE2GrammarError, match="Unsupported"):
        parse_ponyge2_grammar("<value> ::= GE_UNKNOWN:3")
