"""Regression coverage for the supported Z3 range and Aeon dependency set."""

from importlib.metadata import requires

import dill
import pytest
import z3
from packaging.requirements import Requirement

from geneticengine.grammar.metahandlers.smt.parser import p_expr


def test_smt_parser_translation_and_solver():
    expression = p_expr("x > 4 && x < 6")
    expression = dill.loads(dill.dumps(expression))
    constraint = expression.translate({"x": "x"}, {"x": z3.Int})
    solver = z3.Solver()
    solver.add(constraint)
    assert solver.check() == z3.sat
    assert solver.model().eval(z3.Int("x")).as_long() == 5
    solver.add(z3.Int("x") == 6)
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("version", ["4.15.0.0", "4.15.3.0", "5.1.0.0"])
def test_published_z3_requirement_accepts_supported_versions(version):
    dependency = next(Requirement(r) for r in requires("GeneticEngine") if Requirement(r).name == "z3-solver")
    assert version in dependency.specifier


@pytest.mark.parametrize("name,version", [("dill", "0.4.1"), ("lark", "1.3.1"), ("pathos", "0.3.5"), ("pytest", "9.1.1")])
def test_published_requirements_accept_aeon_dependencies(name, version):
    dependency = next(Requirement(r) for r in requires("GeneticEngine") if Requirement(r).name == name)
    assert version in dependency.specifier
