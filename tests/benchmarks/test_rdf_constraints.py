from examples.benchmarks.rdf_constraints import GRAPH
from examples.benchmarks.rdf_constraints import PAPER_BENCHMARKS
from examples.benchmarks.rdf_constraints import RdfConstraintsBenchmark
from examples.benchmarks.rdf_constraints import TypeShape
from examples.benchmarks.rdf_constraints import shape_fitness


def test_rdf_constraint_grammar_is_constructible():
    assert RdfConstraintsBenchmark().get_grammar() is not None


def test_known_paper_shape_has_zero_error():
    assert shape_fitness(TypeShape(0, 0)) == 0


def test_false_shape_has_positive_error():
    assert shape_fitness(TypeShape(0, 2)) > 0


def test_paper_benchmark_matrix_is_represented():
    assert len(PAPER_BENCHMARKS) == 21
    assert {benchmark.selection for benchmark in PAPER_BENCHMARKS} == {"roulette", "tournament"}
    assert len(GRAPH.triples) == 8
