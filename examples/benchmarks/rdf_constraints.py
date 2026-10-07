"""Mine simple SHACL-style RDF type constraints with GeneticEngine.

This is a small, dependency-free reproduction of the benchmark shape used by
Felin et al. (EuroGP 2024).  The paper mines ``rdf:type`` associations from
the Covid-on-the-Web graph; this example embeds a tiny representative graph so
the benchmark is reproducible without a SPARQL service or a downloaded dump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.problems import Problem, SingleObjectiveProblem

RDF_TYPE = "rdf:type"


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class RdfGraph:
    triples: tuple[Triple, ...]

    def objects(self, subject: str, predicate: str) -> set[str]:
        return {t.object for t in self.triples if t.subject == subject and t.predicate == predicate}


GRAPH = RdfGraph(
    (
        Triple("alice", RDF_TYPE, "Person"),
        Triple("alice", RDF_TYPE, "Researcher"),
        Triple("bob", RDF_TYPE, "Person"),
        Triple("bob", RDF_TYPE, "Researcher"),
        Triple("carol", RDF_TYPE, "Person"),
        Triple("carol", RDF_TYPE, "Student"),
        Triple("dave", RDF_TYPE, "Person"),
        Triple("erin", RDF_TYPE, "Animal"),
    ),
)


@dataclass
class TypeShape:
    """A SHACL node shape with one ``rdf:type``/``sh:hasValue`` constraint."""

    target_class: Annotated[int, IntRange(0, 2)]
    value_class: Annotated[int, IntRange(0, 3)]

    def __str__(self) -> str:
        return (
            "a sh:NodeShape ; sh:targetClass "
            f"<{CLASSES[self.target_class]}> ; sh:property [ sh:path rdf:type ; "
            f"sh:hasValue <{CLASSES[self.value_class]}> ; ] ."
        )


CLASSES = ("Person", "Researcher", "Student", "Animal")


def shape_fitness(shape: TypeShape, graph: RdfGraph = GRAPH) -> float:
    """Return the paper-inspired association-rule error for a candidate shape."""
    target = CLASSES[shape.target_class]
    value = CLASSES[shape.value_class]
    subjects = {t.subject for t in graph.triples if t.predicate == RDF_TYPE and t.object == target}
    if not subjects:
        return 1.0
    confirmations = sum(value in graph.objects(subject, RDF_TYPE) for subject in subjects)
    violations = len(subjects) - confirmations
    return violations / len(subjects)


@dataclass(frozen=True)
class PaperBenchmark:
    """One reproducible configuration from the paper's experiment matrix."""

    name: str
    population_size: int
    effort: int
    selection: str
    selection_rate: float
    tournament_rate: float | None = None


PAPER_BENCHMARKS = tuple(
    [
        PaperBenchmark(f"V1_{population}_{effort}", population, effort, "roulette", 0.4)
        for population in (100, 200, 500)
        for effort in (5_000, 10_000, 20_000)
    ]
    + [
        PaperBenchmark(f"V2_Roulette_{rate:g}", 100, 20_000, "roulette", rate)
        for rate in (0.2, 0.4, 0.6)
    ]
    + [
        PaperBenchmark(
            f"V2_Tournament_{rate:g}_{tournament:g}",
            100,
            20_000,
            "tournament",
            rate,
            tournament,
        )
        for rate in (0.2, 0.4, 0.6)
        for tournament in (0.1, 0.25, 0.5)
    ]
)


class RdfConstraintsBenchmark(Benchmark):
    """Benchmark wrapper for the embedded RDF constraint-mining instance."""

    def __init__(self):
        self.problem = SingleObjectiveProblem(shape_fitness, minimize=True, target=0)
        self.grammar = extract_grammar([TypeShape], TypeShape)

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(RdfConstraintsBenchmark())
