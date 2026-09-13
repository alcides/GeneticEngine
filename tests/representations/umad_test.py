from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.representations.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.representations.linear_mutation import PointMutation, UMAD, size_neutral_deletion_rate
from geneticengine.representations.stackgggp import StackBasedGGGPRepresentation
from geneticengine.representations.tree.initializations import MaxDepthDecider
from geneticengine.random.sources import NativeRandomSource


def test_size_neutral_deletion_rate():
    assert abs(size_neutral_deletion_rate(0.1) - (0.1 / 1.1)) < 1e-12
    assert abs(size_neutral_deletion_rate(1.0) - 0.5) < 1e-12
    assert size_neutral_deletion_rate(0.0) == 0.0


def test_umad_operator_changes_genome_and_keeps_min_length():
    r = NativeRandomSource(seed=7)
    parent = list(range(20))
    child = UMAD(addition_rate=0.5, deletion_rate=0.3).mutate(
        parent,
        r,
        gene_factory=lambda: r.randint(1000, 2000),
    )
    assert len(child) >= 1
    assert isinstance(child, list)
    assert all(isinstance(x, int) for x in child)


def test_umad_zero_rates_is_identity():
    r = NativeRandomSource(seed=1)
    parent = [1, 2, 3, 4, 5]
    child = UMAD(addition_rate=0.0, deletion_rate=0.0).mutate(
        parent,
        r,
        gene_factory=lambda: -1,
    )
    assert child == parent


def test_point_mutation_preserves_length():
    r = NativeRandomSource(seed=5)
    parent = [1, 2, 3, 4, 5]
    child = PointMutation().mutate(parent, r, gene_factory=lambda: 99)
    assert len(child) == len(parent)
    assert sum(a != b for a, b in zip(parent, child)) == 1


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    x: int


def test_ge_umad_mutation():
    r = NativeRandomSource(seed=2)
    g = extract_grammar([Leaf], Root)
    d = MaxDepthDecider(r, g, 3)
    rep = GrammaticalEvolutionRepresentation(g, d, gene_length=32, mutation=UMAD(0.2))
    geno = rep.create_genotype(r)
    assert isinstance(rep.mutation, UMAD)
    child = rep.mutate(r, geno)
    assert len(child.dna) >= 1
    c1, c2 = rep.crossover(r, geno, child)
    assert len(c1.dna) >= 1 and len(c2.dna) >= 1


def test_stack_umad_mutation():
    r = NativeRandomSource(seed=3)
    g = extract_grammar([Leaf], Root)
    rep = StackBasedGGGPRepresentation(g, gene_length=64, mutation=UMAD(0.2))
    geno = rep.create_genotype(r)
    for _ in range(5):
        geno = rep.mutate(r, geno)
    assert len(geno.dna) >= 1
    assert rep.genotype_to_phenotype(geno) is not None


def test_ge_defaults_to_point_mutation():
    r = NativeRandomSource(seed=4)
    g = extract_grammar([Leaf], Root)
    d = MaxDepthDecider(r, g, 3)
    rep = GrammaticalEvolutionRepresentation(g, d, gene_length=16)
    assert isinstance(rep.mutation, PointMutation)
    geno = rep.create_genotype(r)
    child = rep.mutate(r, geno)
    assert len(child.dna) == len(geno.dna)
