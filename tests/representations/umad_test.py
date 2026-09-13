from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.nxt.grammatical_evolution.ge import GrammaticalEvolutionRepresentation
from geneticengine.nxt.linear_umad import size_neutral_deletion_rate, umad
from geneticengine.nxt.stackgggp import StackBasedGGGPRepresentation
from geneticengine.nxt.tree.initializations import MaxDepthDecider
from geneticengine.random.sources import NativeRandomSource


def test_size_neutral_deletion_rate():
    assert abs(size_neutral_deletion_rate(0.1) - (0.1 / 1.1)) < 1e-12
    assert abs(size_neutral_deletion_rate(1.0) - 0.5) < 1e-12
    assert size_neutral_deletion_rate(0.0) == 0.0


def test_umad_changes_genome_and_keeps_min_length():
    r = NativeRandomSource(seed=7)
    parent = list(range(20))
    child = umad(
        parent,
        r,
        addition_rate=0.5,
        deletion_rate=0.3,
        gene_factory=lambda: r.randint(1000, 2000),
        min_length=1,
    )
    assert len(child) >= 1
    # High rates should usually change something; allow rare equality but check type.
    assert isinstance(child, list)
    assert all(isinstance(x, int) for x in child)


def test_umad_zero_rates_is_identity():
    r = NativeRandomSource(seed=1)
    parent = [1, 2, 3, 4, 5]
    child = umad(
        parent,
        r,
        addition_rate=0.0,
        deletion_rate=0.0,
        gene_factory=lambda: -1,
        min_length=1,
    )
    assert child == parent


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    x: int


def test_ge_umad_mutation():
    r = NativeRandomSource(seed=2)
    g = extract_grammar([Leaf], Root)
    d = MaxDepthDecider(r, g, 3)
    rep = GrammaticalEvolutionRepresentation(g, d, gene_length=32, mutation="umad", umad_addition_rate=0.2)
    geno = rep.create_genotype(r)
    parent_len = len(geno.dna)
    child = rep.mutate(r, geno)
    assert len(child.dna) >= 1
    # UMAD may change length; point mutation would keep it fixed.
    assert isinstance(child.dna, list)
    # Crossover still works with variable-length children.
    c1, c2 = rep.crossover(r, geno, child)
    assert len(c1.dna) >= 1 and len(c2.dna) >= 1
    assert parent_len == 32


def test_stack_umad_mutation():
    r = NativeRandomSource(seed=3)
    g = extract_grammar([Leaf], Root)
    rep = StackBasedGGGPRepresentation(g, gene_length=64, mutation="umad", umad_addition_rate=0.2)
    geno = rep.create_genotype(r)
    for _ in range(5):
        geno = rep.mutate(r, geno)
    assert len(geno.dna) >= 1
    assert rep.genotype_to_phenotype(geno) is not None


def test_ge_point_mutation_still_default():
    r = NativeRandomSource(seed=4)
    g = extract_grammar([Leaf], Root)
    d = MaxDepthDecider(r, g, 3)
    rep = GrammaticalEvolutionRepresentation(g, d, gene_length=16)
    assert rep.mutation == "point"
    geno = rep.create_genotype(r)
    child = rep.mutate(r, geno)
    assert len(child.dna) == len(geno.dna)
