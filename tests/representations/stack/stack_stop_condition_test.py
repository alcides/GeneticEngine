"""Stack mapping stops after one full genome pass with a start-symbol value."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from geneticengine.grammar.grammar import extract_grammar
from geneticengine.nxt.stackgggp import ListWrapper, StackBasedGGGPRepresentation, create_tree_using_stacks
from geneticengine.random.sources import NativeRandomSource


class Root(ABC):
    pass


@dataclass
class Leaf(Root):
    pass


@dataclass
class Node(Root):
    a: Root
    b: Root


def test_list_wrapper_tracks_consumed_across_wraps():
    w = ListWrapper(dna=[10, 20, 30])
    assert w.consumed == 0
    assert not w.completed_one_pass()
    w.randint(0, 9)
    w.randint(0, 9)
    w.randint(0, 9)
    assert w.consumed == 3
    assert w.completed_one_pass()
    w.randint(0, 9)  # wraps
    assert w.consumed == 4
    assert w.completed_one_pass()


def test_stack_mapping_requires_full_genome_pass_before_return():
    g = extract_grammar([Leaf, Node], Root)
    rep = StackBasedGGGPRepresentation(g, gene_length=64, failures_limit=10_000)
    geno = rep.create_genotype(NativeRandomSource(0))
    wrapper = ListWrapper(dna=list(geno.dna))
    tree = create_tree_using_stacks(g, wrapper, failures_limit=10_000)
    assert isinstance(tree, Root)
    assert wrapper.completed_one_pass()
    assert wrapper.consumed >= len(geno.dna)


def test_stack_mapping_continues_until_genome_consumed():
    """Even if a start-symbol value appears early, keep reading until one full pass."""
    g = extract_grammar([Leaf, Node], Root)
    rep = StackBasedGGGPRepresentation(g, gene_length=128, failures_limit=10_000)
    geno = rep.create_genotype(NativeRandomSource(1))
    wrapper = ListWrapper(dna=list(geno.dna))
    create_tree_using_stacks(g, wrapper, failures_limit=10_000)
    assert wrapper.consumed >= len(geno.dna)
