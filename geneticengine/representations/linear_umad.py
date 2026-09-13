"""Uniform Mutation by Addition and Deletion (UMAD) for linear genomes.

Based on Helmuth, McPhee & Spector, GECCO 2018:
"Program synthesis using uniform mutation by addition and deletion".

UMAD walks a variable-length linear genome, optionally inserting a new gene
before or after each existing gene, then independently deleting genes. For a
given addition rate ``a``, the size-neutral deletion rate is ``a / (1 + a)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from geneticengine.random.sources import RandomSource

T = TypeVar("T")


def size_neutral_deletion_rate(addition_rate: float) -> float:
    """Deletion rate that keeps expected genome length unchanged."""
    if addition_rate < 0:
        raise ValueError(f"addition_rate must be >= 0, got {addition_rate}")
    return addition_rate / (1.0 + addition_rate)


def umad(
    genome: list[T],
    random: RandomSource,
    *,
    addition_rate: float = 0.1,
    deletion_rate: float | None = None,
    gene_factory: Callable[[], T],
    min_length: int = 1,
) -> list[T]:
    """Apply UMAD to a linear genome.

    Args:
        genome: Parent gene list (not mutated in place).
        random: Random source.
        addition_rate: Probability of inserting one new gene adjacent to each
            existing gene during the addition pass.
        deletion_rate: Probability of removing each gene after addition. When
            ``None``, uses the size-neutral rate ``addition_rate / (1 + addition_rate)``.
        gene_factory: Zero-arg callable that produces a new random gene.
        min_length: Ensure the child has at least this many genes (default 1).
            If deletions would empty the genome, fresh genes are inserted.
    """
    if addition_rate < 0:
        raise ValueError(f"addition_rate must be >= 0, got {addition_rate}")
    if deletion_rate is None:
        deletion_rate = size_neutral_deletion_rate(addition_rate)
    if not (0.0 <= deletion_rate <= 1.0):
        raise ValueError(f"deletion_rate must be in [0, 1], got {deletion_rate}")
    if min_length < 0:
        raise ValueError(f"min_length must be >= 0, got {min_length}")

    # Addition pass: every original gene is kept; optionally insert one neighbour.
    augmented: list[T] = []
    for gene in genome:
        if random.random_float(0.0, 1.0) < addition_rate:
            new_gene = gene_factory()
            if random.random_float(0.0, 1.0) < 0.5:
                augmented.append(new_gene)
                augmented.append(gene)
            else:
                augmented.append(gene)
                augmented.append(new_gene)
        else:
            augmented.append(gene)

    # Deletion pass.
    child = [gene for gene in augmented if random.random_float(0.0, 1.0) >= deletion_rate]

    while len(child) < min_length:
        child.append(gene_factory())
    return child
