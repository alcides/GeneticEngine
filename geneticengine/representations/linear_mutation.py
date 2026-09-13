"""Mutation operators for variable-length linear genomes.

Includes point mutation and Uniform Mutation by Addition and Deletion (UMAD)
(Helmuth, McPhee & Spector, GECCO 2018).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Generic, TypeVar

from geneticengine.random.sources import RandomSource

T = TypeVar("T")


def size_neutral_deletion_rate(addition_rate: float) -> float:
    """Deletion rate that keeps expected genome length unchanged."""
    if addition_rate < 0:
        raise ValueError(f"addition_rate must be >= 0, got {addition_rate}")
    return addition_rate / (1.0 + addition_rate)


class LinearGenomeMutation(Generic[T], abc.ABC):
    """Strategy for mutating a linear (list) genotype."""

    @abc.abstractmethod
    def mutate(
        self,
        genome: list[T],
        random: RandomSource,
        gene_factory: Callable[[], T],
    ) -> list[T]:
        """Return a mutated copy of ``genome``."""


class PointMutation(LinearGenomeMutation[T]):
    """Replace a single uniformly chosen gene with a new random gene."""

    def mutate(
        self,
        genome: list[T],
        random: RandomSource,
        gene_factory: Callable[[], T],
    ) -> list[T]:
        if not genome:
            return [gene_factory()]
        child = list(genome)
        index = random.randint(0, len(child) - 1)
        child[index] = gene_factory()
        return child


class UMAD(LinearGenomeMutation[T]):
    """Uniform Mutation by Addition and Deletion.

    Args:
        addition_rate: Probability of inserting one new gene adjacent to each
            existing gene during the addition pass (default 0.1).
        deletion_rate: Probability of removing each gene after addition. When
            ``None``, uses the size-neutral rate ``addition_rate / (1 + addition_rate)``.
        min_length: Ensure the child has at least this many genes (default 1).
    """

    def __init__(
        self,
        addition_rate: float = 0.1,
        deletion_rate: float | None = None,
        min_length: int = 1,
    ):
        if addition_rate < 0:
            raise ValueError(f"addition_rate must be >= 0, got {addition_rate}")
        if min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {min_length}")
        self.addition_rate = addition_rate
        self.deletion_rate = (
            size_neutral_deletion_rate(addition_rate) if deletion_rate is None else deletion_rate
        )
        if not (0.0 <= self.deletion_rate <= 1.0):
            raise ValueError(f"deletion_rate must be in [0, 1], got {self.deletion_rate}")
        self.min_length = min_length

    def mutate(
        self,
        genome: list[T],
        random: RandomSource,
        gene_factory: Callable[[], T],
    ) -> list[T]:
        augmented: list[T] = []
        for gene in genome:
            if random.random_float(0.0, 1.0) < self.addition_rate:
                new_gene = gene_factory()
                if random.random_float(0.0, 1.0) < 0.5:
                    augmented.append(new_gene)
                    augmented.append(gene)
                else:
                    augmented.append(gene)
                    augmented.append(new_gene)
            else:
                augmented.append(gene)

        child = [gene for gene in augmented if random.random_float(0.0, 1.0) >= self.deletion_rate]
        while len(child) < self.min_length:
            child.append(gene_factory())
        return child
