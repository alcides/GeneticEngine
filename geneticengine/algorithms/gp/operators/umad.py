"""Variable-length mutation operators for linear genomes.

UMAD is Uniform Mutation by Addition and Deletion.  SARMS is its
self-adaptive-rate variant: the mutation rate is carried by the individual
and perturbed before it is used to mutate the genome.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

from geneticengine.random.sources import RandomSource

T = TypeVar("T")


def umad_mutate(
    genome: Sequence[T],
    random: RandomSource,
    gene_factory: Callable[[], T],
    rate: float = 0.1,
) -> list[T]:
    """Apply size-neutral Uniform Mutation by Addition and Deletion.

    For each original gene, ``floor(rate)`` genes are inserted plus one more
    with probability equal to the fractional part.  Every gene in the
    augmented genome is then deleted independently with probability
    ``rate / (1 + rate)``, which preserves the expected genome size.
    """
    if rate < 0 or not math.isfinite(rate):
        raise ValueError("UMAD rate must be finite and non-negative")
    source = list(genome)
    additions: list[T] = []
    whole = math.floor(rate)
    fractional = rate - whole
    for _ in source:
        additions.extend(gene_factory() for _ in range(whole))
        if random.random_float(0.0, 1.0) < fractional:
            additions.append(gene_factory())

    augmented: list[T] = []
    for index, gene in enumerate(source):
        augmented.append(gene)
        augmented.extend(additions[index * whole : (index + 1) * whole]) if whole else None
    # Fractional additions are inserted uniformly between genes.
    if fractional:
        for gene in additions[whole * len(source) :]:
            position = random.randint(0, len(augmented))
            augmented.insert(position, gene)

    deletion_probability = rate / (1.0 + rate)
    return [gene for gene in augmented if random.random_float(0.0, 1.0) >= deletion_probability]


def sarms_mutate(
    genome: Sequence[T],
    random: RandomSource,
    gene_factory: Callable[[], T],
    rate: float = 0.1,
    meta_rate: float = 0.1,
    minimum_rate: float = 1e-6,
    maximum_rate: float = 10.0,
) -> tuple[list[T], float]:
    """Mutate a genome and its rate using a log-normal self-adaptation step."""
    if not 0 < meta_rate or not math.isfinite(meta_rate):
        raise ValueError("SARMS meta_rate must be finite and positive")
    if not minimum_rate <= rate <= maximum_rate:
        raise ValueError("SARMS rate must be within the configured bounds")
    updated_rate = rate * math.exp(meta_rate * _standard_normal(random))
    updated_rate = min(maximum_rate, max(minimum_rate, updated_rate))
    return umad_mutate(genome, random, gene_factory, updated_rate), updated_rate


def _standard_normal(random: RandomSource) -> float:
    """Generate a standard normal using only the RandomSource interface."""
    u1 = max(random.random_float(0.0, 1.0), 1e-12)
    u2 = random.random_float(0.0, 1.0)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
