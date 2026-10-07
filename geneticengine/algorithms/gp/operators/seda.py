"""Smoothed estimation of distribution for grammar-guided GP.

The model is deliberately independent of a particular genotype encoding:
representations provide the production choices made by an individual, and
SEDA turns elite choices into smoothed production probabilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from geneticengine.grammar.grammar import Grammar
from geneticengine.random.sources import RandomSource


@dataclass
class SEDA:
    """Learn a smoothed distribution over grammar productions.

    ``samples`` contains ``(fitness, choices)`` pairs.  Each choices mapping
    maps a non-terminal to the productions used while deriving one individual.
    Lower fitness is considered better when ``minimize`` is true.
    """

    grammar: Grammar
    elite_fraction: float = 0.2
    smoothing: float = 0.1
    minimize: bool = True
    probabilities: dict[Any, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1]")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.probabilities = {
            nonterminal: [1 / len(productions)] * len(productions)
            for nonterminal, productions in self.grammar.alternatives.items()
            if productions
        }

    def estimate(self, samples: Iterable[tuple[float, Mapping[Any, Sequence[Any]]]]) -> None:
        """Estimate production probabilities from the best sample fraction."""
        observations = list(samples)
        if not observations:
            return
        observations.sort(key=lambda sample: sample[0], reverse=not self.minimize)
        elite_count = max(1, round(len(observations) * self.elite_fraction))
        elite = observations[:elite_count]
        for nonterminal, productions in self.grammar.alternatives.items():
            if not productions:
                continue
            counts = [self.smoothing] * len(productions)
            for _, choices in elite:
                for choice in choices.get(nonterminal, []):
                    if choice in productions:
                        counts[productions.index(choice)] += 1
            total = sum(counts)
            self.probabilities[nonterminal] = [count / total for count in counts]

    def sample(self, random: RandomSource, nonterminal: Any) -> Any:
        """Sample a production for ``nonterminal`` from the learned model."""
        productions = self.grammar.alternatives[nonterminal]
        probabilities = self.probabilities[nonterminal]
        point = random.random_float(0, 1)
        for production, probability in zip(productions, probabilities):
            point -= probability
            if point <= 0:
                return production
        return productions[-1]
