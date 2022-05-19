from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional


class Individual:
    genotype: Any
    phenotype: Any | None
    fitness: float | None = None

    def __init__(self, genotype: Any, fitness=None):
        self.genotype = genotype
        self.fitness = fitness

    def __str__(self) -> str:
        return str(self.genotype)
