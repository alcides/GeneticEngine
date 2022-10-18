from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geneticengine.core.problems import FitnessType


@dataclass
class Individual:
    genotype: Any
    fitness: FitnessType | None = None

    def __str__(self) -> str:
        return str(self.genotype)
