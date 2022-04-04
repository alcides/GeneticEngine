from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class Individual:
    genotype: Any
    fitness: float | None = None

    def __str__(self) -> str:
        return str(self.genotype)
