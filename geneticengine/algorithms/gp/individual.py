from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Individual(object):
    genotype: Any
    fitness: Optional[float] = None

    def __str__(self) -> str:
        return str(self.genotype)
