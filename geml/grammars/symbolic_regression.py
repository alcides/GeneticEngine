from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar.decorators import weight
from geneticengine.grammar.metahandlers.vars import VarRange


class Expression(ABC):
    @abstractmethod
    def to_sympy(self) -> str: ...

    @abstractmethod
    def to_numpy(self) -> str: ...

    def __str__(self) -> str:
        return self.to_sympy()


@weight(100)
@dataclass
class Plus(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} + {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} + {self.r.to_numpy()})"


@weight(50)
@dataclass
class Minus(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} - {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} - {self.r.to_numpy()})"


@weight(100)
@dataclass
class Mult(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} * {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"({self.l.to_numpy()} * {self.r.to_numpy()})"


@weight(100)
@dataclass
class SafeDiv(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} / {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"(lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b!=0.0))({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(100)
@dataclass
class Pow(Expression):
    l: Expression
    r: Expression

    def to_sympy(self) -> str:
        return f"({self.l.to_sympy()} ** {self.r.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.power({self.l.to_numpy()}, {self.r.to_numpy()})"


@weight(10)
@dataclass
class Sin(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"sin({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.sin({self.e.to_numpy()})"


@weight(10)
@dataclass
class Cos(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"cos({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.cos({self.e.to_numpy()})"


@weight(10)
@dataclass
class Log(Expression):
    e: Expression

    def to_sympy(self) -> str:
        return f"log({self.e.to_sympy()})"

    def to_numpy(self) -> str:
        return f"np.log({self.e.to_numpy()})"


@weight(5)
@dataclass
class Pi(Expression):
    def to_sympy(self) -> str:
        return "pi"

    def to_numpy(self) -> str:
        return "np.pi"


@weight(5)
@dataclass
class E(Expression):
    def to_sympy(self) -> str:
        return "e"

    def to_numpy(self) -> str:
        return "np.e"


@weight(25)
@dataclass
class Zero(Expression):
    def to_sympy(self) -> str:
        return "0.0"

    def to_numpy(self) -> str:
        return "0.0"


@weight(25)
@dataclass
class One(Expression):
    def to_sympy(self) -> str:
        return "1.0"

    def to_numpy(self) -> str:
        return "1.0"


@weight(25)
@dataclass
class Two(Expression):
    def to_sympy(self) -> str:
        return "2.0"

    def to_numpy(self) -> str:
        return "2.0"


@weight(25)
@dataclass
class Ten(Expression):
    def to_sympy(self) -> str:
        return "10.0"

    def to_numpy(self) -> str:
        return "10.0"


@weight(25)
@dataclass
class FloatLiteral(Expression):
    value: float

    def to_sympy(self) -> str:
        return f"{self.value}"

    def to_numpy(self) -> str:
        return f"{self.value}"


def make_var(options: list[str], relative_weight: float = 1):
    @weight(relative_weight)
    @dataclass
    class Var(Expression):
        name: Annotated[str, VarRange(options)]
        # The list of vars should always be filled in dynamically

        def to_sympy(self) -> str:
            return f"{self.name}"

        def to_numpy(self) -> str:
            return f"{self.name}"

    return Var


components = [Plus, Minus, Mult, SafeDiv, Pow, Sin, Cos, Log, Pi, E, Zero, One, Two, Ten]
