from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Annotated, Any

from geneticengine.grammar.decorators import weight
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.vars import VarRange


def make_grammar(feature_names: list[str], classes: list[Any]) -> tuple[list[type], type]:

    @dataclass
    class Klass:
        value: Annotated[str, VarRange(classes)]

        def to_numpy(self) -> str:
            return f"{self.value}"

        def __str__(self):
            return f"Class={self.value}"

    class BExpression(ABC):

        @abstractmethod
        def to_numpy(self) -> str: ...

        def __str__(self) -> str:
            return self.to_numpy()

    class Expression(ABC):

        @abstractmethod
        def to_numpy(self) -> str: ...

        def __str__(self) -> str:
            return self.to_numpy()

    @weight(100)
    @dataclass
    class BinaryOp(BExpression):
        l: BExpression
        r: BExpression
        name: Annotated[str, VarRange(["logical_and", "logical_or", "logical_xor"])]

        def to_numpy(self) -> str:
            return f"np.{self.name}({self.l.to_numpy()}, {self.r.to_numpy()})"

    @weight(100)
    @dataclass
    class UnaryOp(BExpression):
        l: BExpression
        name: Annotated[str, VarRange(["logical_not"])]

        def to_numpy(self) -> str:
            return f"np.{self.name}({self.l.to_numpy()})"

    @weight(100)
    @dataclass
    class Cmp(BExpression):
        l: Expression
        r: Expression
        name: Annotated[str, VarRange(["<", "<=", "==", "!="])]

        def to_numpy(self) -> str:
            return f"({self.l.to_numpy()} {self.name} {self.r.to_numpy()})"

    @weight(5)
    @dataclass
    class Pi(Expression):

        def to_numpy(self) -> str:
            return "np.pi"

    @weight(5)
    @dataclass
    class E(Expression):

        def to_numpy(self) -> str:
            return "np.e"

    @weight(25)
    @dataclass
    class Zero(Expression):

        def to_numpy(self) -> str:
            return "0.0"

    @weight(25)
    @dataclass
    class One(Expression):

        def to_numpy(self) -> str:
            return "1.0"

    @weight(25)
    @dataclass
    class Two(Expression):

        def to_numpy(self) -> str:
            return "2.0"

    @weight(25)
    @dataclass
    class Ten(Expression):

        def to_numpy(self) -> str:
            return "10.0"

    @weight(100)
    @dataclass
    class Plus(Expression):
        l: Expression
        r: Expression

        def to_numpy(self) -> str:
            return f"({self.l.to_numpy()} + {self.r.to_numpy()})"

    @weight(50)
    @dataclass
    class Minus(Expression):
        l: Expression
        r: Expression

        def to_numpy(self) -> str:
            return f"({self.l.to_numpy()} - {self.r.to_numpy()})"

    @weight(100)
    @dataclass
    class Mult(Expression):
        l: Expression
        r: Expression

        def to_numpy(self) -> str:
            return f"({self.l.to_numpy()} * {self.r.to_numpy()})"

    @weight(25)
    @dataclass
    class FloatLiteral(Expression):
        value: float

        def to_numpy(self) -> str:
            return f"{self.value}"

    @dataclass
    class Var(Expression):
        name: Annotated[str, VarRange(feature_names)]
        # The list of vars should be filled in dynamically

        def to_numpy(self) -> str:
            return f"{self.name}"

        def __str__(self) -> str:
            return f"{self.name}"

    @dataclass
    class Rule:
        condition: BExpression
        answer: Klass

        def __str__(self):
            return f"IF ({self.condition}) THEN {self.answer}"

    @dataclass
    class RuleSet:
        rules: Annotated[list[Rule], ListSizeBetween(0, 20)]
        default: Klass

        def to_numpy(self) -> str:
            return functools.reduce(
                lambda v, rule: f"np.where({rule.condition.to_numpy()}, {rule.answer.to_numpy()}, {v})",
                self.rules,
                self.default.to_numpy(),
            )

        def to_sympy(self) -> str:
            return None

        def __str__(self):
            rules_ext = "; ".join(str(r) for r in self.rules)
            return f"{rules_ext}; default={self.default}"

    components = [
        BExpression,
        Expression,
        Rule,
        RuleSet,
        BinaryOp,
        UnaryOp,
        Cmp,
        Plus,
        Minus,
        Mult,
        Pi,
        E,
        Zero,
        One,
        Two,
        Ten,
        FloatLiteral,
        Klass,
        Var,  # must be last
    ]

    return components, RuleSet
