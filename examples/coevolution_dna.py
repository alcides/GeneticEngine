from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

from geneticengine.algorithms.gp.cooperativegp import CooperativeGP
from geneticengine.evaluation.budget import EvaluationBudget
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.grammar.metahandlers.strings import StringSizeBetween

from scipy.stats import entropy


def blackbox_classifier(s: str) -> float:
    aapos = s.find("aa")
    ttpos = s.find("tt")
    if aapos == -1 and ttpos == -1:
        return 0
    elif (aapos == -1) or (ttpos == -1):
        return 0.1
    elif aapos > ttpos:
        return 0.2
    else:
        return (ttpos - aapos) / len(s)


@dataclass
class Line:
    str: Annotated[str, StringSizeBetween(20, 40, ["a", "t", "c", "g"])]

    def __repr__(self):
        return self.str


@dataclass
class Dataset:
    lines: Annotated[list[Line], ListSizeBetween(100, 100)]

    def __repr__(self):
        return " ".join(repr(x) for x in self.lines)


def dataset_fitness_function(d: Dataset):
    fitnesses = [blackbox_classifier(line.str) for line in d.lines]
    return entropy(fitnesses)


class Explanation(ABC):
    @abstractmethod
    def eval(self, line: str) -> bool: ...


@dataclass
class Pattern:
    pattern: Annotated[str, StringSizeBetween(3, 30, ["a", "t", "c", "g"])]

    def eval(self, line: str):
        return self.pattern


@dataclass
class Contains(Explanation):
    pattern: Pattern

    def eval(self, line: str) -> bool:
        return self.pattern.eval(line) in line


@dataclass
class Before(Explanation):
    pattern1: Pattern
    pattern2: Pattern

    def eval(self, line: str) -> bool:
        apos = line.find(self.pattern1.eval(line))
        bpos = line.find(self.pattern2.eval(line))
        if apos != -1 and bpos != -1:
            return apos > bpos
        else:
            return False


@dataclass
class And(Explanation):
    one: Explanation
    other: Explanation

    def eval(self, line: str) -> bool:
        return self.one.eval(line) and self.other.eval(line)


@dataclass
class Or(Explanation):
    one: Explanation
    other: Explanation

    def eval(self, line: str) -> bool:
        return self.one.eval(line) or self.other.eval(line)


def eval_explanation_on_dataset(d: Dataset, e: Explanation) -> float:
    correct_fitnesses = [blackbox_classifier(line.str) >= 0.5 for line in d.lines]
    predicted_fitnesses = [e.eval(line.str) for line in d.lines]
    return sum(1 if a == b else 0 for a, b in zip(correct_fitnesses, predicted_fitnesses))


dataset_grammar = extract_grammar([Line, Dataset], Dataset)

explanation_grammar = extract_grammar([Contains, Before, Pattern, And, Or], Explanation)


alg = CooperativeGP(
    dataset_grammar,
    explanation_grammar,
    eval_explanation_on_dataset,
    population1_size=50,
    population2_size=500,
    coevolutions=10,
    kwargs1={
        "budget": EvaluationBudget(10000),
    },
    kwargs2={
        "budget": EvaluationBudget(10000),
    },
)
x: tuple[Dataset, Explanation] = alg.search()
best_dataset, best_explanation = x
print("Best dataset:", best_dataset)
print("Best explanation:", best_explanation)
