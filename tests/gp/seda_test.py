from abc import ABC

from geneticengine.algorithms.gp.operators.seda import SEDA
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import extract_grammar
from geneticengine.random.sources import NativeRandomSource


@abstract
class Choice(ABC):
    pass


class Good(Choice):
    pass


class Bad(Choice):
    pass


def test_seda_smooths_towards_elite_productions():
    grammar = extract_grammar([Good, Bad], Choice)
    model = SEDA(grammar, elite_fraction=0.5, smoothing=1)
    model.estimate([(10, {Choice: [Bad]}), (1, {Choice: [Good]})])
    probabilities = model.probabilities[Choice]
    assert probabilities[0] > probabilities[1]
    assert sum(probabilities) == 1


def test_seda_sampling_uses_the_learned_distribution():
    grammar = extract_grammar([Good, Bad], Choice)
    model = SEDA(grammar, elite_fraction=1, smoothing=0)
    model.estimate([(1, {Choice: [Good]})])
    assert model.sample(NativeRandomSource(1), Choice) is Good
