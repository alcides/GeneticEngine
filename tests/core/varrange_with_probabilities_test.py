from typing import Any, Callable, Generator


from geneticengine.grammar.metahandlers.vars import VarRangeWithProbabilities
from geneticengine.random.sources import NativeRandomSource


def test_generate_weighted_bias():
    options = ["a", "b"]
    probs = [0.9, 0.1]
    mh = VarRangeWithProbabilities(options, probs)
    r = NativeRandomSource(seed=123)

    trials = 5000
    count_a = 0
    count_b = 0
    for _ in range(trials):
        v = mh.generate(r, None, str, lambda t: None, {}, [])
        if v == "a":
            count_a += 1
        elif v == "b":
            count_b += 1

    assert count_a > count_b
    expected_a = probs[0] / sum(probs)
    observed_a = count_a / trials
    # Loose tolerance due to RNG and integer-weight discretization
    assert abs(observed_a - expected_a) < 0.1


def test_generate_equal_weights_near_uniform():
    options = ["x", "y", "z"]
    probs = [1.0, 1.0, 1.0]
    mh = VarRangeWithProbabilities(options, probs)
    r = NativeRandomSource(seed=456)

    trials = 6000
    counts = {o: 0 for o in options}
    for _ in range(trials):
        v = mh.generate(r, None, str, lambda t: None, {}, [])
        counts[v] += 1

    expected = trials / len(options)
    tolerance = trials * 0.1
    for c in counts.values():
        assert abs(c - expected) < tolerance


def test_iterate_descending_order():
    options = ["low", "mid", "high"]
    probs = [0.1, 0.3, 0.6]
    mh = VarRangeWithProbabilities(options, probs)

    # iterate parameters are not used by this MH, pass dummies
    it = list(
        mh.iterate(
            str,
            cast_combine_lists(lambda xs: (x for x in xs)),
            None,
            {},
        ),
    )

    assert it == ["high", "mid", "low"]


def test_iterate_stable_for_equal_weights():
    options = ["a", "b", "c", "d"]
    probs = [1.0, 1.0, 1.0, 1.0]
    mh = VarRangeWithProbabilities(options, probs)

    it = list(
        mh.iterate(
            str,
            cast_combine_lists(lambda xs: (x for x in xs)),
            None,
            {},
        ),
    )

    # Python sort is stable; equal probabilities keep original order
    assert it == options


def cast_combine_lists(f: Callable[[list[type]], Generator[Any, Any, Any]]):
    # Helper to satisfy type signature without importing heavy modules
    return f
