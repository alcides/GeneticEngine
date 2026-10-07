import pytest

from geneticengine.algorithms.gp.operators.umad import sarms_mutate
from geneticengine.algorithms.gp.operators.umad import umad_mutate
from geneticengine.random.sources import NativeRandomSource


def test_umad_keeps_expected_size_without_deletions():
    random = NativeRandomSource(1)
    result = umad_mutate([1, 2, 3], random, lambda: 0, rate=0)
    assert result == [1, 2, 3]


def test_umad_supports_rates_above_one():
    random = NativeRandomSource(2)
    result = umad_mutate([1, 2], random, lambda: 0, rate=2)
    assert all(value in (0, 1, 2) for value in result)


def test_sarms_returns_a_bounded_adapted_rate():
    random = NativeRandomSource(3)
    _, rate = sarms_mutate([1, 2, 3], random, lambda: 0, rate=0.1, meta_rate=0.2)
    assert 1e-6 <= rate <= 10


def test_umad_rejects_invalid_rates():
    with pytest.raises(ValueError):
        umad_mutate([1], NativeRandomSource(1), lambda: 0, rate=-1)
