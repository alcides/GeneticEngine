import pytest

from geneticengine.core.grammar import extract_grammar


class A(object):
    pass


class B(A):
    pass


def test_non_abs_error():
    with pytest.raises(Exception):
        extract_grammar([A, B], A)
