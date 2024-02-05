from __future__ import annotations

import pytest

from geneticengine.grammar.grammar import extract_grammar


class A:
    pass


class B(A):
    pass


def test_non_abs_error():
    with pytest.raises(Exception):
        extract_grammar([A, B], A)
