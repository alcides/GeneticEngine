from geml.grammars.symbolic_regression import Plus, Zero, Two, optimize_expression


def test_saturation():
    e = Plus(Zero(), Two())
    e2 = optimize_expression(e)
    assert e2 == Two()
