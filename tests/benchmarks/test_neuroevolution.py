from examples.benchmarks.neuroevolution import AndBenchmark
from examples.benchmarks.neuroevolution import NeuralNetwork
from examples.benchmarks.neuroevolution import XorBenchmark
from examples.benchmarks.neuroevolution import xor_fitness


def test_neuroevolution_grammars_are_constructible():
    assert XorBenchmark().get_grammar() is not None
    assert AndBenchmark().get_grammar() is not None


def test_xor_network_has_low_error_for_a_known_solution():
    network = NeuralNetwork([5, 5, -2.5, 5, 5, -7.5, 5, -10, -2.5])
    assert xor_fitness(network) < 1.1


def test_network_predictions_are_bounded():
    network = NeuralNetwork([0.0] * 9)
    assert -1 <= network.predict(0.0, 1.0) <= 1
