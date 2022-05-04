from __future__ import annotations

from optparse import OptionParser


def parse_args(args):
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option(
        "-p",
        "--population_size",
        dest="population_size",
        type="int",
        default=200,
    )
    parser.add_option("-v", "--verbose", dest="verbose", type="int", default=0)
    parser.add_option("-s", "--seed", dest="seed", type="int", default=0)
    parser.add_option("-e", "--elites", dest="n_elites", type="int", default=5)
    parser.add_option("-n", "--novelties", dest="n_novelties", type="int", default=10)
    parser.add_option(
        "-g",
        "--generations",
        dest="number_of_generations",
        type="int",
        default=100,
    )
    parser.add_option("-d", "--max_depth", dest="max_depth", type="int", default=7)
    parser.add_option(
        "-m",
        "--mutation",
        dest="probability_mutation",
        type="float",
        default=0.01,
    )
    parser.add_option(
        "-c",
        "--crossover",
        dest="probability_crossover",
        type="float",
        default=0.9,
    )
    parser.add_option("--csv", dest="save_to_csv", type="string", default=None)
    parser.add_option("--sequential", dest="parallel_evaluation", action="store_false")

    return parser.parse_args()
