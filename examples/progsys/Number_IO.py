from __future__ import annotations

import csv
import time
from optparse import OptionParser
from typing import Annotated

import examples.global_vars as gv
from examples.progsys.utils import get_data
from examples.progsys.utils import import_embedded
from geneticengine.algorithms.gp.gp import GP
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.grammatical_evolution.dynamic_structured_ge import (
    dsge_representation,
)
from geneticengine.core.representations.grammatical_evolution.ge import (
    ge_representation,
)
from geneticengine.core.representations.grammatical_evolution.structured_ge import (
    sge_representation,
)
from geneticengine.core.representations.tree.treebased import treebased_representation
from geneticengine.grammars.coding.numbers import Literal
from geneticengine.grammars.coding.numbers import Mul
from geneticengine.grammars.coding.numbers import Number
from geneticengine.grammars.coding.numbers import Plus
from geneticengine.grammars.coding.numbers import SafeDiv
from geneticengine.grammars.coding.numbers import Var
from geneticengine.metahandlers.vars import VarRange


FILE_NAME = "Number_IO"
DATA_FILE_TRAIN = f"GeneticEngine/examples/progsys/data/{FILE_NAME}/Train.txt"
DATA_FILE_TEST = f"GeneticEngine/examples/progsys/data/{FILE_NAME}/Test.txt"

inval, outval = get_data(DATA_FILE_TRAIN, DATA_FILE_TEST)
imported = import_embedded(FILE_NAME)

vars = ["in0", "in1"]
variables = {}
for i, n in enumerate(vars):
    variables[n] = i

Var.__init__.__annotations__["name"] = Annotated[str, VarRange(vars)]
Var.feature_indices = variables  # type: ignore


def fitness_function(n: Number):
    fitness, error, cases = imported.fitness(inval, outval, n.evaluate_lines())
    return fitness


def preprocess():
    return extract_grammar([Plus, Mul, SafeDiv, Literal, Var], Number)


def evolve(
    seed,
    mode,
    save_to_csv: str = None,
    representation="treebased_representation",
):
    if representation == "ge":
        representation = ge_representation
    elif representation == "sge":
        representation = sge_representation
    elif representation == "dsge":
        representation = dsge_representation
    else:
        representation = treebased_representation

    g = preprocess()
    alg = GP(
        g,
        representation=representation,
        probability_crossover=gv.PROBABILITY_CROSSOVER,
        probability_mutation=gv.PROBABILITY_MUTATION,
        number_of_generations=gv.NUMBER_OF_GENERATIONS,
        max_depth=gv.MAX_DEPTH,
        population_size=gv.POPULATION_SIZE,
        selection_method=gv.SELECTION_METHOD,
        n_elites=gv.N_ELITES,
        # ----------------
        problem=SingleObjectiveProblem(
            minimize=True,
            fitness_function=fitness_function,
            target_fitness=None,
        ),
        seed=seed,
        timer_stop_criteria=mode,
        save_to_csv=save_to_csv,
        save_genotype_as_string=False,
    )
    (b, bf, bp) = alg.evolve(verbose=1)
    return b, bf


if __name__ == "__main__":
    representations = ["treebased_representation", "ge", "dsge"]

    parser = OptionParser()
    parser.add_option("-s", "--seed", dest="seed", type="int")
    parser.add_option("-r", "--representation", dest="representation", type="int")
    parser.add_option(
        "-t",
        "--timed",
        dest="timed",
        action="store_const",
        const=True,
        default=False,
    )
    (options, args) = parser.parse_args()

    timed = options.timed
    seed = options.seed
    example_name = __file__.split(".")[0].split("\\")[-1].split("/")[-1]
    representation = representations[options.representation]
    print(seed, example_name, representation)
    evol_method = evolve

    mode = "generations"
    if timed:
        mode = "time"
    dest_file = f"{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/{seed}.csv"

    start = time.time()
    b, bf = evolve(seed, timed, dest_file, representation)
    end = time.time()
    csv_row = [mode, example_name, representation, seed, bf, (end - start), b]
    with open(
        f"./{gv.RESULTS_FOLDER}/{mode}/{example_name}/{representation}/main.csv",
        "a",
        newline="",
    ) as outfile:
        writer = csv.writer(outfile)
        writer.writerow(csv_row)
