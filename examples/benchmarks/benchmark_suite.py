from examples.benchmarks.classification import ClassificationBenchmark
from examples.benchmarks.classification_lexicase import ClassificationLexicaseBenchmark
from examples.benchmarks.datasets import get_banknote, get_game_of_life, get_vladislavleva
from examples.benchmarks.domino import DominoBenchmark, blacks, top_target, side_target
from examples.benchmarks.game_of_life_vectorial import GameOfLifeVectorialBenchmark
from examples.benchmarks.regression import RegressionBenchmark
from examples.benchmarks.regression_lexicase import RegressionLexicaseBenchmark
from examples.benchmarks.string_match import StringMatchBenchmark
from examples.benchmarks.santafe import SantaFeBenchmark, map
from examples.benchmarks.vectorialgp import VectorialGPBenchmark, dataset
from examples.benchmarks.pymax import PyMaxBenchmark
from geml.common import PopulationRecorder
from geneticengine.algorithms.random_search import RandomSearch
from geneticengine.evaluation.budget import TimeBudget
from geneticengine.evaluation.sequential import SequentialEvaluator
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from examples.benchmarks.mario_level import MarioBenchmark
from examples.benchmarks.lambda_calculus import LambdaCalculusBenchmark
from examples.benchmarks.median import MedianBenchmark


banknote = get_banknote()
vladislavleva = get_vladislavleva()
gol = get_game_of_life()

benchmarks = [
    ClassificationBenchmark(*banknote),
    ClassificationLexicaseBenchmark(*banknote),
    GameOfLifeVectorialBenchmark(*gol),
    PyMaxBenchmark(),
    RegressionLexicaseBenchmark(*vladislavleva),
    RegressionBenchmark(*vladislavleva),
    SantaFeBenchmark(map),
    StringMatchBenchmark(),
    VectorialGPBenchmark(dataset),
    DominoBenchmark(blacks, top_target, side_target),
    MarioBenchmark(),
    LambdaCalculusBenchmark(),
    MedianBenchmark(),
]


methods = [
    lambda grammar, problem, random, budget, tracker: RandomSearch(
        problem=problem,
        budget=budget,
        representation=TreeBasedRepresentation(grammar, decider=ProgressivelyTerminalDecider(random, grammar)),
        random=random,
        tracker=tracker,
    ),
]

if __name__ == "__main__":
    budget = TimeBudget(10)
    for b in benchmarks:
        problem = b.get_problem()
        grammar = b.get_grammar()

        for m in methods:
            random = NativeRandomSource(1337)
            tracker = ProgressTracker(
                problem,
                evaluator=SequentialEvaluator(),
                recorders=[PopulationRecorder()],
            )
            instance = m(grammar, problem, random, budget, tracker)
            bests = instance.search()
            best = bests[0]
            print(
                f"Fitness of {best.get_fitness(problem)} for {best.get_phenotype()}",
            )
