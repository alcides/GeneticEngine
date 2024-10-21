from examples.benchmarks.classification import ClassificationBenchmark
from examples.benchmarks.classification_lexicase import ClassificationLexicaseBenchmark
from examples.benchmarks.datasets import get_banknote, get_game_of_life, get_vladislavleva
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
from geneticengine.evaluation.tracker import SingleObjectiveProgressTracker
from geneticengine.random.sources import NativeRandomSource
from geneticengine.representations.tree.initializations import ProgressivelyTerminalDecider
from geneticengine.representations.tree.treebased import TreeBasedRepresentation


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
            tracker = SingleObjectiveProgressTracker(
                problem,
                evaluator=SequentialEvaluator(),
                recorders=[PopulationRecorder()],
            )
            instance = m(grammar, problem, random, budget, tracker)
            best = instance.search()
            print(
                f"Fitness of {best.get_fitness(problem)} by genotype: {best.genotype} with phenotype: {best.get_phenotype()}",
            )
