from geneticengine.evaluation.parallel import ParallelEvaluator
from geneticengine.problems import InvalidFitnessException, SingleObjectiveProblem
from geneticengine.solutions.individual import ConcreteIndividual


class TestParallel:
    def test_parallel_evaluation_consumes_a_stream(self):
        evaluator = ParallelEvaluator(workers=2)

        def fitness(value: int):
            if value == 1:
                raise InvalidFitnessException()
            return value

        problem = SingleObjectiveProblem(fitness_function=fitness, minimize=True)
        individuals = (ConcreteIndividual(value) for value in [1, 2, 3, 4])

        evaluated = list(evaluator.evaluate(problem, individuals))

        assert [individual.get_phenotype() for individual in evaluated] == [2, 3, 4]
        assert evaluator.number_of_evaluations() == 4
