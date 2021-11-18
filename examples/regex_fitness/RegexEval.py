#Class extracted from PonyGE
import re
from multiprocessing import Process, Queue

import examples.regex_fitness.testing.RegexTestGenerator as TestGen
from examples.regex_fitness.testing.RegexTimer import time_regex_test_case
from stats.stats import stats

# Author: Brendan Cody-Kenny - codykenny at gmail


class RegexEval(object):
    """
    Fitness function for regex (lower fitness value is better)
    Fitness = functionality error + time
    The regex is presented with a number of strings, resulting matches are
    checked for correctness against known correct answers.
    Sum of wall-clock time taken to execute the test strings.
    """

    # these need to be class variables, not object variables
    test_cases = []
    seed_regex = None
    time = True
    q = Queue()
    pstartup, prunner = None, None

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def call_fitness(self, individual):
        """
        This method is called when this class is instantiated with an
        individual (a regex)
        
        :param individual:
        :param q:
        :return:
        """
        regex_string = str(individual)

        compiled_regex = re.compile(regex_string)
        eval_results = self.test_regex(compiled_regex)
        result_error, time_sum = self.calculate_fitness(eval_results)
        fitness = result_error + time_sum

        # We are running this code in a thread so put the fitness on the
        # queue so it can be read back by the first length of the phenotype
        # puts parsimony pressure toward shorter regex
        return fitness

    def calculate_fitness(self, eval_results):
        """
        Sum the functionality error with time (and any other fitness penalties
        you want to add, e.g. length of regex)
        
        :param eval_results:
        :return:
        """

        result_error = 0
        time_sum = 0.0
        for a_result in eval_results:
            time_sum += a_result[0]
            result_error += a_result[3].calc_match_errors(list(a_result[1]))

        return result_error, time_sum

    def test_regex(self, compiled_regex):
        """
        Iterate through test cases
        
        :param compiled_regex:
        :return:
        """

        results = list()
        testing_iterations = 1

        for test_case in RegexEval.test_cases:
            results.append(
                time_regex_test_case(compiled_regex, test_case,
                                     testing_iterations))
        return results

    def evaluate(self, ind, **kwargs):
        """
        When this class is instantiated with individual, evaluate in a new
        process, timeout and kill process if it runs for 1 second.

        :param ind: An individual to be evaluated.
        :return: The fitness of the evaluated individual.
        """

        if RegexEval.seed_regex is None:
            # We can't initialise the seed regex when we initialise the
            # fitness function as the representation.individual.Individual
            # class has not yet been instantiated.

            RegexEval.seed_regex = 1  # TODO: this should be a random number

            RegexEval.test_cases = TestGen.generate_test_suite(
                RegexEval.seed_regex.phenotype)

            if len(RegexEval.test_cases) == 0:
                s = "fitness.regex.RegexEval.RegexEval\n" \
                    "Error: no regex test cases found! " \
                    "       Please add at least one passing regex test string."
                raise Exception(s)

        if RegexEval.pstartup is None:
            RegexEval.pstartup = Process(target=self.call_fitness,
                                         name="self.call_fitness")
        RegexEval.pstartup._args = (ind, RegexEval.q)

        RegexEval.pstartup.start()
        RegexEval.prunner = RegexEval.pstartup
        RegexEval.pstartup = Process(target=self.call_fitness,
                                     name="self.call_fitness")

        # Set one second time limit for running thread.
        self.prunner.join(1)

        # If thread is active
        if self.prunner.is_alive():
            # After one second, if prunner is still running, kill it.
            print("Regex evaluation timeout reached, "
                  "killing evaluation process")
            self.prunner.terminate()
            self.prunner.join()

            # Count individual as a runtime error.
            stats['runtime_error'] += 1

            return self.default_fitness

        else:
            return self.q.get()
