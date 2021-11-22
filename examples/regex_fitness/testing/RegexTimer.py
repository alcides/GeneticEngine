# Class extracted from PonyGE
import timeit
import traceback

# http://stackoverflow.com/questions/24812253/
# how-can-i-capture-return-value-with-python-timeit-module/
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""


def time_regex_test_case(compiled_regex, test_case, iterations):
    """
    Execute and time a single regex on a single test case
    
    :param compiled_regex:
    :param test_case:
    :param iterations:
    :return:
    """

    try:
        repeats = 10
        search_string = test_case.search_string

        def wrap():
            # Timing bug, lazy eval defers computation if we don't
            # force (convert to list evals result here)
            # https://swizec.com/blog/python-and-lazy-evaluation/swizec/5148
            return list(compiled_regex.finditer(search_string))

        t = timeit.Timer(wrap)
        repeat_iterations = t.repeat(repeat=repeats, number=iterations)

        best_run = list(repeat_iterations[0])

        for repeated_timeit in repeat_iterations:
            if best_run[0] > list(repeated_timeit)[0]:
                best_run = list(repeated_timeit)

        return_vals = list(best_run)
        return_vals.append(iterations)
        return_vals.append(test_case)

    except:
        traceback.print_exc()

    return return_vals
