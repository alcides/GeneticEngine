# Class extracted from PonyGE
import re

from examples.regex_fitness.testing.RegexTest import RegexTest
from examples.regex_fitness.testing.RegexTimer import time_regex_test_case


def generate_equivalence_test_suite_replacement(a_match, compiled_regex):
    """
    This is a 'booster' for test suite generation. We know a single good
    match, and we can use that to find search strings which do not
    contain a match. Given a regex, generate/discover a test suite of
    examples which match, and those that don't. The test suite is used to
    define (or outline) the functionality boundaries of the regex. When we
    go to evolve new regexs, we can use the test suite to measure
    functionality equivalence with the original test regex.

    :param a_match:
    :param compiled_regex:
    :return:
    """
    test_cases = []
    # go through the whole known search string, changing letters until you
    # find one which does not match.
    # compiled_regex = re.compile(a_regex)
    if len(a_match.matches) > 0:
        for i in range(0, len(a_match.search_string)):
            for char in [a for a in range(ord('0'), ord('9'))] + \
                        [ord('a'), ord('Z')]:
                new_search_string = a_match.search_string[:i] + \
                                    chr(char) + \
                                    a_match.search_string[i + 1:]
                a_test_case_string = RegexTest(new_search_string)
                vals = time_regex_test_case(compiled_regex, a_test_case_string,
                                            1)
                if len(list(vals[1])) == 0:
                    test_cases.append(a_test_case_string)
    return test_cases


def generate_equivalence_test_suite_length(a_match, compiled_regex):
    """
    Generate shorter or longer test cases, add those which do not match

    :param a_match:
    :param compiled_regex:
    :return:
    """
    test_cases = []
    # add and remove characters from the string until we find a regex which
    # fails
    # compiled_regex = re.compile(a_regex)
    if len(a_match.matches) > 0:

        # check string with one character added at the front
        new_search_string = 'a' + a_match.search_string

        add_test_case_if_fails(new_search_string, compiled_regex, test_cases)
        # check string with one character added at the end
        new_search_string = a_match.search_string + 'a'
        add_test_case_if_fails(new_search_string, compiled_regex, test_cases)
        for i in range(len(a_match.search_string) - 1):
            # TODO: refactor this
            new_search_string = a_match.search_string[i:]
            add_test_case_if_fails(new_search_string, compiled_regex,
                                   test_cases)

        for i in range(len(a_match.search_string) - 1):
            # TODO: refactor this
            new_search_string = a_match.search_string[:i]
            add_test_case_if_fails(new_search_string, compiled_regex,
                                   test_cases)
    return test_cases


def add_test_case_if_fails(new_search_string, compiled_regex, test_cases):
    """
    run a test case, if it fails, add it to the suite of tests

    :param new_search_string:
    :param compiled_regex:
    :return:
    """
    a_test_case_string = RegexTest(new_search_string)
    vals = time_regex_test_case(compiled_regex, a_test_case_string, 1)
    if len(list(vals[1])) == 0:
        test_cases.append(a_test_case_string)


def generate_test_suite(regex_string):
    """
    
    :param regex_string:
    :return:
    """

    # do some test generation
    # find a string which the regex is able to match against
    # find the minimal variant of this string which does not match
    # test strategies - length, values

    # collect strings which identify the different regex
    # cache and reuse these (read/write to a file before/after GP)
    known_test_strings = [
        "5C0A5B634A82",
        "Jan 12 06:26:20: ACCEPT service dns from 140.105.48.16 to firewall(pub-nic-dns), prefix: \"none\" (in: eth0 140.105.48.16(00:21:dd:bc:95:44):4263 -> 140.105.63.158(00:14:31:83:c6:8d):53 UDP len:76",
        "Jan 12 06:27:09: DROP service 68->67(udp) from 216.34.211.83 to 216.34.253.94, prefix: \"spoof iana-0/8\" (in: eth0 213.92.153.78(00:1f:d6:19:0a:80):68 -> 69.43.177.110(00:30:fe:fd:d6:51):67 UDP le"
        "Jan 12 06:26:19: ACCEPT service http from 119.63.193.196 to firewall(pub-nic), prefix: ",
        "Jan 12 06:26:19: ACCEPT service http from 119.63.193.196 to firewall(pub-nic), prefix: ",
        "26:19: ACCEPT service http from 119.63.193.196 to firewall(pub-nic), prefix: ",
        " -> 140.105.63.164(50:j6:04:92:53:44):80 TCP flags: ****S* len:60 ttl:32)sdkfjhaklsjdhfglksjhdfgk",
        " -> 140.105.63.16(50:06:04:9r:53:44):80 TCP flags: ****S* len:60 ttl:32)ssjdhfglksjhdfgk",
        "Jan 12 06:26:20: ACCEPT service dns from 140.105.48.16 to firewall(pub-nic-dns), prefix: ",
        "Jan 12 06:27:09: DROP service 68->67(udp) from 216.34.211.83 to 216.34.253.94, prefix: ",
        "105.63.1650:06:04:92:53:44:80",
        " -> 140.105.63.164(50:06:g4:92:53:44):80 TCP flags: ****S* len:60 ttl:32)",
        " -> 140.105.63.164(50:06:54:92:r3:44):80 TCP flags: ****S* len:60 ttl:32)",
        "1,2,3,4,5,6,7,8,9,10,11,12,13777,5P,5,5,6,5P",
        "1,2,3,4,5,6,7,8,9,10,11,12,13777,24,5P",
        "1,2,3,4,5,6,7,8,9,10,11,12,13777,243,3P",
        "1,2,3,4,5,6,7,8,9,10,11,12,13777,P",
        "1,2,3,4,5,6,7,8,9,10,11,12,P",
        "1,2,3,4,5,6,7,8,9,10,11,P",
        "1,2,3,4,5,6,7,8,9,10,11,3P",
        "1,2,3,4,5,6,7,8,9,10,P",
        "codykenny@gmailcom",
        "2016-12-09T08:21:15.9+00:00",
        "2016-12-09T08:21:15.9+00:0",
        "2016-22-09T08:21:15.9+00:00000000000",
        "2016-22-09T08:21:15.9+00:00",
        "1911-02-19T22:35:42.3+08:43",
        "2016-09-05T15:22:26.286Z",
        "230.234E-10",
        "971.829E+26",
        "3566",
        "4",
        "-7",
        "+94",
        "            36",
        "78      ",
        "87465.345345",
        "2346.533",
        "0.045e-10",
        "3566.",
        ".3456",
        "<string> ::= <letter>|<letter><string>",
        "hryxioXcXXdornct",
        "bbbbXcyXXaaa",
        "230.234E-10",
        "971.829E+26",
        "3566",
        "4",
        "-7",
        "+94",
        "            36",
        "78      ",
        "87465.345345",
        "2346.533",
        "  3566.   ",
        " .3456  ",
        "a46b  ",
        "0.045e-10",
        "aXXXXas",
        "<s_char>        ::= !|\"#\"|$|%|&|\\(|\\)|*|+|,|-|.|\\/|:|;|\"<\"|=|\">\"|?|@|\\[|\\|\\]|^|_|\"`\"|{|}|~|\"|\"|'\"'|\"'\"|\" \""
        "!|\"#\"|$|%|&|\\(|\\)|*|+|,|-|.|\\/|:|;|\"<\"|=|\">\"|?|@|\\[|\\|\\]|^|_|\"`\"|{|}|~|\"|\"|'\"'|\"'\"|\" \"",
        "<A_Z>           ::= A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z",
        "A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z",
    ]

    compiled_regex = re.compile(regex_string)
    test_cases = []
    for test_string in known_test_strings:
        test_cases += generate_tests_if_string_match(compiled_regex,
                                                     test_string)

    # if we don't have any known test strings, see if the regex matches it.
    test_cases += generate_tests_if_string_match(compiled_regex, regex_string)

    print("Number of test cases in suite:", len(test_cases))

    return test_cases


def add_re_match_to_test(matches, passing_test_string):
    """
    take matching values as found by the regex library, and add them to our
    RegexTest object

    :param vals:
    :param passing_test_string:
    :return:
    """

    for a_match in matches:  # this vals[1] business is not good
        passing_test_string.matches.append(a_match)

    return passing_test_string


def generate_tests_if_string_match(compiled_regex, test_string):
    """
    
    :param compiled_regex:
    :param test_string:
    :return:
    """

    test_cases = []
    a_test_candidate = RegexTest(test_string)
    vals = time_regex_test_case(compiled_regex, a_test_candidate, 1)

    if len(list(vals[1])) > 0:  # the regex found a match, add it
        a_positive_test = add_re_match_to_test(vals[1], a_test_candidate)
        test_cases.append(a_positive_test)

        # now find regex which negate
        test_cases += generate_equivalence_test_suite_replacement(
            a_positive_test, compiled_regex)
        test_cases += generate_equivalence_test_suite_length(
            a_positive_test, compiled_regex)
    return test_cases
