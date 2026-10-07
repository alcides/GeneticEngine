from examples.benchmarks.ihtc import IHTCBenchmark
from examples.benchmarks.ihtc import NurseAssignment
from examples.benchmarks.ihtc import PatientAssignment
from examples.benchmarks.ihtc import Timetable
from examples.benchmarks.ihtc import timetable_fitness


def valid_timetable():
    return Timetable(
        patients=[
            PatientAssignment(0, 1, 0, 1),
            PatientAssignment(1, 0, 0, 0),
            PatientAssignment(2, 2, 0, 2),
            PatientAssignment(3, 0, 1, 0),
        ],
        nurses=[
            NurseAssignment(0, 0, 1, 1),
            NurseAssignment(1, 1, 0, 1),
            NurseAssignment(2, 0, 2, 0),
            NurseAssignment(3, 1, 0, 1),
        ],
    )


def test_ihtc_grammar_is_constructible():
    assert IHTCBenchmark().get_grammar() is not None


def test_valid_timetable_has_zero_penalty():
    assert timetable_fitness(valid_timetable()) == 0


def test_invalid_timetable_has_positive_penalty():
    timetable = valid_timetable()
    timetable.patients[0].admission_day = 0
    assert timetable_fitness(timetable) > 0
