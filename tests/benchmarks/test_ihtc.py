from examples.benchmarks.ihtc import IHTCBenchmark
from examples.benchmarks.ihtc import NurseAssignment
from examples.benchmarks.ihtc import PatientAssignment
from examples.benchmarks.ihtc import Timetable
from examples.benchmarks.ihtc import timetable_fitness
from examples.benchmarks.ihtc import load_ihtp_instance
from examples.benchmarks.ihtc import load_ihtp_solution
from examples.benchmarks.ihtc import validate_ihtp_solution


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


def test_official_ihtp_json_format_is_loaded_and_validated(tmp_path):
    instance_path = tmp_path / "instance.json"
    solution_path = tmp_path / "solution.json"
    instance_path.write_text(
        '{"days": 2, "shift_types": ["early"], "patients": [], "occupants": [], '
        '"nurses": [{"id": "n0"}], "surgeons": [], "operating_theaters": [], '
        '"rooms": [{"id": "r0"}]}',
        encoding="utf-8",
    )
    solution_path.write_text('{"patients": [], "nurses": []}', encoding="utf-8")
    instance = load_ihtp_instance(str(instance_path))
    solution = load_ihtp_solution(str(solution_path))
    assert validate_ihtp_solution(instance, solution) == []
