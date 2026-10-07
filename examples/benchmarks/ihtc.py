"""A compact IHTC-2024-style integrated healthcare timetable.

The official competition combines patient admission, operating-theater
scheduling, room assignment, and nurse rostering.  This example keeps the
same decisions and demonstrates the main hard constraints on a small, fully
embedded instance so it can be run without downloading the competition data.
See https://ihtc2024.github.io for the complete problem and file format.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated

from examples.benchmarks.benchmark import Benchmark, example_run
from geneticengine.grammar.grammar import Grammar, extract_grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.lists import ListSizeBetween
from geneticengine.problems import Problem, SingleObjectiveProblem

DAYS = 5
ROOM_CAPACITY = (2, 2)
SHIFT_NAMES = ("early", "late")


def load_ihtp_instance(path: str) -> dict:
    """Load an official IHTC-2024 JSON instance.

    The competition publishes each instance as one JSON document.  Keeping
    the loader deliberately format-preserving makes it possible to use the
    official validator and avoids bundling the 30 large public instances.
    """
    with open(path, encoding="utf-8") as instance_file:
        instance = json.load(instance_file)
    required = {"days", "occupants", "patients", "nurses", "surgeons", "operating_theaters", "rooms"}
    missing = required - instance.keys()
    if missing:
        raise ValueError(f"IHTP instance is missing fields: {sorted(missing)}")
    return instance


def load_ihtp_solution(path: str) -> dict:
    """Load an official IHTC-2024 solution JSON document."""
    with open(path, encoding="utf-8") as solution_file:
        solution = json.load(solution_file)
    if not {"patients", "nurses"} <= solution.keys():
        raise ValueError("IHTP solution must contain patients and nurses")
    return solution


def validate_ihtp_solution(instance: dict, solution: dict) -> list[str]:
    """Return basic structural violations in an official-format solution."""
    errors = []
    patient_ids = {patient["id"] for patient in instance["patients"]}
    room_ids = {room["id"] for room in instance["rooms"]}
    theater_ids = {theater["id"] for theater in instance["operating_theaters"]}
    nurse_ids = {nurse["id"] for nurse in instance["nurses"]}
    if len({patient.get("id") for patient in solution["patients"]}) != len(solution["patients"]):
        errors.append("duplicate patient assignment")
    for patient in solution["patients"]:
        if patient.get("id") not in patient_ids:
            errors.append(f"unknown patient: {patient.get('id')}")
        if patient.get("admission_day") == "none":
            continue
        day = patient.get("admission_day")
        if not isinstance(day, int) or not 0 <= day < instance["days"]:
            errors.append(f"invalid admission day for {patient.get('id')}")
        if patient.get("room") not in room_ids:
            errors.append(f"invalid room for {patient.get('id')}")
        if patient.get("operating_theater") not in theater_ids:
            errors.append(f"invalid operating theater for {patient.get('id')}")
    for nurse in solution["nurses"]:
        if nurse.get("id") not in nurse_ids:
            errors.append(f"unknown nurse: {nurse.get('id')}")
        for assignment in nurse.get("assignments", []):
            if not isinstance(assignment.get("day"), int) or not 0 <= assignment["day"] < instance["days"]:
                errors.append(f"invalid nurse assignment day for {nurse.get('id')}")
            if assignment.get("shift") not in instance["shift_types"]:
                errors.append(f"invalid nurse shift for {nurse.get('id')}")
            errors.extend(
                f"invalid room {room} for nurse {nurse.get('id')}"
                for room in assignment.get("rooms", [])
                if room not in room_ids
            )
    return errors


@dataclass(frozen=True)
class Patient:
    mandatory: bool
    release_day: int
    due_day: int | None
    length_of_stay: int
    surgery_duration: int
    incompatible_rooms: frozenset[int]
    workload: tuple[int, ...]
    required_skill: int


PATIENTS = (
    Patient(True, 1, 2, 2, 2, frozenset({1}), (2, 1), 0),
    Patient(False, 0, None, 2, 1, frozenset(), (1, 2), 1),
    Patient(True, 2, 4, 2, 2, frozenset(), (2, 2), 0),
    Patient(False, 0, None, 1, 1, frozenset(), (1, 1), 0),
)


@dataclass(frozen=True)
class Nurse:
    skill_level: int
    shifts: frozenset[tuple[int, int]]
    max_load: int


NURSES = (
    Nurse(0, frozenset({(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)}), 10),
    Nurse(1, frozenset({(0, 1), (1, 0), (2, 1), (3, 0), (4, 1)}), 10),
)


class Assignment(ABC):
    """A decision made by the timetable."""

    @abstractmethod
    def __str__(self):
        pass


@dataclass
class PatientAssignment(Assignment):
    patient: Annotated[int, IntRange(0, len(PATIENTS) - 1)]
    admission_day: Annotated[int, IntRange(0, DAYS - 1)]
    room: Annotated[int, IntRange(0, len(ROOM_CAPACITY) - 1)]
    surgery_day: Annotated[int, IntRange(0, DAYS - 1)]

    def __str__(self):
        return f"p{self.patient}: day={self.admission_day}, room=r{self.room}, surgery={self.surgery_day}"


@dataclass
class NurseAssignment(Assignment):
    patient: Annotated[int, IntRange(0, len(PATIENTS) - 1)]
    nurse: Annotated[int, IntRange(0, len(NURSES) - 1)]
    day: Annotated[int, IntRange(0, DAYS - 1)]
    shift: Annotated[int, IntRange(0, len(SHIFT_NAMES) - 1)]

    def __str__(self):
        return f"p{self.patient}: n{self.nurse}, day={self.day}, shift={SHIFT_NAMES[self.shift]}"


@dataclass
class Timetable:
    patients: Annotated[list[PatientAssignment], ListSizeBetween(len(PATIENTS), len(PATIENTS))]
    nurses: Annotated[list[NurseAssignment], ListSizeBetween(len(PATIENTS), len(PATIENTS))]


def timetable_fitness(timetable: Timetable) -> float:
    """Return a weighted sum of timetable constraint violations."""
    penalty = 0
    patient_assignments = {assignment.patient: assignment for assignment in timetable.patients}

    penalty += 10 * (len(PATIENTS) - len(patient_assignments))
    for patient_id, patient in enumerate(PATIENTS):
        assignment = patient_assignments.get(patient_id)
        if assignment is None:
            continue
        if assignment.admission_day < patient.release_day:
            penalty += 10 * (patient.release_day - assignment.admission_day)
        if patient.mandatory and patient.due_day is not None and assignment.admission_day > patient.due_day:
            penalty += 10 * (assignment.admission_day - patient.due_day)
        if assignment.surgery_day != assignment.admission_day:
            penalty += 10
        if assignment.room in patient.incompatible_rooms:
            penalty += 10

    for day in range(DAYS):
        occupied = [0] * len(ROOM_CAPACITY)
        surgery_load = 0
        for patient_id, assignment in patient_assignments.items():
            patient = PATIENTS[patient_id]
            if assignment.admission_day <= day < assignment.admission_day + patient.length_of_stay:
                occupied[assignment.room] += 1
            if assignment.surgery_day == day:
                surgery_load += patient.surgery_duration
        penalty += sum(max(0, count - capacity) for count, capacity in zip(occupied, ROOM_CAPACITY))
        penalty += max(0, surgery_load - 3)

    nurse_assignments = {assignment.patient: assignment for assignment in timetable.nurses}
    penalty += 10 * (len(PATIENTS) - len(nurse_assignments))
    nurse_load = [0] * len(NURSES)
    for patient_id, patient in enumerate(PATIENTS):
        assignment = nurse_assignments.get(patient_id)
        patient_assignment = patient_assignments.get(patient_id)
        if assignment is None or patient_assignment is None:
            continue
        nurse = NURSES[assignment.nurse]
        if (assignment.day, assignment.shift) not in nurse.shifts:
            penalty += 5
        if not patient_assignment.admission_day <= assignment.day < patient_assignment.admission_day + patient.length_of_stay:
            penalty += 5
        if nurse.skill_level < patient.required_skill:
            penalty += 5
        nurse_load[assignment.nurse] += sum(patient.workload)

    penalty += sum(max(0, load - nurse.max_load) for load, nurse in zip(nurse_load, NURSES))
    return penalty


class IHTCBenchmark(Benchmark):
    """Benchmark wrapper for the embedded IHTC-style toy instance."""

    def __init__(self):
        self.problem = SingleObjectiveProblem(
            fitness_function=timetable_fitness,
            minimize=True,
            target=0,
        )
        self.grammar = extract_grammar(
            [Assignment, PatientAssignment, NurseAssignment, Timetable],
            Timetable,
        )

    def get_problem(self) -> Problem:
        return self.problem

    def get_grammar(self) -> Grammar:
        return self.grammar


if __name__ == "__main__":
    example_run(IHTCBenchmark())
