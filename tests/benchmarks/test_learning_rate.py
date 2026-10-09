from __future__ import annotations

import numpy as np

from examples.benchmarks.learning_rate import EPOCHS, TARGET_SCHEDULE, LearningRateBenchmark, LearningRateSchedule


def test_learning_rate_benchmark_scores_a_schedule():
    schedule = LearningRateSchedule(list(TARGET_SCHEDULE))
    assert LearningRateBenchmark().fitness(schedule) == 0.0


def test_learning_rate_schedule_has_one_rate_per_epoch():
    schedule = LearningRateSchedule([0.01] * EPOCHS)
    assert len(schedule.values()) == EPOCHS
    assert np.all(schedule.values() > 0)
