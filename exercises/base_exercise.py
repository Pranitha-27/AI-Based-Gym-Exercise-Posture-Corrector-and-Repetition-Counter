"""
Base Exercise class defining state machine lifecycle, rep counting,
and posture evaluation metrics.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from core.geometry import ExponentialMovingAverageFilter
from core.pose_detector import PoseLandmarks


@dataclass
class ExerciseMetrics:
    """Snapshot of real-time exercise telemetry and evaluation results."""
    exercise_name: str
    total_reps: int
    good_reps: int
    bad_reps: int
    stage: str
    current_angles: Dict[str, float] = field(default_factory=dict)
    active_feedback: List[str] = field(default_factory=list)
    feedback_severity: str = "GOOD"  # "GOOD", "WARNING", "ALERT"
    rep_progress: float = 0.0        # 0.0 to 1.0 (0% - 100%)
    accuracy_percentage: float = 100.0
    elapsed_seconds: float = 0.0
    side_view: str = "FRONT"         # "LEFT", "RIGHT", "FRONT"
    extra_data: Dict[str, Any] = field(default_factory=dict)


class BaseExercise(ABC):
    """
    Abstract base class for all exercise monitors.
    Manages state machine transitions, rep categorization, and smoothing.
    """

    def __init__(self, name: str):
        self.name = name
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "IDLE"
        self.active_feedback: List[str] = []
        self.feedback_severity = "GOOD"
        self.rep_progress = 0.0
        self.start_time = time.time()
        self.filter = ExponentialMovingAverageFilter(alpha=0.3)
        self.current_rep_faults: List[str] = []
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        """Reset exercise statistics and state machine."""
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.stage = "IDLE"
        self.active_feedback = []
        self.feedback_severity = "GOOD"
        self.rep_progress = 0.0
        self.start_time = time.time()
        self.filter.reset()
        self.current_rep_faults = []
        self.history = []

    @property
    def accuracy(self) -> float:
        """Percentage of good reps relative to total reps."""
        if self.total_reps == 0:
            return 100.0
        return round((self.good_reps / self.total_reps) * 100.0, 1)

    def record_rep(self):
        """Record completed repetition and check for any posture infractions."""
        self.total_reps += 1
        is_good = len(self.current_rep_faults) == 0
        if is_good:
            self.good_reps += 1
        else:
            self.bad_reps += 1

        self.history.append({
            "rep_number": self.total_reps,
            "is_good": is_good,
            "faults": list(set(self.current_rep_faults)),
            "timestamp": time.time() - self.start_time
        })
        self.current_rep_faults.clear()

    @abstractmethod
    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        """
        Evaluate body posture landmarks against biomechanical rules.
        Must update self.stage, rep counts, and return ExerciseMetrics.
        """
        pass
