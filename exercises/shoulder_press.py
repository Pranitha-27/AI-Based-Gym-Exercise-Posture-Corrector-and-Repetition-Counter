"""
Overhead Shoulder Press Posture Monitor.
Evaluates overhead lockout, bottom depth, and prevents lower back arching / hyperextension.
"""

from __future__ import annotations
from typing import Dict, List
from core.geometry import calculate_angle_2d, calculate_inclination
from core.pose_detector import PoseLandmarks
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class ShoulderPressExercise(BaseExercise):
    """
    Monitors Overhead Shoulder Press:
    - Lockout: Elbow angle reaches >= 165 degrees overhead.
    - Bottom Rack: Elbow drops to <= 90 degrees at shoulders.
    - Torso Arching: Torso inclination must not hyperextend backward (> 15 deg).
    """

    LOCKOUT_ANGLE = 165.0       # Overhead arms extended
    BOTTOM_RACK_ANGLE = 90.0    # At shoulder/chin level
    MAX_BACK_ARCH = 20.0        # Excessive backward lean

    def __init__(self):
        super().__init__("Shoulder Press")
        self.max_elbow_angle = 0.0
        self.lowest_elbow_angle = 180.0
        self.max_back_arch_in_rep = 0.0

    def reset(self):
        super().reset()
        self.max_elbow_angle = 0.0
        self.lowest_elbow_angle = 180.0
        self.max_back_arch_in_rep = 0.0

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"

        left_vis = (landmarks.left_shoulder.visibility + landmarks.left_elbow.visibility + landmarks.left_wrist.visibility) / 3.0
        right_vis = (landmarks.right_shoulder.visibility + landmarks.right_elbow.visibility + landmarks.right_wrist.visibility) / 3.0

        left_elbow = calculate_angle_2d(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist)
        right_elbow = calculate_angle_2d(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)

        # Inclination of spine
        if left_vis >= right_vis:
            side = "LEFT"
            elbow_angle = left_elbow
            spine_lean = calculate_inclination(landmarks.left_hip, landmarks.left_shoulder, reference="vertical")
        else:
            side = "RIGHT"
            elbow_angle = right_elbow
            spine_lean = calculate_inclination(landmarks.right_hip, landmarks.right_shoulder, reference="vertical")

        smooth_elbow = self.filter.filter("press_elbow", elbow_angle)
        smooth_spine = self.filter.filter("press_spine", spine_lean)

        # Rep progress: 90 deg -> 0%, 165 deg -> 100%
        self.rep_progress = max(0.0, min(1.0, (smooth_elbow - self.BOTTOM_RACK_ANGLE) / (self.LOCKOUT_ANGLE - self.BOTTOM_RACK_ANGLE)))

        # State machine
        if self.stage in ("IDLE", "RACK"):
            if smooth_elbow > self.BOTTOM_RACK_ANGLE + 15.0:
                self.stage = "PRESSING"
                self.max_elbow_angle = smooth_elbow
                self.max_back_arch_in_rep = smooth_spine
        elif self.stage == "PRESSING":
            if smooth_elbow > self.max_elbow_angle:
                self.max_elbow_angle = smooth_elbow
            self.max_back_arch_in_rep = max(self.max_back_arch_in_rep, smooth_spine)

            if smooth_elbow >= self.LOCKOUT_ANGLE:
                self.stage = "TOP_LOCKOUT"
            elif smooth_elbow < self.max_elbow_angle - 15.0:
                self.stage = "LOWERING"
        elif self.stage == "TOP_LOCKOUT":
            if smooth_elbow > self.max_elbow_angle:
                self.max_elbow_angle = smooth_elbow
            if smooth_elbow < self.LOCKOUT_ANGLE - 15.0:
                self.stage = "LOWERING"
        elif self.stage == "LOWERING":
            if smooth_elbow <= self.BOTTOM_RACK_ANGLE + 10.0:
                # Rep completed! Validate form
                if self.max_elbow_angle < 155.0:
                    self.current_rep_faults.append("Press all the way overhead (incomplete lockout)")
                if self.max_back_arch_in_rep > self.MAX_BACK_ARCH:
                    self.current_rep_faults.append("Excessive back arching! Brace your core")

                self.record_rep()
                self.stage = "RACK"
                self.max_elbow_angle = 0.0
                self.max_back_arch_in_rep = 0.0

        # Feedback cues
        if smooth_spine > self.MAX_BACK_ARCH:
            self.active_feedback.append("Stop arching back! Squeeze glutes & tighten core")
            self.feedback_severity = "ALERT"
        elif self.stage == "TOP_LOCKOUT":
            self.active_feedback.append("Full lockout! Lower under control")
            self.feedback_severity = "GOOD"
        elif self.stage == "PRESSING":
            self.active_feedback.append("Drive straight overhead")
        elif self.stage in ("RACK", "IDLE"):
            self.active_feedback.append("Weights at collarbone, ready to press")

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=self.total_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            stage=self.stage,
            current_angles={
                "elbow_angle": round(smooth_elbow, 1),
                "spine_lean": round(smooth_spine, 1)
            },
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=self.rep_progress,
            accuracy_percentage=self.accuracy,
            side_view=side
        )
