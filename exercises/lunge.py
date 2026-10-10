"""
Lunge Posture Monitor.
Evaluates front and rear knee angles, depth, and upright torso posture.
"""

from __future__ import annotations
from typing import Dict, List
from core.geometry import calculate_angle_2d, calculate_inclination
from core.pose_detector import PoseLandmarks
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class LungeExercise(BaseExercise):
    """
    Monitors Lunges:
    - Front Knee Depth: Front knee angle should reach ~90 degrees.
    - Torso Upright: Spine should remain vertical (<= 20 deg lean).
    - Prevents half-depth lunges and excessive forward leaning.
    """

    STAND_ANGLE = 160.0    # Standing straight
    LUNGE_DEPTH = 95.0     # Deep lunge position
    MAX_TORSO_LEAN = 25.0  # Max forward lean during lunge

    def __init__(self):
        super().__init__("Lunge")
        self.lowest_knee_angle = 180.0
        self.max_torso_lean_in_rep = 0.0

    def reset(self):
        super().reset()
        self.lowest_knee_angle = 180.0
        self.max_torso_lean_in_rep = 0.0

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"

        left_vis = (landmarks.left_hip.visibility + landmarks.left_knee.visibility + landmarks.left_ankle.visibility) / 3.0
        right_vis = (landmarks.right_hip.visibility + landmarks.right_knee.visibility + landmarks.right_ankle.visibility) / 3.0

        left_knee = calculate_angle_2d(landmarks.left_hip, landmarks.left_knee, landmarks.left_ankle)
        right_knee = calculate_angle_2d(landmarks.right_hip, landmarks.right_knee, landmarks.right_ankle)

        # Front leg is usually the one bent more during descent
        if left_knee <= right_knee:
            primary_knee = left_knee
            side = "LEFT_LEAD"
            torso_lean = calculate_inclination(landmarks.left_hip, landmarks.left_shoulder, reference="vertical")
        else:
            primary_knee = right_knee
            side = "RIGHT_LEAD"
            torso_lean = calculate_inclination(landmarks.right_hip, landmarks.right_shoulder, reference="vertical")

        smooth_knee = self.filter.filter("lunge_knee", primary_knee)
        smooth_torso = self.filter.filter("lunge_torso", torso_lean)

        # Rep progress: 160 deg -> 0%, 95 deg -> 100%
        self.rep_progress = max(0.0, min(1.0, (self.STAND_ANGLE - smooth_knee) / (self.STAND_ANGLE - self.LUNGE_DEPTH)))

        # State machine
        if self.stage in ("IDLE", "STAND"):
            if smooth_knee < self.STAND_ANGLE - 15.0:
                self.stage = "STEPPING_DOWN"
                self.lowest_knee_angle = smooth_knee
                self.max_torso_lean_in_rep = smooth_torso
        elif self.stage == "STEPPING_DOWN":
            if smooth_knee < self.lowest_knee_angle:
                self.lowest_knee_angle = smooth_knee
            self.max_torso_lean_in_rep = max(self.max_torso_lean_in_rep, smooth_torso)

            if smooth_knee <= self.LUNGE_DEPTH:
                self.stage = "BOTTOM"
            elif smooth_knee > self.lowest_knee_angle + 15.0:
                self.stage = "RETURNING"
        elif self.stage == "BOTTOM":
            if smooth_knee < self.lowest_knee_angle:
                self.lowest_knee_angle = smooth_knee
            self.max_torso_lean_in_rep = max(self.max_torso_lean_in_rep, smooth_torso)

            if smooth_knee > self.LUNGE_DEPTH + 15.0:
                self.stage = "RETURNING"
        elif self.stage == "RETURNING":
            self.max_torso_lean_in_rep = max(self.max_torso_lean_in_rep, smooth_torso)

            if smooth_knee >= self.STAND_ANGLE - 10.0:
                # Rep completed!
                if self.lowest_knee_angle > 110.0:
                    self.current_rep_faults.append("Lunge lower (reach 90 deg knee angle)")
                if self.max_torso_lean_in_rep > self.MAX_TORSO_LEAN:
                    self.current_rep_faults.append("Keep torso upright (avoid forward collapse)")

                self.record_rep()
                self.stage = "STAND"
                self.lowest_knee_angle = 180.0
                self.max_torso_lean_in_rep = 0.0

        # Feedback
        if smooth_torso > self.MAX_TORSO_LEAN:
            self.active_feedback.append("Chest up! Keep torso tall and upright")
            self.feedback_severity = "ALERT"
        elif self.stage == "BOTTOM":
            self.active_feedback.append("Great depth! Push back through front heel")
            self.feedback_severity = "GOOD"
        elif self.stage in ("STAND", "IDLE"):
            self.active_feedback.append("Step forward into lunge")

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=self.total_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            stage=self.stage,
            current_angles={
                "front_knee": round(smooth_knee, 1),
                "torso_lean": round(smooth_torso, 1)
            },
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=self.rep_progress,
            accuracy_percentage=self.accuracy,
            side_view=side
        )
