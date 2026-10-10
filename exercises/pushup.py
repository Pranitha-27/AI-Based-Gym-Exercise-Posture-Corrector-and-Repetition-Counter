"""
Push-up Exercise Posture Monitor.
Evaluates elbow depth, spine/hip plank alignment (sagging/piking), and neck posture.
"""

from __future__ import annotations
from typing import Dict, List
from core.geometry import calculate_angle_2d, calculate_inclination
from core.pose_detector import PoseLandmarks
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class PushupExercise(BaseExercise):
    """
    Monitors push-ups:
    - Depth: Elbow flexion should reach <= 90 degrees at bottom.
    - Body Line: Shoulder-Hip-Ankle line should remain straight (160 - 180 degrees).
    - Prevents lower back sagging or excessive hip pike.
    """

    PLANK_ELBOW_UP = 155.0       # Elbow extended (top of push-up)
    PUSHUP_DEPTH_TARGET = 90.0   # Elbow at bottom
    MIN_HIP_PLANK_ANGLE = 155.0  # Under this = hips sagging
    MAX_HIP_PLANK_ANGLE = 195.0  # Over this = hips piking

    def __init__(self):
        super().__init__("Push-up")
        self.lowest_elbow_angle = 180.0
        self.min_hip_angle_in_rep = 180.0
        self.max_hip_angle_in_rep = 180.0

    def reset(self):
        super().reset()
        self.lowest_elbow_angle = 180.0
        self.min_hip_angle_in_rep = 180.0
        self.max_hip_angle_in_rep = 180.0

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"

        left_vis = (landmarks.left_shoulder.visibility + landmarks.left_elbow.visibility + landmarks.left_hip.visibility) / 3.0
        right_vis = (landmarks.right_shoulder.visibility + landmarks.right_elbow.visibility + landmarks.right_hip.visibility) / 3.0

        if left_vis >= right_vis:
            side = "LEFT"
            elbow_angle = calculate_angle_2d(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist)
            hip_angle = calculate_angle_2d(landmarks.left_shoulder, landmarks.left_hip, landmarks.left_ankle)
        else:
            side = "RIGHT"
            elbow_angle = calculate_angle_2d(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)
            hip_angle = calculate_angle_2d(landmarks.right_shoulder, landmarks.right_hip, landmarks.right_ankle)

        smooth_elbow = self.filter.filter("elbow_angle", elbow_angle)
        smooth_hip = self.filter.filter("hip_angle", hip_angle)

        # Progress bar: 155 deg -> 0%, 90 deg -> 100%
        self.rep_progress = max(0.0, min(1.0, (self.PLANK_ELBOW_UP - smooth_elbow) / (self.PLANK_ELBOW_UP - self.PUSHUP_DEPTH_TARGET)))

        # State Machine Transitions
        if self.stage in ("IDLE", "UP"):
            if smooth_elbow < self.PLANK_ELBOW_UP - 12.0:
                self.stage = "DESCENDING"
                self.lowest_elbow_angle = smooth_elbow
                self.min_hip_angle_in_rep = smooth_hip
                self.max_hip_angle_in_rep = smooth_hip
        elif self.stage == "DESCENDING":
            if smooth_elbow < self.lowest_elbow_angle:
                self.lowest_elbow_angle = smooth_elbow
            self.min_hip_angle_in_rep = min(self.min_hip_angle_in_rep, smooth_hip)
            self.max_hip_angle_in_rep = max(self.max_hip_angle_in_rep, smooth_hip)

            if smooth_elbow <= self.PUSHUP_DEPTH_TARGET:
                self.stage = "BOTTOM"
            elif smooth_elbow > self.lowest_elbow_angle + 12.0:
                self.stage = "ASCENDING"
        elif self.stage == "BOTTOM":
            if smooth_elbow < self.lowest_elbow_angle:
                self.lowest_elbow_angle = smooth_elbow
            self.min_hip_angle_in_rep = min(self.min_hip_angle_in_rep, smooth_hip)

            if smooth_elbow > self.PUSHUP_DEPTH_TARGET + 15.0:
                self.stage = "ASCENDING"
        elif self.stage == "ASCENDING":
            self.min_hip_angle_in_rep = min(self.min_hip_angle_in_rep, smooth_hip)

            if smooth_elbow >= self.PLANK_ELBOW_UP - 5.0:
                # Rep completed! Validate form
                if self.lowest_elbow_angle > 105.0:
                    self.current_rep_faults.append("Lower your chest more (reach 90 deg)")
                if self.min_hip_angle_in_rep < self.MIN_HIP_PLANK_ANGLE:
                    self.current_rep_faults.append("Hips sagging! Engage core and glutes")
                if self.max_hip_angle_in_rep > self.MAX_HIP_PLANK_ANGLE:
                    self.current_rep_faults.append("Hips too high! Maintain straight plank line")

                self.record_rep()
                self.stage = "UP"
                self.lowest_elbow_angle = 180.0
                self.min_hip_angle_in_rep = 180.0
                self.max_hip_angle_in_rep = 180.0

        # Real-time feedback cues
        if smooth_hip < self.MIN_HIP_PLANK_ANGLE:
            self.active_feedback.append("Lift hips! Don't let your lower back sag")
            self.feedback_severity = "ALERT"
        elif smooth_hip > self.MAX_HIP_PLANK_ANGLE:
            self.active_feedback.append("Lower hips into a flat plank line")
            self.feedback_severity = "WARNING"
        elif self.stage == "BOTTOM":
            self.active_feedback.append("Good depth! Push the ground away")
            self.feedback_severity = "GOOD"
        elif self.stage == "ASCENDING" and self.lowest_elbow_angle > 105.0:
            self.active_feedback.append("Go deeper next time!")
            self.feedback_severity = "WARNING"
        elif self.stage in ("UP", "IDLE"):
            self.active_feedback.append("Plank straight, ready to descend")

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=self.total_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            stage=self.stage,
            current_angles={
                "elbow_angle": round(smooth_elbow, 1),
                "plank_hip_angle": round(smooth_hip, 1)
            },
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=self.rep_progress,
            accuracy_percentage=self.accuracy,
            side_view=side,
            extra_data={"lowest_elbow": round(self.lowest_elbow_angle, 1)}
        )
