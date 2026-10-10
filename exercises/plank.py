"""
Plank Posture Monitor.
Measures isometric hold duration, good-form time vs bad-form time,
and flags hip sagging or piking.
"""

from __future__ import annotations
import time
from typing import Dict, List
from core.geometry import calculate_angle_2d
from core.pose_detector import PoseLandmarks
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class PlankExercise(BaseExercise):
    """
    Monitors Planks:
    - Straight Spine Line: Shoulder-Hip-Ankle should be 160-180 degrees.
    - Hip Sagging: Flags if hip drops below 155 degrees.
    - Hip Piking: Flags if hip raises above 195 degrees.
    - Tracks total hold time and good-form time.
    """

    MIN_HIP_ANGLE = 155.0  # Under this = hips sagging
    MAX_HIP_ANGLE = 195.0  # Over this = hips piking

    def __init__(self):
        super().__init__("Plank")
        self.hold_start_time: float = 0.0
        self.is_holding: bool = False
        self.total_hold_seconds: float = 0.0
        self.good_hold_seconds: float = 0.0
        self.bad_hold_seconds: float = 0.0
        self.last_update_time: float = time.time()

    def reset(self):
        super().reset()
        self.hold_start_time = 0.0
        self.is_holding = False
        self.total_hold_seconds = 0.0
        self.good_hold_seconds = 0.0
        self.bad_hold_seconds = 0.0
        self.last_update_time = time.time()

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"
        now = time.time()
        dt = min(now - self.last_update_time, 0.5)  # Cap dt to avoid huge jumps on pause
        self.last_update_time = now

        left_vis = (landmarks.left_shoulder.visibility + landmarks.left_hip.visibility + landmarks.left_ankle.visibility) / 3.0
        right_vis = (landmarks.right_shoulder.visibility + landmarks.right_hip.visibility + landmarks.right_ankle.visibility) / 3.0

        if left_vis >= right_vis:
            side = "LEFT"
            hip_angle = calculate_angle_2d(landmarks.left_shoulder, landmarks.left_hip, landmarks.left_ankle)
        else:
            side = "RIGHT"
            hip_angle = calculate_angle_2d(landmarks.right_shoulder, landmarks.right_hip, landmarks.right_ankle)

        smooth_hip = self.filter.filter("plank_hip", hip_angle)

        # Detect if person is in plank position (horizontal body, shoulders and hips off ground)
        in_plank = 130.0 < smooth_hip < 220.0 and landmarks.left_shoulder.y > 0.2 and landmarks.left_hip.y > 0.2

        if in_plank:
            if not self.is_holding:
                self.is_holding = True
                self.stage = "HOLDING"
                self.hold_start_time = now

            self.total_hold_seconds += dt

            # Form evaluation
            if smooth_hip < self.MIN_HIP_ANGLE:
                self.bad_hold_seconds += dt
                self.active_feedback.append("Lift hips! Core is sagging, protect your back")
                self.feedback_severity = "ALERT"
                self.current_rep_faults.append("Hip sagging")
            elif smooth_hip > self.MAX_HIP_ANGLE:
                self.bad_hold_seconds += dt
                self.active_feedback.append("Lower hips into a flat plank line")
                self.feedback_severity = "WARNING"
                self.current_rep_faults.append("Hip piking")
            else:
                self.good_hold_seconds += dt
                self.active_feedback.append(f"Solid plank! Holding strong ({int(self.total_hold_seconds)}s)")
                self.feedback_severity = "GOOD"
        else:
            if self.is_holding:
                # Plank completed / rested
                self.is_holding = False
                self.stage = "REST"
                if self.total_hold_seconds >= 5.0:
                    self.record_rep()
            self.active_feedback.append("Get into plank position (shoulders over wrists/elbows)")

        # Accuracy calculated by percentage of time held in good form
        if self.total_hold_seconds > 0.0:
            form_accuracy = (self.good_hold_seconds / self.total_hold_seconds) * 100.0
        else:
            form_accuracy = 100.0

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=int(self.total_hold_seconds),  # For plank, show seconds held as total
            good_reps=int(self.good_hold_seconds),
            bad_reps=int(self.bad_hold_seconds),
            stage=self.stage,
            current_angles={"hip_angle": round(smooth_hip, 1)},
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=min(1.0, (self.total_hold_seconds % 30.0) / 30.0),  # 30s milestone progress
            accuracy_percentage=round(form_accuracy, 1),
            side_view=side,
            extra_data={
                "hold_time_seconds": round(self.total_hold_seconds, 1),
                "good_time": round(self.good_hold_seconds, 1)
            }
        )
