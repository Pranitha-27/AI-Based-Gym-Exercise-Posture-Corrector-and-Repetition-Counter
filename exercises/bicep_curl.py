"""
Bicep Curl Posture Monitor.
Monitors arm range of motion (ROM) and flags elbow swinging / cheating.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from core.geometry import calculate_angle_2d
from core.pose_detector import PoseLandmarks
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class BicepCurlExercise(BaseExercise):
    """
    Monitors Bicep Curls:
    - Full Extension: Elbow angle > 150 degrees at bottom.
    - Full Flexion: Elbow angle < 45 degrees at top.
    - Upper Arm Stability: Hip-Shoulder-Elbow angle should stay <= 25 degrees (no swinging).
    """

    EXTENDED_ANGLE = 150.0  # Bottom of curl
    FLEXED_ANGLE = 45.0     # Top of curl
    MAX_ELBOW_SWING = 25.0  # Degrees upper arm moves away from torso

    def __init__(self, target_arm: str = "AUTO"):
        """
        Args:
            target_arm: 'LEFT', 'RIGHT', or 'AUTO' (picks more visible or moving arm)
        """
        super().__init__("Bicep Curl")
        self.target_arm = target_arm
        self.max_curl_angle = 180.0
        self.min_curl_angle = 180.0
        self.max_swing_in_rep = 0.0

    def reset(self):
        super().reset()
        self.max_curl_angle = 180.0
        self.min_curl_angle = 180.0
        self.max_swing_in_rep = 0.0

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"

        left_vis = (landmarks.left_shoulder.visibility + landmarks.left_elbow.visibility + landmarks.left_wrist.visibility) / 3.0
        right_vis = (landmarks.right_shoulder.visibility + landmarks.right_elbow.visibility + landmarks.right_wrist.visibility) / 3.0

        left_elbow = calculate_angle_2d(landmarks.left_shoulder, landmarks.left_elbow, landmarks.left_wrist)
        right_elbow = calculate_angle_2d(landmarks.right_shoulder, landmarks.right_elbow, landmarks.right_wrist)

        left_swing = calculate_angle_2d(landmarks.left_hip, landmarks.left_shoulder, landmarks.left_elbow)
        right_swing = calculate_angle_2d(landmarks.right_hip, landmarks.right_shoulder, landmarks.right_elbow)

        # Arm selection
        if self.target_arm == "LEFT":
            active_side = "LEFT"
            elbow_angle = left_elbow
            swing_angle = left_swing
        elif self.target_arm == "RIGHT":
            active_side = "RIGHT"
            elbow_angle = right_elbow
            swing_angle = right_swing
        else:  # AUTO: select based on visibility and motion
            if left_vis >= right_vis:
                active_side = "LEFT"
                elbow_angle = left_elbow
                swing_angle = left_swing
            else:
                active_side = "RIGHT"
                elbow_angle = right_elbow
                swing_angle = right_swing

        smooth_elbow = self.filter.filter("bicep_elbow", elbow_angle)
        smooth_swing = self.filter.filter("bicep_swing", swing_angle)

        # Rep progress: 150 deg -> 0%, 45 deg -> 100%
        self.rep_progress = max(0.0, min(1.0, (self.EXTENDED_ANGLE - smooth_elbow) / (self.EXTENDED_ANGLE - self.FLEXED_ANGLE)))

        # State Machine Transitions
        if self.stage in ("IDLE", "DOWN"):
            if smooth_elbow < self.EXTENDED_ANGLE - 15.0:
                self.stage = "CURLING_UP"
                self.min_curl_angle = smooth_elbow
                self.max_swing_in_rep = smooth_swing
        elif self.stage == "CURLING_UP":
            if smooth_elbow < self.min_curl_angle:
                self.min_curl_angle = smooth_elbow
            self.max_swing_in_rep = max(self.max_swing_in_rep, smooth_swing)

            if smooth_elbow <= self.FLEXED_ANGLE:
                self.stage = "PEAK"
            elif smooth_elbow > self.min_curl_angle + 20.0:
                self.stage = "LOWERING"
        elif self.stage == "PEAK":
            if smooth_elbow < self.min_curl_angle:
                self.min_curl_angle = smooth_elbow
            self.max_swing_in_rep = max(self.max_swing_in_rep, smooth_swing)

            if smooth_elbow > self.FLEXED_ANGLE + 20.0:
                self.stage = "LOWERING"
        elif self.stage == "LOWERING":
            self.max_swing_in_rep = max(self.max_swing_in_rep, smooth_swing)

            if smooth_elbow >= self.EXTENDED_ANGLE - 10.0:
                # Rep completed! Validate form
                if self.min_curl_angle > 65.0:
                    self.current_rep_faults.append("Curl higher (incomplete contraction)")
                if self.max_swing_in_rep > self.MAX_ELBOW_SWING:
                    self.current_rep_faults.append("Elbow swung forward (keep elbows tucked to sides)")

                self.record_rep()
                self.stage = "DOWN"
                self.min_curl_angle = 180.0
                self.max_swing_in_rep = 0.0

        # Real-time feedback
        if smooth_swing > self.MAX_ELBOW_SWING:
            self.active_feedback.append("Keep elbow pinned! Stop swinging upper arm")
            self.feedback_severity = "ALERT"
        elif self.stage == "PEAK":
            self.active_feedback.append("Squeeze bicep at peak!")
            self.feedback_severity = "GOOD"
        elif self.stage == "LOWERING" and self.min_curl_angle > 65.0:
            self.active_feedback.append("Full contraction next rep!")
            self.feedback_severity = "WARNING"
        elif self.stage in ("DOWN", "IDLE"):
            self.active_feedback.append(f"{active_side} arm ready. Keep shoulders steady")

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=self.total_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            stage=self.stage,
            current_angles={
                "elbow_angle": round(smooth_elbow, 1),
                "elbow_swing": round(smooth_swing, 1)
            },
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=self.rep_progress,
            accuracy_percentage=self.accuracy,
            side_view=active_side,
            extra_data={"target_arm": active_side}
        )
