"""
Squat Exercise Posture Monitor.
Evaluates squat depth, torso inclination (chest up), and knee valgus (caving knees).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import math
from core.geometry import calculate_angle_2d, calculate_inclination, calculate_distance_2d
from core.pose_detector import PoseLandmarks, PoseLandmark
from exercises.base_exercise import BaseExercise, ExerciseMetrics


class SquatExercise(BaseExercise):
    """
    Monitors squats:
    - Depth: Knee angle should reach <= 95 degrees for full depth.
    - Torso: Back angle relative to vertical should stay <= 45 degrees.
    - Knee Valgus: In front view, knees should not cave inward.
    """

    STAND_THRESHOLD = 160.0    # Degrees above which user is standing
    SQUAT_DEPTH_TARGET = 95.0  # Degrees at bottom of parallel squat
    MAX_TORSO_LEAN = 45.0      # Max degrees torso can lean from vertical

    def __init__(self):
        super().__init__("Squat")
        self.lowest_knee_angle = 180.0
        self.max_torso_lean_in_rep = 0.0
        self.knee_valgus_detected = False

    def reset(self):
        super().reset()
        self.lowest_knee_angle = 180.0
        self.max_torso_lean_in_rep = 0.0
        self.knee_valgus_detected = False

    def evaluate(self, landmarks: PoseLandmarks) -> ExerciseMetrics:
        self.active_feedback.clear()
        self.feedback_severity = "GOOD"

        # Determine best perspective (Left, Right, or Front)
        left_vis = (landmarks.left_hip.visibility + landmarks.left_knee.visibility + landmarks.left_ankle.visibility) / 3.0
        right_vis = (landmarks.right_hip.visibility + landmarks.right_knee.visibility + landmarks.right_ankle.visibility) / 3.0

        is_front_view = abs(left_vis - right_vis) < 0.2 and left_vis > 0.4 and right_vis > 0.4

        # Compute left and right angles
        left_knee_angle = calculate_angle_2d(landmarks.left_hip, landmarks.left_knee, landmarks.left_ankle)
        right_knee_angle = calculate_angle_2d(landmarks.right_hip, landmarks.right_knee, landmarks.right_ankle)

        # Torso inclination
        if left_vis >= right_vis:
            side = "LEFT"
            primary_knee = left_knee_angle
            torso_lean = calculate_inclination(landmarks.left_hip, landmarks.left_shoulder, reference="vertical")
        else:
            side = "RIGHT"
            primary_knee = right_knee_angle
            torso_lean = calculate_inclination(landmarks.right_hip, landmarks.right_shoulder, reference="vertical")

        if is_front_view:
            side = "FRONT"
            primary_knee = (left_knee_angle + right_knee_angle) / 2.0

        # Apply smoothing
        smooth_knee = self.filter.filter("knee_angle", primary_knee)
        smooth_torso = self.filter.filter("torso_lean", torso_lean)

        # Calculate progress: 165 deg -> 0%, 90 deg -> 100%
        self.rep_progress = max(0.0, min(1.0, (self.STAND_THRESHOLD - smooth_knee) / (self.STAND_THRESHOLD - self.SQUAT_DEPTH_TARGET)))

        # State Machine Transitions
        if self.stage in ("IDLE", "STAND"):
            if smooth_knee < self.STAND_THRESHOLD - 10.0:
                self.stage = "DESCENDING"
                self.lowest_knee_angle = smooth_knee
                self.max_torso_lean_in_rep = smooth_torso
                self.knee_valgus_detected = False
        elif self.stage == "DESCENDING":
            if smooth_knee < self.lowest_knee_angle:
                self.lowest_knee_angle = smooth_knee
            if smooth_torso > self.max_torso_lean_in_rep:
                self.max_torso_lean_in_rep = smooth_torso

            if smooth_knee <= self.SQUAT_DEPTH_TARGET:
                self.stage = "BOTTOM"
            elif smooth_knee > self.lowest_knee_angle + 12.0:
                # Started ascending without reaching proper depth
                self.stage = "ASCENDING"
        elif self.stage == "BOTTOM":
            if smooth_knee < self.lowest_knee_angle:
                self.lowest_knee_angle = smooth_knee
            if smooth_torso > self.max_torso_lean_in_rep:
                self.max_torso_lean_in_rep = smooth_torso

            if smooth_knee > self.SQUAT_DEPTH_TARGET + 15.0:
                self.stage = "ASCENDING"
        elif self.stage == "ASCENDING":
            if smooth_torso > self.max_torso_lean_in_rep:
                self.max_torso_lean_in_rep = smooth_torso

            if smooth_knee >= self.STAND_THRESHOLD - 5.0:
                # Rep completed! Validate form
                if self.lowest_knee_angle > 105.0:
                    self.current_rep_faults.append("Squat deeper (aim for parallel)")
                if self.max_torso_lean_in_rep > self.MAX_TORSO_LEAN:
                    self.current_rep_faults.append("Keep chest upright (excessive forward lean)")
                if self.knee_valgus_detected:
                    self.current_rep_faults.append("Knees caved inward (push knees out)")

                self.record_rep()
                self.stage = "STAND"
                self.lowest_knee_angle = 180.0
                self.max_torso_lean_in_rep = 0.0

        # Real-time posture feedback cues during movement
        if self.stage in ("DESCENDING", "BOTTOM"):
            if smooth_torso > self.MAX_TORSO_LEAN:
                self.active_feedback.append("Lift chest up! Spine leaning too far forward")
                self.feedback_severity = "ALERT"

            # Check knee valgus if front view
            if is_front_view and self.stage == "BOTTOM":
                knee_dist = calculate_distance_2d(landmarks.left_knee, landmarks.right_knee)
                ankle_dist = calculate_distance_2d(landmarks.left_ankle, landmarks.right_ankle)
                if ankle_dist > 0.05 and (knee_dist / ankle_dist) < 0.8:
                    self.knee_valgus_detected = True
                    self.active_feedback.append("Push knees outward! Avoid knee cave")
                    self.feedback_severity = "WARNING"

            if self.stage == "BOTTOM" and len(self.active_feedback) == 0:
                self.active_feedback.append("Great depth! Drive up through your heels")
                self.feedback_severity = "GOOD"
        elif self.stage == "ASCENDING" and self.lowest_knee_angle > 105.0:
            self.active_feedback.append("Go deeper next rep!")
            self.feedback_severity = "WARNING"
        elif self.stage in ("STAND", "IDLE"):
            if len(self.current_rep_faults) == 0 and self.total_reps > 0:
                self.active_feedback.append("Good form! Ready for next rep")
            else:
                self.active_feedback.append("Stand tall, feet shoulder-width apart")

        current_angles = {
            "knee_angle": round(smooth_knee, 1),
            "torso_lean": round(smooth_torso, 1),
        }
        if is_front_view:
            current_angles["left_knee"] = round(left_knee_angle, 1)
            current_angles["right_knee"] = round(right_knee_angle, 1)

        return ExerciseMetrics(
            exercise_name=self.name,
            total_reps=self.total_reps,
            good_reps=self.good_reps,
            bad_reps=self.bad_reps,
            stage=self.stage,
            current_angles=current_angles,
            active_feedback=self.active_feedback,
            feedback_severity=self.feedback_severity,
            rep_progress=self.rep_progress,
            accuracy_percentage=self.accuracy,
            side_view=side,
            extra_data={"lowest_depth": round(self.lowest_knee_angle, 1)}
        )
