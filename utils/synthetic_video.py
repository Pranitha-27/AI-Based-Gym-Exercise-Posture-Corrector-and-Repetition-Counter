"""
Synthetic Pose Generator & Test Video Utilities.
Provides mathematically exact biomechanical trajectory generators for automated testing
and generates exercise-specific synthetic video files for all 6 supported exercises.
"""

from __future__ import annotations
import math
import os
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np

from core.geometry import Point
from core.pose_detector import PoseLandmarks, PoseLandmark


class MockPoseGenerator:
    """Generates synthetic PoseLandmarks streams for testing exercises."""

    @staticmethod
    def _create_angle_points(
        vertex: Point,
        p1: Point,
        target_angle_deg: float,
        p2_dist: float = 0.25,
        clockwise: bool = True
    ) -> Point:
        """Constructs p2 such that angle(p1, vertex, p2) == target_angle_deg."""
        v1 = (p1.x - vertex.x, p1.y - vertex.y)
        alpha = math.atan2(v1[1], v1[0])
        angle_rad = math.radians(target_angle_deg)
        beta = alpha + (angle_rad if clockwise else -angle_rad)
        return Point(
            vertex.x + p2_dist * math.cos(beta),
            vertex.y + p2_dist * math.sin(beta),
            0.0,
            1.0
        )

    @staticmethod
    def generate_squat_cycle(
        num_frames: int = 60,
        lowest_knee_angle: float = 85.0,
        torso_lean: float = 15.0,
        valgus: bool = False
    ) -> List[PoseLandmarks]:
        """Generate frames of a squat cycle from standing (170 deg) -> bottom -> standing."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            phase = math.sin((i / (num_frames - 1)) * math.pi)  # 0 -> 1 -> 0
            current_knee_angle = 170.0 - phase * (170.0 - lowest_knee_angle)

            hip_y = 0.4 + phase * 0.15
            hip = Point(0.5, hip_y, 0.0, 1.0)
            knee = Point(0.42, hip_y + 0.2, 0.0, 1.0)
            ankle = MockPoseGenerator._create_angle_points(knee, hip, current_knee_angle, p2_dist=0.22, clockwise=True)

            lean_rad = math.radians(torso_lean)
            shoulder = Point(
                hip.x + 0.25 * math.sin(lean_rad),
                hip.y - 0.25 * math.cos(lean_rad),
                0.0,
                1.0
            )

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_SHOULDER] = shoulder
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_KNEE] = knee
            pts[PoseLandmark.LEFT_ANKLE] = ankle

            right_knee_x = knee.x + (0.1 if not valgus else -0.05)
            pts[PoseLandmark.RIGHT_SHOULDER] = Point(shoulder.x + 0.05, shoulder.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_HIP] = Point(hip.x + 0.05, hip.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_KNEE] = Point(right_knee_x, knee.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_ANKLE] = Point(ankle.x + 0.05, ankle.y, 0.0, 1.0)

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @staticmethod
    def generate_pushup_cycle(
        num_frames: int = 60,
        lowest_elbow: float = 85.0,
        hip_angle: float = 175.0
    ) -> List[PoseLandmarks]:
        """Generate frames of a push-up cycle with exact elbow and hip angles."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            phase = math.sin((i / (num_frames - 1)) * math.pi)
            current_elbow = 165.0 - phase * (165.0 - lowest_elbow)

            shoulder = Point(0.25, 0.45 + phase * 0.1, 0.0, 1.0)
            elbow = Point(0.20, shoulder.y + 0.12, 0.0, 1.0)
            wrist = MockPoseGenerator._create_angle_points(elbow, shoulder, current_elbow, p2_dist=0.12, clockwise=True)

            hip = Point(0.52, shoulder.y + 0.02, 0.0, 1.0)
            ankle = MockPoseGenerator._create_angle_points(hip, shoulder, hip_angle, p2_dist=0.3, clockwise=False)

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_SHOULDER] = shoulder
            pts[PoseLandmark.LEFT_ELBOW] = elbow
            pts[PoseLandmark.LEFT_WRIST] = wrist
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_ANKLE] = ankle

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @staticmethod
    def generate_bicep_cycle(
        num_frames: int = 60,
        min_elbow: float = 38.0,
        swing: float = 10.0
    ) -> List[PoseLandmarks]:
        """Generate frames of a bicep curl cycle with exact angles."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            phase = math.sin((i / (num_frames - 1)) * math.pi)
            current_elbow = 160.0 - phase * (160.0 - min_elbow)

            shoulder = Point(0.5, 0.3, 0.0, 1.0)
            hip = Point(0.5, 0.6, 0.0, 1.0)

            swing_rad = math.radians(swing)
            elbow = Point(
                shoulder.x + 0.15 * math.sin(swing_rad),
                shoulder.y + 0.15 * math.cos(swing_rad),
                0.0,
                1.0
            )
            wrist = MockPoseGenerator._create_angle_points(elbow, shoulder, current_elbow, p2_dist=0.14, clockwise=True)

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_SHOULDER] = shoulder
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_ELBOW] = elbow
            pts[PoseLandmark.LEFT_WRIST] = wrist

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @staticmethod
    def generate_plank_cycle(num_frames: int = 60) -> List[PoseLandmarks]:
        """Generate frames of a plank hold with straight body alignment."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            # Subtle breathing motion (micro-oscillation in hip)
            micro_shift = math.sin((i / 15.0) * math.pi) * 0.003
            shoulder = Point(0.28, 0.52, 0.0, 1.0)
            hip = Point(0.50, 0.52 + micro_shift, 0.0, 1.0)
            ankle = Point(0.76, 0.52, 0.0, 1.0)

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_SHOULDER] = shoulder
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_ANKLE] = ankle
            pts[PoseLandmark.RIGHT_SHOULDER] = Point(shoulder.x + 0.04, shoulder.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_HIP] = Point(hip.x + 0.04, hip.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_ANKLE] = Point(ankle.x + 0.04, ankle.y, 0.0, 1.0)

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @staticmethod
    def generate_press_cycle(num_frames: int = 60) -> List[PoseLandmarks]:
        """Generate frames of an overhead shoulder press cycle."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            phase = math.sin((i / (num_frames - 1)) * math.pi)
            target_elbow = 85.0 + phase * (170.0 - 85.0)

            shoulder = Point(0.5, 0.35, 0.0, 1.0)
            hip = Point(0.5, 0.65, 0.0, 1.0)
            elbow = Point(0.40, 0.45, 0.0, 1.0)
            wrist = MockPoseGenerator._create_angle_points(elbow, shoulder, target_elbow, p2_dist=0.15, clockwise=True)

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_SHOULDER] = shoulder
            pts[PoseLandmark.LEFT_ELBOW] = elbow
            pts[PoseLandmark.LEFT_WRIST] = wrist

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @staticmethod
    def generate_lunge_cycle(num_frames: int = 60) -> List[PoseLandmarks]:
        """Generate frames of a walking/forward lunge cycle."""
        frames: List[PoseLandmarks] = []
        for i in range(num_frames):
            phase = math.sin((i / (num_frames - 1)) * math.pi)
            target_knee = 165.0 - phase * (165.0 - 90.0)

            hip = Point(0.5, 0.45 + phase * 0.12, 0.0, 1.0)
            knee = Point(0.40, hip.y + 0.20, 0.0, 1.0)
            ankle = MockPoseGenerator._create_angle_points(knee, hip, target_knee, p2_dist=0.22, clockwise=True)

            pts = [Point(0.5, 0.5, 0.0, 0.9)] * 33
            pts[PoseLandmark.LEFT_HIP] = hip
            pts[PoseLandmark.LEFT_KNEE] = knee
            pts[PoseLandmark.LEFT_ANKLE] = ankle
            pts[PoseLandmark.LEFT_SHOULDER] = Point(0.5, hip.y - 0.25, 0.0, 1.0)

            pts[PoseLandmark.RIGHT_HIP] = Point(0.52, hip.y, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_KNEE] = Point(0.60, hip.y + 0.20, 0.0, 1.0)
            pts[PoseLandmark.RIGHT_ANKLE] = Point(0.68, hip.y + 0.35, 0.0, 1.0)

            frames.append(PoseLandmarks(pts, width=640, height=480))
        return frames

    @classmethod
    def get_cycle_for_exercise(cls, exercise_name: str, num_frames: int = 60) -> List[PoseLandmarks]:
        """Returns the corresponding mock cycle for any requested exercise name."""
        name = exercise_name.lower().strip()
        if "push" in name:
            return cls.generate_pushup_cycle(num_frames)
        elif "curl" in name:
            return cls.generate_bicep_cycle(num_frames)
        elif "plank" in name:
            return cls.generate_plank_cycle(num_frames)
        elif "press" in name:
            return cls.generate_press_cycle(num_frames)
        elif "lunge" in name:
            return cls.generate_lunge_cycle(num_frames)
        else:  # Default squat
            return cls.generate_squat_cycle(num_frames)


def create_demo_video(
    output_path: str = "demo_workout.mp4",
    exercise_name: str = "squat",
    duration_sec: int = 6,
    fps: int = 30
) -> str:
    """
    Creates an exercise-specific synthetic video file with clear animations,
    ground mats, dumbbells, and stylized athletic figures for all 6 exercises.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    ex = exercise_name.lower().strip()

    # Colors
    c_bg = (24, 28, 34)         # Dark modern gym floor
    c_floor = (45, 52, 62)      # Floor perspective line
    c_mat = (30, 90, 130)       # Workout floor mat
    c_skin = (190, 210, 235)     # Athletic avatar skin tone
    c_shirt = (0, 180, 240)      # Electric cyan activewear
    c_shorts = (40, 40, 48)     # Dark shorts
    c_accent = (255, 255, 255)   # White details
    c_weights = (80, 85, 95)    # Dumbbell iron gray

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8)
        frame[:] = c_bg

        # Floor grid and workout mat
        cv2.rectangle(frame, (0, 390), (width, height), c_floor, -1)
        cv2.rectangle(frame, (60, 400), (580, 440), c_mat, -1)

        # Exercise title banner
        banner_txt = f"DEMONSTRATION: {exercise_name.upper()}"
        cv2.putText(frame, banner_txt, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2)
        cv2.putText(frame, "Automated Biomechanical Testing Feed", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1)

        # Repetition cycle phase: 0 -> 1 -> 0 every 2 seconds (60 frames)
        cycle_len = 60
        cycle_idx = i % cycle_len
        phase = math.sin((cycle_idx / (cycle_len - 1)) * math.pi)

        # Render specific exercise movement
        if "push" in ex:
            # PUSH-UP: Horizontal figure on floor
            hand_x, hand_y = 200, 395
            feet_x, feet_y = 510, 385
            # Chest lowers towards ground on phase
            chest_y = int(320 + phase * 65)
            hip_y = int(330 + phase * 55)

            # Feet / Shoes
            cv2.circle(frame, (feet_x, feet_y), 9, c_accent, -1)
            # Legs
            cv2.line(frame, (feet_x, feet_y), (360, hip_y), c_shorts, 12)
            # Torso / Shirt
            cv2.line(frame, (360, hip_y), (hand_x + 50, chest_y), c_shirt, 18)
            # Head
            cv2.circle(frame, (hand_x + 10, chest_y - 15), 18, c_skin, -1)
            # Arms (Upper arm + Forearm bending)
            elbow_x = hand_x + 35 - int(phase * 20)
            elbow_y = chest_y + int((1.0 - phase) * 20)
            cv2.line(frame, (hand_x + 40, chest_y), (elbow_x, elbow_y), c_skin, 8)
            cv2.line(frame, (elbow_x, elbow_y), (hand_x, hand_y), c_skin, 8)
            cv2.circle(frame, (hand_x, hand_y), 6, c_accent, -1)

        elif "curl" in ex:
            # BICEP CURL: Standing figure curling forearm
            cx = 320
            # Head
            cv2.circle(frame, (cx, 140), 20, c_skin, -1)
            # Torso
            cv2.line(frame, (cx, 160), (cx, 270), c_shirt, 20)
            # Legs
            cv2.line(frame, (cx - 10, 270), (cx - 15, 400), c_shorts, 14)
            cv2.line(frame, (cx + 10, 270), (cx + 15, 400), c_shorts, 14)
            # Upper arm stays vertical
            shoulder_x, shoulder_y = cx - 18, 175
            elbow_x, elbow_y = cx - 22, 245
            cv2.line(frame, (shoulder_x, shoulder_y), (elbow_x, elbow_y), c_skin, 10)
            # Forearm curls: from 160 deg (down) to 38 deg (up)
            curl_angle = math.radians(160.0 - phase * 120.0)
            forearm_len = 65
            wrist_x = int(elbow_x - forearm_len * math.sin(curl_angle))
            wrist_y = int(elbow_y - forearm_len * math.cos(curl_angle))
            cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), c_skin, 8)
            # Dumbbell
            cv2.circle(frame, (wrist_x, wrist_y), 11, c_weights, -1)
            cv2.circle(frame, (wrist_x, wrist_y), 6, (0, 200, 255), -1)

        elif "plank" in ex:
            # PLANK: Horizontal static hold
            elbow_x, elbow_y = 200, 395
            feet_x, feet_y = 490, 385
            # Subtle breathing displacement
            breathe = math.sin((i / 15.0) * math.pi) * 3
            hip_y = int(325 + breathe)
            shoulder_y = int(320 + breathe)

            # Feet
            cv2.circle(frame, (feet_x, feet_y), 8, c_accent, -1)
            # Legs
            cv2.line(frame, (feet_x, feet_y), (360, hip_y), c_shorts, 14)
            # Torso
            cv2.line(frame, (360, hip_y), (elbow_x + 30, shoulder_y), c_shirt, 18)
            # Head
            cv2.circle(frame, (elbow_x + 10, shoulder_y - 12), 16, c_skin, -1)
            # Forearm on ground & upper arm
            cv2.line(frame, (elbow_x + 30, shoulder_y), (elbow_x, elbow_y), c_skin, 10)
            cv2.line(frame, (elbow_x, elbow_y), (elbow_x + 40, elbow_y), c_skin, 8)

        elif "press" in ex:
            # SHOULDER PRESS: Standing, pressing dumbbells overhead
            cx = 320
            # Head
            cv2.circle(frame, (cx, 170), 20, c_skin, -1)
            # Torso
            cv2.line(frame, (cx, 190), (cx, 290), c_shirt, 22)
            # Legs
            cv2.line(frame, (cx - 15, 290), (cx - 18, 400), c_shorts, 14)
            cv2.line(frame, (cx + 15, 290), (cx + 18, 400), c_shorts, 14)

            # Left & Right arms pressing overhead
            # At phase 0: hands at shoulders (y = 190)
            # At phase 1: hands fully overhead (y = 80)
            hand_y = int(195 - phase * 115)
            for side_sign in (-1, 1):
                s_x = cx + side_sign * 22
                s_y = 195
                elbow_x = cx + side_sign * (38 - int(phase * 10))
                elbow_y = int(s_y + (1.0 - phase) * 20 - phase * 40)
                hand_x = cx + side_sign * 32
                cv2.line(frame, (s_x, s_y), (elbow_x, elbow_y), c_skin, 9)
                cv2.line(frame, (elbow_x, elbow_y), (hand_x, hand_y), c_skin, 8)
                # Dumbbells in both hands
                cv2.circle(frame, (hand_x, hand_y), 10, c_weights, -1)
                cv2.line(frame, (hand_x - 12, hand_y), (hand_x + 12, hand_y), (0, 200, 255), 3)

        elif "lunge" in ex:
            # LUNGE: Stepping forward into deep knee bend
            cx = 320
            hip_y = int(240 + phase * 65)
            # Torso stays upright
            cv2.circle(frame, (cx, hip_y - 120), 20, c_skin, -1)
            cv2.line(frame, (cx, hip_y - 100), (cx, hip_y), c_shirt, 18)

            # Front leg (bends to 90 deg forward)
            front_knee_x = int(cx - phase * 65)
            front_knee_y = int(hip_y + phase * 50 + (1.0 - phase) * 80)
            front_foot_x = int(cx - phase * 65)
            front_foot_y = 400
            cv2.line(frame, (cx, hip_y), (front_knee_x, front_knee_y), c_shorts, 14)
            cv2.line(frame, (front_knee_x, front_knee_y), (front_foot_x, front_foot_y), c_skin, 10)
            cv2.circle(frame, (front_foot_x, front_foot_y), 7, c_accent, -1)

            # Back leg (drops knee down)
            back_knee_x = int(cx + phase * 50)
            back_knee_y = int(hip_y + phase * 75 + (1.0 - phase) * 80)
            cv2.line(frame, (cx, hip_y), (back_knee_x, back_knee_y), c_shorts, 12)
            cv2.line(frame, (back_knee_x, back_knee_y), (back_knee_x + 35, 400), c_skin, 8)

            # Arms on hips
            cv2.line(frame, (cx, hip_y - 80), (cx - 25, hip_y - 30), c_skin, 7)
            cv2.line(frame, (cx - 25, hip_y - 30), (cx - 10, hip_y), c_skin, 7)

        else:
            # SQUAT (Default)
            cx = 320
            hip_y = int(230 + phase * 75)
            knee_y = int(hip_y + 80 - phase * 15)
            knee_x = int(cx - phase * 40)

            # Head
            cv2.circle(frame, (cx, hip_y - 120), 22, c_skin, -1)
            # Torso
            cv2.line(frame, (cx, hip_y - 95), (cx, hip_y), c_shirt, 18)
            # Thigh
            cv2.line(frame, (cx, hip_y), (knee_x, knee_y), c_shorts, 14)
            # Shin to ground
            cv2.line(frame, (knee_x, knee_y), (cx, 400), c_skin, 10)
            cv2.circle(frame, (cx, 400), 8, c_accent, -1)
            # Arms extended forward for balance
            cv2.line(frame, (cx, hip_y - 80), (cx - 45, hip_y - 45), c_skin, 8)

        out.write(frame)

    out.release()
    return output_path
