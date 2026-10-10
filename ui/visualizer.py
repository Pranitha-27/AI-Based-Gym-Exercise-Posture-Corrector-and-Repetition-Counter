"""
Real-Time HUD Visualizer for Posture Monitoring.
Renders high-contrast dark-mode glassmorphic overlays, skeleton tracking,
angle degree badges, rep progress meters, and dynamic posture feedback banners.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from core.geometry import Point
from core.pose_detector import PoseLandmarks, PoseLandmark, POSE_CONNECTIONS
from exercises.base_exercise import ExerciseMetrics


class Visualizer:
    """Renders professional biomechanical overlays on top of video frames."""

    # Color palette (BGR for OpenCV)
    COLOR_BG_DARK = (20, 24, 28)
    COLOR_ACCENT = (245, 140, 20)       # Electric cyan-blue
    COLOR_GOOD = (70, 200, 70)          # Emerald Green
    COLOR_WARNING = (30, 180, 255)       # Amber/Orange
    COLOR_ALERT = (50, 50, 240)         # Crimson Red
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (140, 140, 140)
    COLOR_SKELETON_DEFAULT = (200, 200, 200)

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hud(
        self,
        frame: np.ndarray,
        metrics: ExerciseMetrics,
        fps: float,
        audio_enabled: bool = True
    ) -> np.ndarray:
        """
        Draw complete HUD overlay onto frame:
        - Top Header (Exercise, Stats, Rep Counter, Accuracy, FPS)
        - Bottom Banner (Posture feedback & hotkey hints)
        - Side Rep Progress Meter
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Top Glassmorphic Panel
        header_h = 95
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Title & Exercise Badge
        cv2.putText(frame, "AI POSTURE COACH", (20, 28), self.font, 0.65, self.COLOR_ACCENT, 2, cv2.LINE_AA)
        exercise_txt = f"{metrics.exercise_name.upper()}"
        cv2.putText(frame, exercise_txt, (20, 68), self.font, 1.1, self.COLOR_WHITE, 3, cv2.LINE_AA)

        # Stage Badge
        stage_txt = f"[{metrics.stage}]"
        cv2.putText(frame, stage_txt, (20 + int(len(exercise_txt) * 22), 64), self.font, 0.65, self.COLOR_GRAY, 2, cv2.LINE_AA)

        # Rep Counters in Center
        col_x = max(340, int(w * 0.38))
        # Total Reps
        cv2.putText(frame, "TOTAL REPS", (col_x, 26), self.font, 0.5, self.COLOR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, str(metrics.total_reps), (col_x, 68), self.font, 1.3, self.COLOR_WHITE, 3, cv2.LINE_AA)

        # Good Reps
        cv2.putText(frame, "GOOD", (col_x + 130, 26), self.font, 0.5, self.COLOR_GOOD, 1, cv2.LINE_AA)
        cv2.putText(frame, str(metrics.good_reps), (col_x + 130, 68), self.font, 1.3, self.COLOR_GOOD, 3, cv2.LINE_AA)

        # Bad Reps
        cv2.putText(frame, "BAD", (col_x + 230, 26), self.font, 0.5, self.COLOR_ALERT, 1, cv2.LINE_AA)
        cv2.putText(frame, str(metrics.bad_reps), (col_x + 230, 68), self.font, 1.3, self.COLOR_ALERT, 3, cv2.LINE_AA)

        # Accuracy Score
        cv2.putText(frame, "FORM SCORE", (col_x + 330, 26), self.font, 0.5, self.COLOR_GRAY, 1, cv2.LINE_AA)
        acc_txt = f"{metrics.accuracy_percentage}%"
        acc_color = self.COLOR_GOOD if metrics.accuracy_percentage >= 80 else (self.COLOR_WARNING if metrics.accuracy_percentage >= 60 else self.COLOR_ALERT)
        cv2.putText(frame, acc_txt, (col_x + 330, 68), self.font, 1.2, acc_color, 3, cv2.LINE_AA)

        # Right Corner: FPS & Audio Status
        fps_txt = f"{int(fps)} FPS"
        cv2.putText(frame, fps_txt, (w - 110, 32), self.font, 0.6, self.COLOR_WHITE, 2, cv2.LINE_AA)
        audio_txt = "AUDIO: ON" if audio_enabled else "AUDIO: OFF"
        cv2.putText(frame, audio_txt, (w - 130, 64), self.font, 0.5, self.COLOR_ACCENT if audio_enabled else self.COLOR_GRAY, 1, cv2.LINE_AA)

        # 2. Side Rep Progress Meter (Vertical Bar on Left)
        self._draw_progress_bar(frame, metrics.rep_progress, h)

        # 3. Bottom Feedback Banner
        self._draw_bottom_banner(frame, metrics, w, h)

        return frame

    def _draw_progress_bar(self, frame: np.ndarray, progress: float, h: int):
        """Draws a vertical progress bar indicating depth/motion completeness."""
        bar_w = 20
        bar_x = 20
        bar_y_start = 120
        bar_y_end = h - 120
        bar_h = bar_y_end - bar_y_start

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y_start), (bar_x + bar_w, bar_y_end), self.COLOR_BG_DARK, -1)
        cv2.rectangle(frame, (bar_x, bar_y_start), (bar_x + bar_w, bar_y_end), (80, 80, 80), 1)

        # Filled progress
        filled_h = int(bar_h * progress)
        fill_color = self.COLOR_GOOD if progress >= 0.95 else self.COLOR_ACCENT
        cv2.rectangle(frame, (bar_x, bar_y_end - filled_h), (bar_x + bar_w, bar_y_end), fill_color, -1)

        # Percentage text
        pct_txt = f"{int(progress * 100)}%"
        cv2.putText(frame, pct_txt, (bar_x - 4, bar_y_start - 10), self.font, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_bottom_banner(self, frame: np.ndarray, metrics: ExerciseMetrics, w: int, h: int):
        """Draws dynamic alert pill and keyboard hotkey guide."""
        banner_h = 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Select color based on feedback severity
        if metrics.feedback_severity == "ALERT":
            status_color = self.COLOR_ALERT
            prefix = "ALERT: "
        elif metrics.feedback_severity == "WARNING":
            status_color = self.COLOR_WARNING
            prefix = "CAUTION: "
        else:
            status_color = self.COLOR_GOOD
            prefix = "FORM: "

        feedback_str = " | ".join(metrics.active_feedback) if metrics.active_feedback else "Maintaining good form"
        full_banner_text = prefix + feedback_str

        # Draw status message
        cv2.putText(frame, full_banner_text, (20, h - 42), self.font, 0.75, status_color, 2, cv2.LINE_AA)

        # Draw Hotkey Hint Bar
        hints = "[1] Squat  [2] Push-up  [3] Curl  [4] Plank  [5] Press  [6] Lunge  |  [R] Reset  [M] Mute  [S] Save  [Q] Quit"
        cv2.putText(frame, hints, (20, h - 14), self.font, 0.45, self.COLOR_GRAY, 1, cv2.LINE_AA)

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: PoseLandmarks,
        metrics: ExerciseMetrics,
        min_visibility: float = 0.4
    ):
        """Draws pose skeleton with color coding based on posture accuracy."""
        h, w = frame.shape[:2]

        # Draw connections
        bone_color = self.COLOR_GOOD if metrics.feedback_severity == "GOOD" else (
            self.COLOR_WARNING if metrics.feedback_severity == "WARNING" else self.COLOR_ALERT
        )

        for p1_idx, p2_idx in POSE_CONNECTIONS:
            pt1 = landmarks.get(p1_idx)
            pt2 = landmarks.get(p2_idx)

            if pt1.visibility >= min_visibility and pt2.visibility >= min_visibility:
                x1, y1 = pt1.to_pixel(w, h)
                x2, y2 = pt2.to_pixel(w, h)
                cv2.line(frame, (x1, y1), (x2, y2), bone_color, 3, cv2.LINE_AA)

        # Draw key landmark joints with glowing dual circles
        for idx in range(len(landmarks.landmarks)):
            pt = landmarks.get(idx)
            if pt.visibility >= min_visibility:
                px, py = pt.to_pixel(w, h)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, bone_color, 2, cv2.LINE_AA)

    def draw_angle_badge(
        self,
        frame: np.ndarray,
        center_pt: Point,
        angle: float,
        label: Optional[str] = None,
        is_safe: bool = True
    ):
        """
        Draws an informative circular angle badge with degree reading
        over a specific joint (e.g. knee, elbow).
        """
        h, w = frame.shape[:2]
        cx, cy = center_pt.to_pixel(w, h)
        radius = 24

        badge_color = self.COLOR_GOOD if is_safe else self.COLOR_ALERT

        # Draw translucent badge background
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), radius, self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Outer border
        cv2.circle(frame, (cx, cy), radius, badge_color, 2, cv2.LINE_AA)

        # Angle text
        deg_str = f"{int(angle)}"
        (tw, th), _ = cv2.getTextSize(deg_str, self.font, 0.48, 1)
        cv2.putText(frame, deg_str, (cx - tw // 2, cy + th // 2), self.font, 0.48, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Optional label below
        if label:
            cv2.putText(frame, label, (cx - 20, cy + radius + 14), self.font, 0.4, self.COLOR_WHITE, 1, cv2.LINE_AA)
