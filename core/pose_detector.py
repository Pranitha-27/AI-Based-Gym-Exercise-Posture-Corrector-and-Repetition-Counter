"""
Pose Detector module leveraging MediaPipe Tasks PoseLandmarker API.
Handles model lifecycle, inference, and convenient object-oriented landmark access.
"""

from __future__ import annotations
import os
import urllib.request
from enum import IntEnum
from typing import Optional, List, Dict, Tuple
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.geometry import Point


class PoseLandmark(IntEnum):
    """Standard 33 MediaPipe Pose Landmark Indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Standard anatomical skeleton pairs for visualization
POSE_CONNECTIONS: List[Tuple[PoseLandmark, PoseLandmark]] = [
    # Torso
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    # Left Arm
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    # Right Arm
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    # Left Leg
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    # Right Leg
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
]


class PoseLandmarks:
    """Wrapper around extracted landmarks providing easy access and validation."""
    
    def __init__(self, landmarks: List[Point], width: int, height: int):
        self.landmarks = landmarks
        self.width = width
        self.height = height

    def get(self, index: int | PoseLandmark) -> Point:
        """Get landmark by index."""
        idx = int(index)
        if 0 <= idx < len(self.landmarks):
            return self.landmarks[idx]
        return Point(0.0, 0.0, 0.0, 0.0)

    def get_pixel(self, index: int | PoseLandmark) -> Tuple[int, int]:
        """Get landmark coordinates in pixel space."""
        pt = self.get(index)
        return pt.to_pixel(self.width, self.height)

    def get_midpoint(self, idx1: int | PoseLandmark, idx2: int | PoseLandmark) -> Point:
        """Calculate midpoint between two landmarks."""
        p1 = self.get(idx1)
        p2 = self.get(idx2)
        return Point(
            x=(p1.x + p2.x) / 2.0,
            y=(p1.y + p2.y) / 2.0,
            z=(p1.z + p2.z) / 2.0,
            visibility=min(p1.visibility, p2.visibility)
        )

    def has_required(self, indices: List[int | PoseLandmark], min_visibility: float = 0.5) -> bool:
        """Check if all specified landmarks meet minimum visibility threshold."""
        for idx in indices:
            pt = self.get(idx)
            if pt.visibility < min_visibility:
                return False
        return True

    # Common anatomical properties
    @property
    def nose(self) -> Point: return self.get(PoseLandmark.NOSE)
    @property
    def left_shoulder(self) -> Point: return self.get(PoseLandmark.LEFT_SHOULDER)
    @property
    def right_shoulder(self) -> Point: return self.get(PoseLandmark.RIGHT_SHOULDER)
    @property
    def left_elbow(self) -> Point: return self.get(PoseLandmark.LEFT_ELBOW)
    @property
    def right_elbow(self) -> Point: return self.get(PoseLandmark.RIGHT_ELBOW)
    @property
    def left_wrist(self) -> Point: return self.get(PoseLandmark.LEFT_WRIST)
    @property
    def right_wrist(self) -> Point: return self.get(PoseLandmark.RIGHT_WRIST)
    @property
    def left_hip(self) -> Point: return self.get(PoseLandmark.LEFT_HIP)
    @property
    def right_hip(self) -> Point: return self.get(PoseLandmark.RIGHT_HIP)
    @property
    def left_knee(self) -> Point: return self.get(PoseLandmark.LEFT_KNEE)
    @property
    def right_knee(self) -> Point: return self.get(PoseLandmark.RIGHT_KNEE)
    @property
    def left_ankle(self) -> Point: return self.get(PoseLandmark.LEFT_ANKLE)
    @property
    def right_ankle(self) -> Point: return self.get(PoseLandmark.RIGHT_ANKLE)
    @property
    def mid_shoulder(self) -> Point: return self.get_midpoint(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)
    @property
    def mid_hip(self) -> Point: return self.get_midpoint(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)


class PoseDetector:
    """High-level pose detector managing MediaPipe Tasks inference."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if model_path is None:
            # Default to models/ directory relative to workspace root
            curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(curr_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pose_landmarker_full.task")

        self.model_path = model_path
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Download model if not present locally."""
        if not os.path.exists(self.model_path):
            print(f"Downloading pose model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

    def process(self, frame_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        """
        Process an OpenCV BGR frame and return detected landmarks.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Take first detected pose
        first_pose = result.pose_landmarks[0]
        points: List[Point] = []
        for lm in first_pose:
            # MediaPipe normalized landmarks: x [0,1], y [0,1], z depth, visibility [0,1]
            vis = getattr(lm, "visibility", 1.0)
            if vis is None:
                vis = 1.0
            points.append(Point(x=lm.x, y=lm.y, z=lm.z, visibility=vis))

        return PoseLandmarks(points, width=w, height=h)

    def close(self):
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
