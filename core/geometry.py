"""
Kinematics and Geometry Engine for Posture Monitoring.
Computes 2D/3D joint angles, inclinations, Euclidean distances,
and provides anti-jitter smoothing filters.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


@dataclass
class Point:
    """Represents a 2D or 3D landmark with optional visibility score."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple_2d(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_tuple_3d(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized (0-1) coordinates to integer pixel coordinates."""
        return (int(self.x * width), int(self.y * height))


def calculate_angle_2d(a: Point, b: Point, c: Point) -> float:
    """
    Calculate the interior angle ABC at vertex B in degrees [0, 180].
    
    Args:
        a: First endpoint
        b: Vertex point
        c: Second endpoint
        
    Returns:
        Angle in degrees between [0.0, 180.0].
    """
    ba_x = a.x - b.x
    ba_y = a.y - b.y
    bc_x = c.x - b.x
    bc_y = c.y - b.y

    dot_product = ba_x * bc_x + ba_y * bc_y
    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)

    if mag_ba * mag_bc == 0:
        return 0.0

    cosine = dot_product / (mag_ba * mag_bc)
    # Clamp to [-1.0, 1.0] to prevent domain errors with float inaccuracies
    cosine = max(-1.0, min(1.0, cosine))
    angle_rad = math.acos(cosine)
    return math.degrees(angle_rad)


def calculate_angle_3d(a: Point, b: Point, c: Point) -> float:
    """
    Calculate the 3D angle ABC at vertex B in degrees [0, 180].
    """
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=float)
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=float)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba * norm_bc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def calculate_inclination(a: Point, b: Point, reference: str = "vertical") -> float:
    """
    Calculate inclination angle of vector A -> B relative to vertical (up/down)
    or horizontal (left/right) reference axis.
    In image coordinates, y increases downward.
    Vertical vector pointing straight up is (0, -1).
    Horizontal vector pointing right is (1, 0).
    
    Args:
        a: Start point (e.g. Hip)
        b: End point (e.g. Shoulder)
        reference: 'vertical' or 'horizontal'
        
    Returns:
        Angle in degrees between 0 and 180.
    """
    dx = b.x - a.x
    dy = b.y - a.y
    mag = math.hypot(dx, dy)
    if mag == 0:
        return 0.0

    if reference == "vertical":
        # Straight up vector is (0, -1)
        # dot = dx*0 + dy*(-1) = -dy
        cosine = (-dy) / mag
    else:  # horizontal
        # Straight right vector is (1, 0)
        cosine = dx / mag

    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def calculate_distance_2d(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(b.x - a.x, b.y - a.y)


def calculate_distance_3d(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)


class ExponentialMovingAverageFilter:
    """
    Exponential Moving Average (EMA) filter to smooth noisy keypoint coordinates
    and joint angles across successive video frames.
    """
    def __init__(self, alpha: float = 0.35):
        """
        Args:
            alpha: Smoothing factor in (0, 1].
                   Higher alpha = more responsive, less smoothing.
                   Lower alpha = smoother, slight lag.
        """
        self.alpha = alpha
        self.values: Dict[str, float] = {}

    def filter(self, key: str, value: float) -> float:
        """Filter a scalar value (e.g. joint angle)."""
        if key not in self.values:
            self.values[key] = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self.values[key]
        self.values[key] = filtered
        return filtered

    def reset(self):
        self.values.clear()
