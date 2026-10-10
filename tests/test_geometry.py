"""
Unit tests for geometric and trigonometric kinematics calculations.
"""

import math
import pytest
from core.geometry import (
    Point,
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_inclination,
    calculate_distance_2d,
    calculate_distance_3d,
    ExponentialMovingAverageFilter,
)


def test_calculate_angle_2d_right_angle():
    a = Point(0.0, 1.0)
    b = Point(0.0, 0.0)
    c = Point(1.0, 0.0)
    angle = calculate_angle_2d(a, b, c)
    assert pytest.approx(angle, 0.01) == 90.0


def test_calculate_angle_2d_straight_line():
    a = Point(-1.0, 0.0)
    b = Point(0.0, 0.0)
    c = Point(1.0, 0.0)
    angle = calculate_angle_2d(a, b, c)
    assert pytest.approx(angle, 0.01) == 180.0


def test_calculate_angle_2d_acute_and_collinear():
    a = Point(1.0, 1.0)
    b = Point(0.0, 0.0)
    c = Point(1.0, 0.0)
    angle = calculate_angle_2d(a, b, c)
    assert pytest.approx(angle, 0.01) == 45.0

    # Overlapping rays = 0 degrees
    angle_zero = calculate_angle_2d(Point(1.0, 0.0), Point(0.0, 0.0), Point(2.0, 0.0))
    assert pytest.approx(angle_zero, 0.01) == 0.0


def test_calculate_angle_3d():
    a = Point(0.0, 0.0, 1.0)
    b = Point(0.0, 0.0, 0.0)
    c = Point(1.0, 0.0, 0.0)
    angle = calculate_angle_3d(a, b, c)
    assert pytest.approx(angle, 0.01) == 90.0


def test_calculate_inclination():
    # Straight vertical up vector (y decreases upwards in image space)
    hip = Point(0.5, 0.6)
    shoulder_up = Point(0.5, 0.3)
    inc_vertical = calculate_inclination(hip, shoulder_up, reference="vertical")
    assert pytest.approx(inc_vertical, 0.01) == 0.0

    # 45 degree tilt
    shoulder_tilt = Point(0.8, 0.3)
    inc_tilt = calculate_inclination(hip, shoulder_tilt, reference="vertical")
    assert pytest.approx(inc_tilt, 1.0) == 45.0

    # Horizontal vector
    shoulder_horiz = Point(0.9, 0.6)
    inc_horiz = calculate_inclination(hip, shoulder_horiz, reference="horizontal")
    assert pytest.approx(inc_horiz, 0.01) == 0.0


def test_calculate_distances():
    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)
    assert pytest.approx(calculate_distance_2d(p1, p2), 0.001) == 5.0

    p3 = Point(0.0, 0.0, 0.0)
    p4 = Point(1.0, 2.0, 2.0)
    assert pytest.approx(calculate_distance_3d(p3, p4), 0.001) == 3.0


def test_exponential_moving_average():
    ema = ExponentialMovingAverageFilter(alpha=0.5)
    assert ema.filter("test", 100.0) == 100.0
    # Next value 50.0 -> 0.5 * 50 + 0.5 * 100 = 75.0
    assert pytest.approx(ema.filter("test", 50.0), 0.01) == 75.0
    # Next value 50.0 -> 0.5 * 50 + 0.5 * 75 = 62.5
    assert pytest.approx(ema.filter("test", 50.0), 0.01) == 62.5
