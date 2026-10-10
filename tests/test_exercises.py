"""
Unit tests for exercise state machines, repetition counting, and posture evaluation.
"""

import pytest
from exercises import get_exercise, list_exercises
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.bicep_curl import BicepCurlExercise
from exercises.plank import PlankExercise
from utils.synthetic_video import MockPoseGenerator


def test_registry_and_shortcuts():
    assert "squat" in list_exercises()
    assert "pushup" in list_exercises()
    assert "curl" in list_exercises()
    assert "plank" in list_exercises()

    sq = get_exercise("1")
    assert isinstance(sq, SquatExercise)
    pu = get_exercise("pushup")
    assert isinstance(pu, PushupExercise)

    with pytest.raises(ValueError):
        get_exercise("invalid_exercise_name")


def test_squat_good_rep():
    squat = SquatExercise()
    # 60 frames of a good squat (reaches 85 deg knee depth, 15 deg torso lean)
    frames = MockPoseGenerator.generate_squat_cycle(num_frames=60, lowest_knee_angle=85.0, torso_lean=15.0)

    for f in frames:
        metrics = squat.evaluate(f)

    assert squat.total_reps == 1
    assert squat.good_reps == 1
    assert squat.bad_reps == 0
    assert squat.accuracy == 100.0


def test_squat_shallow_rep():
    squat = SquatExercise()
    # Shallow squat (only reaches 125 deg)
    frames = MockPoseGenerator.generate_squat_cycle(num_frames=60, lowest_knee_angle=125.0, torso_lean=15.0)

    for f in frames:
        squat.evaluate(f)

    assert squat.total_reps == 1
    assert squat.good_reps == 0
    assert squat.bad_reps == 1
    assert "Squat deeper" in squat.history[0]["faults"][0]


def test_squat_excessive_forward_lean():
    squat = SquatExercise()
    # Squat with 55 degree forward lean (fails torso lean rule > 45 deg)
    frames = MockPoseGenerator.generate_squat_cycle(num_frames=60, lowest_knee_angle=85.0, torso_lean=55.0)

    for f in frames:
        squat.evaluate(f)

    assert squat.total_reps == 1
    assert squat.bad_reps == 1
    assert any("chest" in fault.lower() for fault in squat.history[0]["faults"])


def test_pushup_good_rep():
    pushup = PushupExercise()
    frames = MockPoseGenerator.generate_pushup_cycle(num_frames=60, lowest_elbow=80.0, hip_angle=175.0)

    for f in frames:
        pushup.evaluate(f)

    assert pushup.total_reps == 1
    assert pushup.good_reps == 1
    assert pushup.bad_reps == 0


def test_pushup_hip_sagging():
    pushup = PushupExercise()
    # Sagging hips (hip angle 140 deg < 155 deg threshold)
    frames = MockPoseGenerator.generate_pushup_cycle(num_frames=60, lowest_elbow=80.0, hip_angle=140.0)

    for f in frames:
        pushup.evaluate(f)

    assert pushup.total_reps == 1
    assert pushup.bad_reps == 1
    assert any("sagging" in fault.lower() for fault in pushup.history[0]["faults"])


def test_bicep_curl_good_and_swing():
    curl = BicepCurlExercise()
    good_frames = MockPoseGenerator.generate_bicep_cycle(num_frames=60, min_elbow=38.0, swing=5.0)
    for f in good_frames:
        curl.evaluate(f)

    assert curl.total_reps == 1
    assert curl.good_reps == 1

    # Now do a second rep with excessive elbow swinging (35 deg swing > 25 deg limit)
    swing_frames = MockPoseGenerator.generate_bicep_cycle(num_frames=60, min_elbow=38.0, swing=35.0)
    for f in swing_frames:
        curl.evaluate(f)

    assert curl.total_reps == 2
    assert curl.good_reps == 1
    assert curl.bad_reps == 1
    assert any("swung" in fault.lower() for fault in curl.history[1]["faults"])
