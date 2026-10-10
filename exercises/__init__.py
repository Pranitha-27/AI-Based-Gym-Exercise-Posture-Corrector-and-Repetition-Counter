"""
Exercise package registry and factory.
"""

from typing import Dict, Type, List
from exercises.base_exercise import BaseExercise, ExerciseMetrics
from exercises.squat import SquatExercise
from exercises.pushup import PushupExercise
from exercises.bicep_curl import BicepCurlExercise
from exercises.plank import PlankExercise
from exercises.shoulder_press import ShoulderPressExercise
from exercises.lunge import LungeExercise

EXERCISE_REGISTRY: Dict[str, Type[BaseExercise]] = {
    "squat": SquatExercise,
    "pushup": PushupExercise,
    "curl": BicepCurlExercise,
    "plank": PlankExercise,
    "press": ShoulderPressExercise,
    "lunge": LungeExercise,
}

EXERCISE_SHORTCUTS: Dict[str, str] = {
    "1": "squat",
    "2": "pushup",
    "3": "curl",
    "4": "plank",
    "5": "press",
    "6": "lunge",
}

def get_exercise(name_or_key: str) -> BaseExercise:
    """Retrieve an exercise monitor instance by name or shortcut key."""
    key = str(name_or_key).lower().strip()
    if key in EXERCISE_SHORTCUTS:
        key = EXERCISE_SHORTCUTS[key]
    if key not in EXERCISE_REGISTRY:
        valid = ", ".join(list(EXERCISE_REGISTRY.keys()))
        raise ValueError(f"Unknown exercise '{name_or_key}'. Available exercises: {valid}")
    return EXERCISE_REGISTRY[key]()

def list_exercises() -> List[str]:
    """Return list of supported exercise names."""
    return list(EXERCISE_REGISTRY.keys())
