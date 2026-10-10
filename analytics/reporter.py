"""
Workout Analytics and Session Reporter.
Tracks session telemetry, rep history, posture infraction frequencies,
and outputs clean summary reports in JSON and formatted text.
"""

from __future__ import annotations
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from exercises.base_exercise import BaseExercise


class WorkoutSessionReporter:
    """Manages workout session logs and creates performance summary reports."""

    def __init__(self):
        self.session_start = time.time()
        self.exercise_summaries: Dict[str, Dict[str, Any]] = {}
        self.angle_timeline: List[Dict[str, Any]] = []

    def record_exercise_state(self, exercise: BaseExercise):
        """Snapshots exercise metrics into session log."""
        fault_counts: Dict[str, int] = {}
        for r in exercise.history:
            for f in r.get("faults", []):
                fault_counts[f] = fault_counts.get(f, 0) + 1

        self.exercise_summaries[exercise.name] = {
            "total_reps": exercise.total_reps,
            "good_reps": exercise.good_reps,
            "bad_reps": exercise.bad_reps,
            "accuracy_percentage": exercise.accuracy,
            "rep_details": exercise.history,
            "fault_breakdown": fault_counts
        }

    def log_angle_telemetry(self, exercise_name: str, angles: Dict[str, float]):
        """Logs angle time-series data for charting."""
        self.angle_timeline.append({
            "timestamp": round(time.time() - self.session_start, 2),
            "exercise": exercise_name,
            **angles
        })

    def generate_summary(self) -> Dict[str, Any]:
        """Compile complete session statistics."""
        duration = round(time.time() - self.session_start, 1)
        total_all_reps = sum(s["total_reps"] for s in self.exercise_summaries.values())
        total_good_reps = sum(s["good_reps"] for s in self.exercise_summaries.values())
        overall_acc = round((total_good_reps / total_all_reps * 100.0), 1) if total_all_reps > 0 else 100.0

        return {
            "session_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration,
            "total_reps": total_all_reps,
            "total_good_reps": total_good_reps,
            "total_bad_reps": total_all_reps - total_good_reps,
            "overall_accuracy_percentage": overall_acc,
            "exercises": self.exercise_summaries,
        }

    def export_json(self, output_path: str = "workout_summary.json") -> str:
        """Export session metrics to JSON file."""
        data = self.generate_summary()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return os.path.abspath(output_path)

    def print_terminal_report(self):
        """Print an attractive summary report card to stdout."""
        summary = self.generate_summary()
        dur_min = int(summary["duration_seconds"] // 60)
        dur_sec = int(summary["duration_seconds"] % 60)

        print("\n" + "=" * 65)
        print("          WORKOUT POSTURE MONITOR - SESSION SUMMARY          ")
        print("=" * 65)
        print(f" Date & Time    : {summary['session_date']}")
        print(f" Total Duration : {dur_min}m {dur_sec}s")
        print(f" Total Reps     : {summary['total_reps']} (Good: {summary['total_good_reps']}, Bad: {summary['total_bad_reps']})")
        print(f" Form Accuracy  : {summary['overall_accuracy_percentage']}%")
        print("-" * 65)

        for name, data in summary["exercises"].items():
            print(f"\n Exercise: {name.upper()}")
            print(f"  - Completed Reps : {data['total_reps']}")
            print(f"  - Correct Form   : {data['good_reps']}")
            print(f"  - Posture Faults : {data['bad_reps']}")
            print(f"  - Accuracy Score : {data['accuracy_percentage']}%")

            if data["fault_breakdown"]:
                print("  - Posture Corrections Needed:")
                for fault, count in data["fault_breakdown"].items():
                    print(f"     * {fault}: {count} time(s)")
            else:
                print("  - Perfect technique! Zero posture violations detected.")

        print("=" * 65 + "\n")
