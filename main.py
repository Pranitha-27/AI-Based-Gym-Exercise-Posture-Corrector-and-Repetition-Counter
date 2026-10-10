"""
Main Desktop Application for Real-Time Exercise Posture Monitoring.
Supports webcam feeds, recorded videos, exercise-specific synthetic demo videos,
real-time HUD overlays, audio coaching, hotkey exercise switching, and automatic workout analytics reporting.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np

from core.pose_detector import PoseDetector, PoseLandmarks, PoseLandmark
from exercises import get_exercise, list_exercises, EXERCISE_SHORTCUTS
from ui.visualizer import Visualizer
from audio.feedback import AudioFeedback
from analytics.reporter import WorkoutSessionReporter
from utils.synthetic_video import create_demo_video, MockPoseGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Posture Monitoring for Effective Exercise Using MediaPipe"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video source: webcam index ('0', '1') or path to video file (.mp4, .mov, etc.)"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="squat",
        choices=list_exercises(),
        help="Initial exercise to monitor (default: squat)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate and run exercise-specific synthetic demonstration video"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable sound cues and audio alerts"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save annotated output video (e.g. output.mp4)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / batch processing)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence (0.0 to 1.0, default: 0.5)"
    )
    return parser.parse_args()


def open_video_source(source_str: str, is_demo: bool = False, exercise_name: str = "squat") -> cv2.VideoCapture:
    """Initialize OpenCV VideoCapture from webcam index, file, or exercise-specific demo generator."""
    if is_demo or source_str.lower() == "demo":
        ex_name = exercise_name.lower().strip()
        demo_path = os.path.abspath(f"demo_{ex_name}.mp4")
        if not os.path.exists(demo_path):
            print(f"Generating synthetic demo video for {ex_name.upper()}...")
            create_demo_video(demo_path, exercise_name=ex_name, duration_sec=8, fps=30)
        source_str = demo_path

    # Check if integer webcam index
    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str), cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    else:
        if not os.path.exists(source_str):
            print(f"Error: Video file not found: {source_str}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(source_str)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {source_str}", file=sys.stderr)
        sys.exit(1)

    return cap


def main():
    args = parse_args()

    print("=" * 65)
    print("      AI REAL-TIME POSTURE MONITOR & WORKOUT COACH           ")
    print("=" * 65)
    print(f" Initial Exercise : {args.exercise.upper()}")
    print(f" Input Source     : {args.source} {'(Demo Mode)' if args.demo else ''}")
    print(f" Audio Feedback   : {'Disabled' if args.no_audio else 'Enabled'}")
    print(" Hotkeys:")
    print("   [1-6] Switch Exercise (1:Squat, 2:Pushup, 3:Curl, 4:Plank, 5:Press, 6:Lunge)")
    print("   [R]   Reset Counters for Current Exercise")
    print("   [M]   Toggle Audio Feedback Mute")
    print("   [S]   Save Screenshot Snapshot")
    print("   [Q]   Quit Session and View Full Workout Report")
    print("=" * 65 + "\n")

    current_exercise_name = args.exercise
    cap = open_video_source(args.source, is_demo=args.demo, exercise_name=current_exercise_name)

    # Initialize subsystems
    pose_detector = PoseDetector(min_detection_confidence=args.confidence)
    visualizer = Visualizer()
    audio = AudioFeedback(enabled=not args.no_audio)
    reporter = WorkoutSessionReporter()

    # Active exercise tracker
    current_exercise = get_exercise(current_exercise_name)
    is_demo_mode = args.demo or "demo" in str(args.source).lower()
    demo_landmarks_list = MockPoseGenerator.get_cycle_for_exercise(current_exercise_name, num_frames=60) if is_demo_mode else []

    # Video Recorder setup
    video_writer = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.record, fourcc, fps_in, (width, height))
        print(f"Recording session output to: {args.record}")

    prev_time = time.time()
    prev_rep_count = 0
    frame_count = 0

    window_name = "AI Real-Time Posture Monitor"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_demo_mode and not args.headless:
                    # Seamlessly loop demo video for continuous evaluation in GUI mode
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            frame_count += 1
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-4)
            prev_time = curr_time

            # 1. Pose Inference
            if is_demo_mode and len(demo_landmarks_list) > 0:
                landmarks = demo_landmarks_list[(frame_count - 1) % len(demo_landmarks_list)]
            else:
                landmarks = pose_detector.process(frame)

            if landmarks is not None:
                # 2. Evaluate Biomechanical Form
                metrics = current_exercise.evaluate(landmarks)

                # 3. Audio Triggers
                if metrics.total_reps > prev_rep_count:
                    # New rep completed
                    is_last_rep_good = (metrics.good_reps > current_exercise.good_reps - 1)
                    audio.play_rep_completed(is_good=is_last_rep_good)
                    prev_rep_count = metrics.total_reps

                if metrics.feedback_severity == "ALERT":
                    audio.play_posture_alert()

                # 4. Draw Skeleton with Dynamic Posture Color Coding
                visualizer.draw_skeleton(frame, landmarks, metrics)

                # 5. Draw Primary Joint Angle Badges
                if "knee_angle" in metrics.current_angles:
                    knee_pt = landmarks.left_knee if metrics.side_view != "RIGHT" else landmarks.right_knee
                    is_safe = metrics.current_angles["knee_angle"] >= 80.0
                    visualizer.draw_angle_badge(frame, knee_pt, metrics.current_angles["knee_angle"], label="Knee", is_safe=is_safe)

                if "front_knee" in metrics.current_angles:
                    knee_pt = landmarks.left_knee
                    visualizer.draw_angle_badge(frame, knee_pt, metrics.current_angles["front_knee"], label="Front Knee")

                if "elbow_angle" in metrics.current_angles:
                    elbow_pt = landmarks.left_elbow if metrics.side_view != "RIGHT" else landmarks.right_elbow
                    visualizer.draw_angle_badge(frame, elbow_pt, metrics.current_angles["elbow_angle"], label="Elbow")

                if "hip_angle" in metrics.current_angles:
                    hip_pt = landmarks.left_hip if metrics.side_view != "RIGHT" else landmarks.right_hip
                    visualizer.draw_angle_badge(frame, hip_pt, metrics.current_angles["hip_angle"], label="Spine")

                # Log telemetry
                reporter.log_angle_telemetry(metrics.exercise_name, metrics.current_angles)
            else:
                # No person detected in frame
                metrics = current_exercise.evaluate(
                    landmarks=PoseLandmarks([], width, height)
                )
                metrics.active_feedback = ["No person detected - step into camera frame"]
                metrics.feedback_severity = "WARNING"

            # 6. Render Glassmorphic HUD Header & Bottom Alert Banner
            frame = visualizer.draw_hud(frame, metrics, fps, audio_enabled=audio.enabled)

            # Record frame if active
            if video_writer is not None:
                video_writer.write(frame)

            # Display GUI window
            if not args.headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):
                    current_exercise.reset()
                    prev_rep_count = 0
                    print(f"Counters reset for {current_exercise.name}.")
                elif key == ord('m'):
                    muted = not audio.toggle()
                    print(f"Audio {'Muted' if muted else 'Unmuted'}.")
                elif key == ord('s'):
                    snap_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snap_name, frame)
                    print(f"Saved snapshot to {snap_name}")
                elif chr(key) in EXERCISE_SHORTCUTS:
                    # Switch exercise hotkey (1-6)
                    reporter.record_exercise_state(current_exercise)
                    new_ex_name = EXERCISE_SHORTCUTS[chr(key)]
                    current_exercise = get_exercise(new_ex_name)
                    prev_rep_count = 0
                    current_exercise_name = new_ex_name

                    # If in demo mode, reload the corresponding demo video!
                    if is_demo_mode:
                        cap.release()
                        cap = open_video_source(args.source, is_demo=True, exercise_name=new_ex_name)
                        demo_landmarks_list = MockPoseGenerator.get_cycle_for_exercise(new_ex_name, num_frames=60)
                        frame_count = 0

                    print(f"\nSwitched exercise to: {current_exercise.name.upper()}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        # Snapshot current exercise
        reporter.record_exercise_state(current_exercise)

        # Cleanup video streams
        cap.release()
        if video_writer is not None:
            video_writer.release()
        pose_detector.close()
        cv2.destroyAllWindows()

        # Generate & export workout report
        json_file = f"workout_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        saved_path = reporter.export_json(json_file)
        reporter.print_terminal_report()
        print(f"Detailed workout telemetry saved to: {saved_path}")


if __name__ == "__main__":
    main()
