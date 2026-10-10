"""
Interactive Streamlit Web Dashboard for Real-Time Exercise Posture Monitoring.
Allows video uploads, exercise-specific demo workouts, live webcam coaching,
telemetry charting, rep analysis, and report exports.
"""

from __future__ import annotations
import os
import tempfile
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.pose_detector import PoseDetector, PoseLandmarks
from exercises import get_exercise, list_exercises
from ui.visualizer import Visualizer
from analytics.reporter import WorkoutSessionReporter
from utils.synthetic_video import create_demo_video, MockPoseGenerator


st.set_page_config(
    page_title="AI Exercise Posture Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222b;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00c0f2;
    }
    .status-good {
        color: #46c846;
        font-weight: bold;
    }
    .status-alert {
        color: #f03232;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🏋️ AI Real-Time Posture Monitor & Exercise Coach")
    st.caption("Powered by MediaPipe Pose Kinematics and Computer Vision")

    # Sidebar Controls
    st.sidebar.header("⚙️ Workout Configuration")
    exercise_name = st.sidebar.selectbox(
        "Select Exercise",
        options=list_exercises(),
        index=0,
        format_func=lambda x: x.replace("_", " ").title()
    )

    # Exercise Instructions Guide in Sidebar
    EXERCISE_GUIDES = {
        "squat": "**Ideal Form:**\n- Thighs parallel to floor (<= 95° knee bend)\n- Chest upright, spine straight (<= 45° torso tilt)\n- Knees tracking over toes (no inward cave)",
        "pushup": "**Ideal Form:**\n- Chest to floor (elbows <= 90°)\n- Straight body plank line (160° - 180° hip angle)\n- No lower back sagging or piking",
        "curl": "**Ideal Form:**\n- Full extension (> 150°) to peak contraction (< 45°)\n- Elbows pinned to ribs (no upper arm swinging)",
        "plank": "**Ideal Form:**\n- Perfectly straight spine line from shoulders to ankles (160° - 180°)\n- Engage core and glutes",
        "press": "**Ideal Form:**\n- Full overhead lockout (> 165°)\n- Prevent excessive lumbar arching",
        "lunge": "**Ideal Form:**\n- Front knee ~90° angle\n- Torso tall and upright"
    }
    st.sidebar.info(EXERCISE_GUIDES.get(exercise_name, ""))

    input_mode = st.sidebar.radio(
        "Video Source",
        options=["Generate Demo Workout", "Upload Video File", "Live Webcam"],
        index=0
    )

    confidence = st.sidebar.slider("Pose Detection Confidence", 0.3, 0.9, 0.5, 0.05)

    video_path = None
    temp_file = None

    if input_mode == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader("Upload workout video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            video_path = temp_file.name
    elif input_mode == "Generate Demo Workout":
        demo_path = os.path.abspath(f"demo_{exercise_name}.mp4")
        if not os.path.exists(demo_path):
            with st.spinner(f"Generating synthetic demo video for {exercise_name.title()}..."):
                create_demo_video(demo_path, exercise_name=exercise_name, duration_sec=6, fps=30)
        video_path = demo_path
    else:  # Live Webcam
        cam_index = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0)
        video_path = int(cam_index)

    # Main Dashboard Layout
    col_video, col_stats = st.columns([3, 2])

    with col_stats:
        st.subheader("📊 Live Telemetry")
        m_col1, m_col2 = st.columns(2)
        total_reps_card = m_col1.metric("Total Reps", 0)
        form_score_card = m_col2.metric("Form Score", "100%")

        m_col3, m_col4 = st.columns(2)
        good_reps_card = m_col3.metric("Good Reps", 0)
        bad_reps_card = m_col4.metric("Bad Reps", 0)

        stage_placeholder = st.empty()
        feedback_placeholder = st.empty()
        chart_placeholder = st.empty()

    with col_video:
        video_placeholder = st.empty()
        start_btn = st.button("▶️ Start Posture Monitoring", type="primary")

    if start_btn and video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video source.")
            return

        detector = PoseDetector(min_detection_confidence=confidence)
        visualizer = Visualizer()
        exercise = get_exercise(exercise_name)
        reporter = WorkoutSessionReporter()

        is_demo = (input_mode == "Generate Demo Workout")
        demo_landmarks_list = MockPoseGenerator.get_cycle_for_exercise(exercise_name, num_frames=60) if is_demo else []

        angle_history = []
        timestamps = []
        fps_val = 30.0
        prev_t = time.time()
        frame_idx = 0

        stop_btn = st.sidebar.button("⏹️ Stop Workout")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                if is_demo:
                    # Loop demo video smoothly for continuous demonstration
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            frame_idx += 1
            now_t = time.time()
            fps_val = 1.0 / max(now_t - prev_t, 1e-4)
            prev_t = now_t

            if is_demo and len(demo_landmarks_list) > 0:
                landmarks = demo_landmarks_list[(frame_idx - 1) % len(demo_landmarks_list)]
            else:
                landmarks = detector.process(frame)

            if landmarks is not None:
                metrics = exercise.evaluate(landmarks)
                visualizer.draw_skeleton(frame, landmarks, metrics)

                # Primary angles
                if "knee_angle" in metrics.current_angles:
                    knee_pt = landmarks.left_knee if metrics.side_view != "RIGHT" else landmarks.right_knee
                    visualizer.draw_angle_badge(frame, knee_pt, metrics.current_angles["knee_angle"], label="Knee")
                    angle_history.append(metrics.current_angles["knee_angle"])
                    timestamps.append(len(timestamps) / fps_val)

                if "front_knee" in metrics.current_angles:
                    knee_pt = landmarks.left_knee
                    visualizer.draw_angle_badge(frame, knee_pt, metrics.current_angles["front_knee"], label="Front Knee")
                    angle_history.append(metrics.current_angles["front_knee"])
                    timestamps.append(len(timestamps) / fps_val)

                if "elbow_angle" in metrics.current_angles:
                    elbow_pt = landmarks.left_elbow if metrics.side_view != "RIGHT" else landmarks.right_elbow
                    visualizer.draw_angle_badge(frame, elbow_pt, metrics.current_angles["elbow_angle"], label="Elbow")
                    angle_history.append(metrics.current_angles["elbow_angle"])
                    timestamps.append(len(timestamps) / fps_val)

                if "hip_angle" in metrics.current_angles:
                    hip_pt = landmarks.left_hip if metrics.side_view != "RIGHT" else landmarks.right_hip
                    visualizer.draw_angle_badge(frame, hip_pt, metrics.current_angles["hip_angle"], label="Spine")
                    angle_history.append(metrics.current_angles["hip_angle"])
                    timestamps.append(len(timestamps) / fps_val)

                reporter.log_angle_telemetry(metrics.exercise_name, metrics.current_angles)
            else:
                metrics = exercise.evaluate(PoseLandmarks([], frame.shape[1], frame.shape[0]))
                metrics.active_feedback = ["No person detected"]

            frame = visualizer.draw_hud(frame, metrics, fps_val, audio_enabled=False)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Metrics Cards
            total_reps_card.metric("Total Reps", metrics.total_reps)
            good_reps_card.metric("Good Reps", metrics.good_reps)
            bad_reps_card.metric("Bad Reps", metrics.bad_reps)
            form_score_card.metric("Form Score", f"{metrics.accuracy_percentage}%")

            stage_placeholder.info(f"**Stage:** `{metrics.stage}`")

            # Feedback banner
            if metrics.feedback_severity == "ALERT":
                feedback_placeholder.error(f"⚠️ **ALERT:** {' | '.join(metrics.active_feedback)}")
            elif metrics.feedback_severity == "WARNING":
                feedback_placeholder.warning(f"⚡ **CAUTION:** {' | '.join(metrics.active_feedback)}")
            else:
                feedback_placeholder.success(f"✅ **GOOD FORM:** {' | '.join(metrics.active_feedback) if metrics.active_feedback else 'Great technique!'}")

            # Plot Angle Telemetry
            if len(angle_history) > 10:
                recent_angles = angle_history[-80:]
                chart_df = pd.DataFrame({"Joint Angle (deg)": recent_angles})
                chart_placeholder.line_chart(chart_df, height=180)

        cap.release()
        detector.close()
        reporter.record_exercise_state(exercise)

        st.success("🎉 Workout session completed!")

        # Rep Breakdown Table
        if exercise.history:
            st.subheader("📋 Repetition Breakdown")
            rep_df = pd.DataFrame(exercise.history)
            st.dataframe(rep_df, use_container_width=True)

        # Download Report
        summary_json = reporter.export_json("web_workout_summary.json")
        with open(summary_json, "r") as f:
            st.download_button(
                label="📥 Download Complete Workout Report (JSON)",
                data=f.read(),
                file_name=f"workout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    # Cleanup temp file
    if temp_file is not None and os.path.exists(temp_file.name):
        try:
            os.remove(temp_file.name)
        except Exception:
            pass


if __name__ == "__main__":
    main()
