# Real-Time Posture Monitoring for Effective Exercise Using MediaPipe Python

An advanced computer vision and biomechanical posture analysis system built with **MediaPipe Tasks**, **OpenCV**, and **Python**. It monitors exercise movements in real time, computes joint kinematics (angles, inclinations, alignment), counts repetitions using finite state machines, alerts the user to posture faults with instant visual & auditory coaching cues, and generates post-workout analytics reports.

---

## 🌟 Key Features

- **6 Monitored Exercises**:
  - 🏋️ **Squats**: Full depth checking ($\le 95^\circ$ knee bend), upright chest tracking ($\le 45^\circ$ forward lean), and knee valgus (caving inward) detection.
  - 💪 **Push-ups**: Chest depth ($\le 90^\circ$ elbow bend), straight body plank line ($160^\circ - 180^\circ$ hip angle), and prevention of lower back sagging or hip piking.
  - 🦾 **Bicep Curls**: Full range of motion ($>150^\circ$ extension to $<45^\circ$ contraction) and upper arm stability (flags cheating / elbow swinging).
  - 🧘 **Planks**: Isometric hold timer, spine alignment monitoring, and real-time tracking of good form vs. sagging/piking duration.
  - 🏋️‍♂️ **Overhead Shoulder Press**: Overhead arm lockout ($>165^\circ$), bottom depth, and lower back hyperextension / excessive arching prevention.
  - 🚶 **Lunges**: Front knee $90^\circ$ depth and upright torso posture tracking.
- **Biomechanical Kinematics Engine**:
  - True 2D and 3D joint angle vector trigonometry.
  - Gravity-referenced vertical and horizontal trunk/neck inclination tracking.
  - Exponential Moving Average (EMA) filtering to remove webcam jitter.
- **Real-Time Glassmorphic HUD Overlay**:
  - Live joint angle degree badges rendered directly over active joints.
  - Dynamic color-coded skeleton (Green = proper form, Amber = warning, Red = posture fault).
  - Repetition progress meter ($0\% - 100\%$) indicating depth and extension completeness.
  - High-contrast alert banners for immediate form correction cues.
- **Auditory Coaching**:
  - Asynchronous non-blocking audio chimes on rep completion and warning tones for bad posture.
- **Dual User Interface**:
  - **High-FPS Desktop Runner (`main.py`)**: Zero-latency OpenCV window with real-time hotkey exercise switching (`1`-`6`).
  - **Interactive Web Dashboard (`app.py`)**: Streamlit web app for video upload, live webcam streaming, angle progression charts, and downloadable session reports.
- **Comprehensive Session Analytics**:
  - Session duration, total reps, good reps vs. bad reps, form score percentage, and frequency breakdown of posture errors.
  - Automatically exports structured JSON session logs (`workout_summary.json`).

---

## 📁 Project Architecture

```
posture_monitoring/
├── .venv/                      # Python 3.13 Virtual Environment
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation and user guide
├── main.py                     # High-performance desktop OpenCV HUD runner
├── app.py                      # Interactive Streamlit Web Dashboard
├── models/
│   └── pose_landmarker_full.task  # Official MediaPipe Pose Landmarker model
├── core/
│   ├── geometry.py             # 2D/3D angle, inclination math & anti-jitter smoothing
│   └── pose_detector.py        # MediaPipe Tasks API wrapper with normalized landmarks
├── exercises/
│   ├── base_exercise.py        # Abstract base class, state machines & metrics
│   ├── squat.py                # Squat biomechanics & posture rules
│   ├── pushup.py               # Push-up biomechanics & plank alignment
│   ├── bicep_curl.py           # Bicep curl range of motion & elbow sway check
│   ├── plank.py                # Plank duration & hip alignment
│   ├── shoulder_press.py       # Overhead press & lumbar arching
│   ├── lunge.py                # Lunge depth & posture
│   └── __init__.py             # Exercise factory & shortcut registry
├── ui/
│   └── visualizer.py           # Glassmorphic HUD, angle badges, progress bars
├── audio/
│   └── feedback.py             # Asynchronous sound cues & alerts
├── analytics/
│   └── reporter.py             # Session logging, terminal summary & JSON export
├── utils/
│   └── synthetic_video.py      # Mathematically exact mock pose generator & demo videos
└── tests/
    ├── test_geometry.py        # Unit tests for kinematics mathematics
    └── test_exercises.py       # Unit tests for state machines & posture rules
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment

```powershell
# From project directory:
.venv\Scripts\activate
```

### 2. Run Desktop HUD (Live Webcam)

```powershell
python main.py --source 0 --exercise squat
```

### 3. Run with a Video File

```powershell
python main.py --source path\to\workout.mp4 --exercise pushup
```

### 4. Run Demonstration Mode (No Webcam Required)

```powershell
python main.py --demo
```

### 5. Launch Interactive Streamlit Web Dashboard

```powershell
streamlit run app.py
```

### 6. Run Automated Tests

```powershell
pytest tests -v
```

---

## ⌨️ In-Workout Hotkeys (Desktop Mode)

While running `main.py`, you can press the following keys at any time:

| Key | Action |
|:---:|:---|
| `1` | Switch exercise to **Squats** |
| `2` | Switch exercise to **Push-ups** |
| `3` | Switch exercise to **Bicep Curls** |
| `4` | Switch exercise to **Planks** |
| `5` | Switch exercise to **Overhead Shoulder Press** |
| `6` | Switch exercise to **Lunges** |
| `R` | **Reset** rep counters for the current exercise |
| `M` | **Mute / Unmute** audio feedback cues |
| `S` | Save a high-resolution **screenshot snapshot** |
| `Q` / `ESC` | **Quit** session and display detailed workout summary report |

---

## 📊 Sample Post-Workout Summary Report

```
=================================================================
          WORKOUT POSTURE MONITOR - SESSION SUMMARY          
=================================================================
 Date & Time    : 2026-09-22 23:25:10
 Total Duration : 12m 45s
 Total Reps     : 36 (Good: 31, Bad: 5)
 Form Accuracy  : 86.1%
-----------------------------------------------------------------

 Exercise: SQUAT
  - Completed Reps : 20
  - Correct Form   : 18
  - Posture Faults : 2
  - Accuracy Score : 90.0%
  - Posture Corrections Needed:
     * Squat deeper (aim for parallel): 2 time(s)

 Exercise: PUSH-UP
  - Completed Reps : 16
  - Correct Form   : 13
  - Posture Faults : 3
  - Accuracy Score : 81.3%
  - Posture Corrections Needed:
     * Hips sagging! Engage core and glutes: 3 time(s)
=================================================================
```
