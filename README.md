# Physio-Vision: Enterprise Biomechanical Analysis Engine

## Project Overview

Physio-Vision is a high-performance computer vision application designed to analyze human movement mechanics in real time. Unlike traditional fitness trackers that rely solely on 2D pose estimation or opaque "black box" AI classification, this system employs a hybrid neuro-symbolic architecture.

It combines MediaPipe's neural network-based skeletal perception with a strict 3D vector calculus engine for deterministic geometric safety enforcement. The current build is optimized for the deep squat movement and is capable of detecting subtle biomechanical faults such as thoracic rounding, excessive trunk lean, and valgus collapse using a standard webcam.

The application also features a procedural augmented reality positioning system to standardize subject placement and a multi-threaded GUI for lag-free visualization. Session data is synchronized with a FastAPI backend that stores user accounts and training history in a local SQLite database.

## Key Features

### 3D Depth Inference

* Utilizes MediaPipe world landmarks to calculate true 3D joint angles.
* Mitigates perspective errors caused by subject rotation or camera distance.

### Holographic AR Guidance

* Implements a procedural "reactor ring" targeting system.
* Projects perspective-corrected floor guides to standardize subject distance and orientation.

### Biomechanical Fault Detection

* Thoracic kyphosis proxy: detects upper back rounding by analyzing 3D shoulder protraction vectors.
* Trunk flexion monitor: measures spinal lean relative to gravity, dynamically calibrated for varying user anthropometry.
* Kinematic state machine: uses angular thresholds to accurately track concentric and eccentric movement phases.

### Live Engineering Console

* Floating debugging interface that allows real-time adjustment of sensitivity thresholds during active sessions.

### Longitudinal Telemetry

* Automatically serializes session metrics into structured patient history logs.
* Tracks repetition counts and form scores across sessions with optional synchronization via the backend.

## System Architecture

The engine operates on a concurrent, two-layer pipeline.

### Perception Layer (Neural)

* A dedicated Vision Worker thread manages camera I/O.
* Runs MediaPipe inference at 30+ FPS.
* Outputs normalized 3D skeletal landmark data.

### Logic Layer (Symbolic)

* A parallel vector physics engine evaluates the live skeleton against deterministic safety constraints.
* Triggers asynchronous audio feedback without blocking the vision pipeline.

### Backend

* FastAPI server (`api_server.py`) with SQLite database (`physio.db`).
* Supports user registration, login, session logging, and history retrieval.

## Installation

### Prerequisites

* Python 3.10 or 3.11
* Webcam

### Dependencies

Install required packages using pip. NumPy version is strictly constrained to maintain compatibility.

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install opencv-contrib-python mediapipe PyQt5 pyttsx3 qfluentwidgets fastapi uvicorn
pip install "numpy<2.0.0" --force-reinstall
```

## Backend Setup

The application requires the FastAPI backend to be running for authentication and session storage.

Place `api_server.py` and `physio.db` in the project root.

Start the server:

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at http://127.0.0.1:8000 (or your network IP if running on a remote machine such as a Raspberry Pi).

Note: The provided `physio.db` contains the initial database schema and will be used automatically by the server.

## Client Configuration and Usage

### Environment Variable Setup (Required)

The client needs the backend URL. Set the `API_URL` environment variable before running the application.

#### In PyCharm (Recommended)

Open Run → Edit Configurations for `test.py`.

In User environment variables, click "+" and add:

* Name: API_URL
* Value: http://127.0.0.1:8000/   (or your server's actual address, ending with /)

Apply and save.

#### From Command Prompt (Windows)

```cmd
set API_URL=http://127.0.0.1:8000/
python test.py
```

#### From PowerShell

```powershell
$env:API_URL = "http://127.0.0.1:8000/"
python test.py
```

Update the value if your backend runs on a different IP or port.

## Guide Video Configuration

In `test.py`, ensure the guide video path in the `AppState` class points to your reference file:

```python
GUIDE_PATH = "Video_Generation_Person_Squatting.mp4"
```

## Execution

Run the client:

```bash
python test.py
```

The login window will appear. Register a new account or log in. The system will then allow live analysis sessions.

## Operation Flow

* Backend: Ensure `api_server.py` is running.
* AR Alignment: Enable Holographic Guidance in Settings. Align feet with the projected floor target until the status indicator shows "TARGET LOCKED".
* Calibration: The system performs a brief static analysis.
* Session: Perform squats. The engine tracks repetitions, evaluates form, and provides real-time audio cues.
* Review: Navigate to Patient Records to view session reports. Data is automatically saved to the backend.

## Current Limitations

* Optimized for side-profile view. Front-facing or oblique angles may reduce accuracy.
* External loads (e.g., barbells) are not yet modeled.
* Lighting sensitivity may introduce depth jitter in extreme conditions.

## Disclaimer

This software is provided for educational and research purposes only. It is not a medical device. Users should consult a qualified healthcare professional before beginning any new exercise program. The developers assume no responsibility for injuries sustained while using this software.
