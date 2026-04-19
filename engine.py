

import cv2
import time
import threading
import numpy as np
import pyttsx3
import math
import os
from datetime import datetime
from keras.models import load_model

# --- Environment flags must be set BEFORE any AI/GPU imports ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from PyQt5.QtCore import QThread, pyqtSignal as Signal
from PyQt5.QtGui import QImage

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


# =============================================================================
#  GLOBAL STATE & TUNING PARAMETERS
# =============================================================================

class AppState:
    """Single source of truth for all runtime parameters.
    Both the GUI and the engine read/write this shared object."""

    # --- User Biometrics ---
    USER_HEIGHT_CM: float = 180.0
    USER_WEIGHT_KG: float = 75.0

    # --- System Toggles ---
    VOICE_ON: bool = True
    AR_MODE: bool = False

    # --- File Paths ---
    GUIDE_PATH: str = "Video_Generation_Person_Squatting.mp4"

    # --- Squat Analysis Thresholds ---
    PARAM_SQUAT_DEPTH: float = 140.0   # Knee angle that counts as "down"
    PARAM_UP_THRESHOLD: float = 160.0  # Knee angle that counts as "standing"
    PARAM_LEAN_WARN: float = 40.0      # Trunk lean degrees → "Chest Up" warning
    PARAM_LEAN_CRIT: float = 55.0      # Trunk lean degrees → critical alert
    PARAM_ROUNDING: float = 18.0       # Max back curvature degrees allowed

    # --- Session History (in-memory, not persisted) ---
    HISTORY: list = []


# Singleton instance shared across the entire application
state = AppState()


# =============================================================================
#  PART 1: HOLOGRAPHIC AR ENGINE
# =============================================================================

class HologramProjector:
    """Draws the animated floor-target AR overlay onto a camera frame.
    Call draw() each frame; it returns True when the subject is in-zone."""

    def __init__(self):
        self.spin_angle_1: float = 0
        self.spin_angle_2: float = 0
        self.pulse_val: float = 0
        self.pulse_dir: int = 1
        self.lock_anim: float = 0.0

    def draw(self, frame, landmarks, width: int, height: int) -> bool:
        # Advance animations
        self.spin_angle_1 = (self.spin_angle_1 + 2) % 360
        self.spin_angle_2 = (self.spin_angle_2 - 3) % 360
        self.pulse_val += 0.05 * self.pulse_dir
        if self.pulse_val > 1.0 or self.pulse_val < 0.0:
            self.pulse_dir *= -1

        cx, cy = int(width // 2), int(height * 0.85)
        base_w, base_h = 140, 50

        overlay = frame.copy()
        status = "SEARCHING..."
        COLOR_BASE = (255, 200, 0)
        COLOR_LOCK = (0, 215, 255)
        COLOR_WARN = (0, 0, 255)
        active_color = COLOR_BASE
        in_zone = False

        if landmarks:
            l_ankle = landmarks[27]
            r_ankle = landmarks[28]
            lx, ly = int(l_ankle.x * width), int(l_ankle.y * height)
            rx, ry = int(r_ankle.x * width), int(r_ankle.y * height)
            fx, fy = (lx + rx) // 2, (ly + ry) // 2

            dist = math.hypot(cx - fx, cy - fy)
            in_zone = dist < 70

            if in_zone:
                self.lock_anim = min(1.0, self.lock_anim + 0.1)
                active_color = COLOR_LOCK
                status = "TARGET LOCKED"
            else:
                self.lock_anim = max(0.0, self.lock_anim - 0.1)
                if fy < (cy - 60):
                    active_color = COLOR_WARN
                    status = "MOVE BACK"
                elif fy > (cy + 60):
                    active_color = COLOR_WARN
                    status = "MOVE FWD"

            cv2.line(overlay, (lx, ly), (cx, cy), active_color, 1)
            cv2.line(overlay, (rx, ry), (cx, cy), active_color, 1)
            cv2.circle(overlay, (lx, ly), 4, active_color, -1)
            cv2.circle(overlay, (rx, ry), 4, active_color, -1)

        # Reactor core pulse
        core_size = int(10 + 5 * self.pulse_val)
        cv2.ellipse(overlay, (cx, cy), (core_size * 2, core_size), 0, 0, 360, active_color, -1)

        # Spinning rings
        for i in range(3):
            s = self.spin_angle_2 + (i * 120)
            cv2.ellipse(overlay, (cx, cy), (base_w - 20, base_h - 10), 0, s, s + 80, active_color, 2)

        if self.lock_anim > 0.8:
            cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, 0, 360, active_color, 3)
        else:
            for i in range(4):
                s = self.spin_angle_1 + (i * 90)
                cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, s, s + 40, active_color, 2)

        # Radial grid lines
        grid_overlay = overlay.copy()
        for i in range(0, 180, 20):
            rad = math.radians(i)
            x_off = int(math.cos(rad) * (base_w + 50))
            y_off = int(math.sin(rad) * (base_h + 30))
            cv2.line(grid_overlay, (cx, cy), (cx + x_off, cy + y_off), active_color, 1)
        cv2.addWeighted(grid_overlay, 0.3, overlay, 0.7, 0, overlay)

        cv2.putText(overlay, f"STATUS: {status}", (cx - 80, cy + base_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, active_color, 1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        return in_zone


# Module-level singleton — imported by the worker
hologram = HologramProjector()


# =============================================================================
#  PART 2: POSE MATH ENGINE
# =============================================================================

def calculate_angle_3d(a, b, c) -> float:
    """Calculate the joint angle at point b, using 3D world coordinates."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def is_profile_view(landmarks) -> bool:
    """Returns True when the subject has turned sideways (shoulder X-spread < threshold)."""
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    return abs(l_sh.x - r_sh.x) < 0.20


try:
    STS_MODEL = load_model("sit_to_stand_robust.keras")
except:
    STS_MODEL = None
    print("[Physio-Vision] WARNING: sit_to_stand_robust.keras not found.")

def normalize_skeleton_sts_live(frames_list):
    """Formats the 88 captured frames exactly how the Keras model expects it."""
    data = np.array(frames_list).reshape(1, 88, 22, 3) # Batch 1, 88 frames, 22 joints, 3 dims
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)


def extract_prmd_features(lm):
    """Translates MediaPipe's 33 landmarks into UI-PRMD's 22 specific joints."""

    # FIX: INVERT THE Y-AXIS (MediaPipe is positive-down, PRMD is positive-up)
    def pt(i): return [lm[i].x, -lm[i].y, lm[i].z]

    def avg(i, j): return [(lm[i].x + lm[j].x) / 2, -(lm[i].y + lm[j].y) / 2, (lm[i].z + lm[j].z) / 2]

    prmd = [
        avg(23, 24), pt(23), avg(11, 12), avg(11, 12), pt(0), pt(0),
        pt(11), pt(13), pt(15), pt(15), pt(12), pt(14), pt(16), pt(16),
        pt(24), pt(26), pt(28), pt(32), pt(23), pt(25), pt(27), pt(31)
    ]
    return [coord for joint in prmd for coord in joint]



def analyze_form_mechanics_3d(world_landmarks, stage: str, knee_angle: float):
    """Analyse squat form from 3-D world landmarks.
    Returns (penalty: float, feedback: list[str])."""
    penalty = 0.0
    feedback = []

    def ext(idx):
        return np.array([world_landmarks[idx].x, world_landmarks[idx].y, world_landmarks[idx].z])

    def unit_vector(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    l_sh,  r_sh  = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    mid_sh  = (l_sh  + r_sh)  / 2
    mid_hip = (l_hip + r_hip) / 2

    # --- 1. Trunk lean check (height-adjusted tolerance) ---
    lean_tolerance = state.PARAM_LEAN_WARN + (state.USER_HEIGHT_CM - 170) * 0.1
    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, 1, 0])
    lean_angle = float(np.degrees(np.arccos(np.clip(np.dot(unit_vector(spine_vec), vertical_vec), -1.0, 1.0))))
    lean_from_vertical = abs(180 - lean_angle)

    if stage == "DOWN" or knee_angle < state.PARAM_SQUAT_DEPTH:
        if lean_from_vertical > state.PARAM_LEAN_CRIT:
            penalty += 0.30
            feedback.append("CRITICAL LEAN")
        elif lean_from_vertical > lean_tolerance:
            penalty += 0.10
            feedback.append("Chest Up")

    # --- 2. Back rounding check ---
    collarbone_vec = r_sh - l_sh
    dot_round = np.abs(np.dot(unit_vector(spine_vec), unit_vector(collarbone_vec)))
    rounding_angle = float(np.degrees(np.arcsin(np.clip(dot_round, 0.0, 1.0))))

    if stage == "DOWN" and rounding_angle > state.PARAM_ROUNDING:
        penalty += 0.20
        feedback.insert(0, "BACK ROUNDING")

    return penalty, feedback


# =============================================================================
#  PART 3: AUDIO (NON-BLOCKING)
# =============================================================================

def speak_async(text: str) -> None:
    """Fire-and-forget TTS call on a daemon thread. Silently ignored if voice is off."""
    if not state.VOICE_ON:
        return

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()



try:
    SQUAT_MODEL = load_model("deep_squat_robust.keras")
except:
    SQUAT_MODEL = None
    print("WARNING: deep_squat_robust.keras not found.")

try:
    STS_MODEL = load_model("sit_to_stand_robust.keras")
except:
    STS_MODEL = None
    print("WARNING: sit_to_stand_robust.keras not found.")


# --- 2. NORMALIZATION FUNCTIONS ---
def normalize_skeleton_squat_live(frames_list):
    """Squat Normalizer: 81 frames, scales by Spine Length"""
    import numpy as np
    data = np.array(frames_list).reshape(1, 81, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    spine_top = data[:, :, 2:3, :]
    spine_len = np.linalg.norm(spine_top, axis=3, keepdims=True)
    data = data / np.maximum(spine_len, 0.0001)
    return data.reshape(1, 81, 66)

def normalize_skeleton_sts_live(frames_list):
    """STS Normalizer: 88 frames, scales by Pelvis Width"""
    import numpy as np
    data = np.array(frames_list).reshape(1, 88, 22, 3)
    root = data[:, :, 0:1, :]
    data = data - root
    left_hip, right_hip = data[:, :, 18:19, :], data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)
    data = data / np.maximum(pelvis_width, 0.0001)
    return data.reshape(1, 88, 66)


# =============================================================================
#  PART 4: VISION WORKER THREAD
# =============================================================================

class VisionWorker(QThread):
    """Runs the camera loop + MediaPipe pose on a background thread.
    Communicates back to the GUI exclusively via Qt signals."""

    frame_processed  = Signal(QImage)       # Rendered camera frame
    stats_update     = Signal(dict)         # Rep count / score / feedback
    system_status    = Signal(str, str)     # (label_text, hex_color)
    session_finished = Signal(dict)         # Full session report dict

    # Internal FSM states
    STATE_CALIB   = 0
    STATE_WARMUP  = 1
    STATE_SESSION = 2

    def __init__(self):
        super().__init__()
        self.running = False
        self.exercise_mode = "squat"  # Default, UI will change this
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.current_state = self.STATE_CALIB
        self.reset_session()

    def reset_session(self) -> None:
        self.reps = 0
        self.stage = "UP"
        self.max_rep_penalty = 0.0
        self.last_speech_time = 0.0
        self.calib_data = []
        self.session_log = []
        self.start_time = None
        self.ar_locked = False

        # --- NEW STS TRACKERS ---
        self.sts_stage = "WAITING"
        self.sts_timer = 0.0
        self.sts_buffer = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.current_state = self.STATE_CALIB
        self.reset_session()
        self.start_time = datetime.now()
        self.system_status.emit("INITIALIZING...", "#ffaa00")

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            results = self.pose.process(rgb_frame)

            # AR overlay
            if state.AR_MODE:
                lm_list = results.pose_landmarks.landmark if results.pose_landmarks else None
                self.ar_locked = hologram.draw(rgb_frame, lm_list, w, h)
                if not self.ar_locked and self.current_state == self.STATE_SESSION:
                    cv2.putText(rgb_frame, "ALIGN WITH TARGET", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                self.ar_locked = True

            # Pose processing
            if results.pose_landmarks and self.ar_locked:
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.process_logic(results)

            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_processed.emit(qt_img)
            self.msleep(30)

        self.cap.release()

        # Build and emit the final session report
        report = {
            "date":      self.start_time.strftime("%Y-%m-%d %H:%M") if self.start_time else "Unknown",
            "reps":      self.reps,
            "avg_score": self._calculate_avg_score(),
            "details":   self.session_log,
        }
        self.session_finished.emit(report)

    # ------------------------------------------------------------------
    # Finite state machine
    # ------------------------------------------------------------------
    def process_logic(self, results) -> None:
        landmarks_2d = results.pose_landmarks.landmark

        # ==========================================
        # STATE 0: PROFILE CALIBRATION
        # ==========================================
        if self.current_state == self.STATE_CALIB:
            if not is_profile_view(landmarks_2d):
                self.system_status.emit("Turn Sideways", "#ff4444")
                self.calib_data = []
            else:
                self.system_status.emit("CALIBRATING...", "#0099ff")
                self.calib_data.append(1)
                if len(self.calib_data) > 30:
                    self.current_state = self.STATE_WARMUP
                    speak_async("System Ready.")

        # ==========================================
        # STATE 1: WARMUP (Transition)
        # ==========================================
        elif self.current_state == self.STATE_WARMUP:
            self.system_status.emit("GET IN POSITION", "#00cc66")
            time.sleep(0.5)
            self.current_state = self.STATE_SESSION

        # ==========================================
        # STATE 2: ACTIVE SESSION (Hybrid Engine)
        # ==========================================
        elif self.current_state == self.STATE_SESSION:
            landmarks_3d = results.pose_world_landmarks.landmark
            knee_angle = calculate_angle_3d(landmarks_3d[23], landmarks_3d[25], landmarks_3d[27])

            # Simple heuristic triggers
            is_standing = knee_angle > 150
            is_sitting = knee_angle < 110

            # --- PHASE 1: WAITING FOR STARTING POSE ---
            if self.sts_stage == "WAITING":
                if self.exercise_mode == "squat" and is_standing:
                    self.sts_stage = "HOLDING"
                    self.sts_timer = time.time()
                    self.system_status.emit("STAND STILL...", "#ffaa00")
                elif self.exercise_mode == "sts" and is_sitting:
                    self.sts_stage = "HOLDING"
                    self.sts_timer = time.time()
                    self.system_status.emit("HOLD STILL...", "#ffaa00")

            # --- PHASE 2: CONFIRMING POSE ---
            elif self.sts_stage == "HOLDING":
                # If they break pose early, reset
                if (self.exercise_mode == "squat" and not is_standing) or (
                        self.exercise_mode == "sts" and not is_sitting):
                    self.sts_stage = "WAITING"
                    self.system_status.emit("GET IN POSITION", "#ffaa00")

                # If they hold perfectly for 2 seconds, begin recording!
                elif time.time() - self.sts_timer > 2.0:
                    self.sts_stage = "RECORDING"
                    self.sts_buffer = []
                    self.hit_bottom = False  # Tracker for the squat depth

                    action_text = "SQUAT DOWN" if self.exercise_mode == "squat" else "STAND UP"
                    self.system_status.emit(f"● RECORDING ({action_text})", "#ff4444")
                    speak_async("Begin.")

            # --- PHASE 3: DYNAMIC RECORDING & DIAGNOSTICS ---
            elif self.sts_stage == "RECORDING":
                # 1. Save inverted frame to buffer
                self.sts_buffer.append(extract_prmd_features(landmarks_3d))

                # 2. Math Engine Diagnostics (Squat Only)
                if self.exercise_mode == "squat":
                    penalty, issues = analyze_form_mechanics_3d(landmarks_3d, "DOWN", knee_angle)
                    if issues and not hasattr(self, 'current_rep_issues'):
                        self.current_rep_issues = issues[0]

                    # Stop Condition: They went deep enough, and are now back to standing
                    if knee_angle < state.PARAM_SQUAT_DEPTH:
                        self.hit_bottom = True
                    if getattr(self, 'hit_bottom', False) and is_standing:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                # 3. Math Engine Diagnostics (STS Only)
                elif self.exercise_mode == "sts":
                    # Stop Condition: The rep is finished the moment they are fully standing
                    if is_standing:
                        self.sts_stage = "INFERENCE"
                        self.system_status.emit("ANALYZING AI...", "#0099ff")

                # 4. Failsafe Timeout
                if len(self.sts_buffer) > 200:
                    self.sts_stage = "WAITING"
                    self.system_status.emit("TIMEOUT. RESETTING.", "#ffaa00")

            # --- PHASE 4: TIME WARPING & AI GRADING ---
            elif self.sts_stage == "INFERENCE":
                try:
                    # 1. DYNAMIC TIME WARPING
                    import cv2
                    raw_frames = np.array(self.sts_buffer, dtype=np.float32)
                    target_length = 81 if self.exercise_mode == "squat" else 88

                    # Stretch or squash the movement to exact AI requirements
                    warped_frames = cv2.resize(raw_frames, (66, target_length), interpolation=cv2.INTER_LINEAR)

                    # 2. KERAS PREDICTION
                    if self.exercise_mode == "squat" and SQUAT_MODEL:
                        normalized = normalize_skeleton_squat_live(warped_frames)
                        prediction = SQUAT_MODEL.predict(normalized, verbose=0)[0][0]
                    elif self.exercise_mode == "sts" and STS_MODEL:
                        normalized = normalize_skeleton_sts_live(warped_frames)
                        prediction = STS_MODEL.predict(normalized, verbose=0)[0][0]
                    else:
                        prediction = 0.85  # Fallback

                    self.reps += 1
                    # --- THE MAGICAL MATH: MIN-MAX SCALING ---
                    # The absolute minimum and maximum scores from the UI-PRMD dataset
                    raw_min = 0.60
                    raw_max = 0.96

                    # Stretch the prediction to a 0-100 scale
                    mapped_score = ((prediction - raw_min) / (raw_max - raw_min)) * 100

                    # "Clamp" the score to guarantee it stays between 0 and 100
                    score = int(max(0, min(100, mapped_score)))

                    # 3. COMBINE FEEDBACK
                    feedback = "Excellent Form"
                    if self.exercise_mode == "squat" and hasattr(self, 'current_rep_issues'):
                        feedback = self.current_rep_issues
                        del self.current_rep_issues
                    elif score < 80:
                        feedback = "Compensatory Motion Detected"

                    # 4. Log and update UI
                    log_entry = {"rep_num": self.reps, "score": score, "issue": feedback}
                    self.session_log.append(log_entry)
                    self.stats_update.emit({"reps": self.reps, "score": score, "feedback": feedback})

                    # 5. Speak the results
                    speak_text = f"Rep {self.reps}."
                    if feedback != "Excellent Form":
                        speak_text += f" {feedback}."
                    speak_async(speak_text)

                except Exception as e:
                    print(f"AI Inference Error: {e}")

                self.sts_stage = "WAITING"
                self.system_status.emit("RESETTING...", "#ffaa00")

    def stop(self) -> None:
        self.running = False
        self.wait()

    def _calculate_avg_score(self) -> int:
        if not self.session_log:
            return 0
        return int(sum(x["score"] for x in self.session_log) / len(self.session_log))