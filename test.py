import sys
import cv2
import time
import threading
import numpy as np
import pyttsx3
import os
import math
from datetime import datetime
import requests
from auth import LoginWindow, API_URL

# --- 1. CRASH PREVENTION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# --- 2. GUI IMPORTS ---
from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot, QSize, QPropertyAnimation, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (QApplication, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QSizePolicy, QScrollArea, QFrame, QGraphicsOpacityEffect)

from qfluentwidgets import (FluentWindow, NavigationItemPosition, TitleLabel,
                            PrimaryPushButton, ProgressBar, BodyLabel, StrongBodyLabel,
                            CardWidget, setTheme, Theme, LineEdit, SwitchButton,
                            InfoBar, InfoBarPosition, ScrollArea,
                            ExpandGroupSettingCard, DoubleSpinBox, IconWidget, ToolButton)
from qfluentwidgets import FluentIcon as FIF

# --- 3. AI IMPORTS ---
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


# =============================================================================
#  GLOBAL STATE & TUNING PARAMETERS
# =============================================================================
class AppState:
    # --- USER STATS ---
    USER_HEIGHT_CM = 180.0
    USER_WEIGHT_KG = 75.0

    # --- SYSTEM SETTINGS ---
    VOICE_ON = True
    AR_MODE = False
    GUIDE_PATH = "Video_Generation_Person_Squatting.mp4"  # <--- CHECK THIS PATH

    # --- LIVE TUNING PARAMETERS (The "Developer" variables) ---
    PARAM_SQUAT_DEPTH = 140.0  # Angle to count as "Down" (Lower is deeper)
    PARAM_UP_THRESHOLD = 160.0  # Angle to count as "Up" (Standing)
    PARAM_LEAN_WARN = 40.0  # Degrees of lean to trigger "Chest Up"
    PARAM_LEAN_CRIT = 55.0  # Degrees of lean to trigger "Critical Lean"
    PARAM_ROUNDING = 18.0  # Degrees of back curvature allowed

    HISTORY = []


state = AppState()


# =============================================================================
#  PART 1: THE HOLOGRAPHIC ENGINE
# =============================================================================

class HologramProjector:
    def __init__(self):
        self.spin_angle_1 = 0
        self.spin_angle_2 = 0
        self.pulse_val = 0
        self.pulse_dir = 1
        self.lock_anim = 0.0

    def draw(self, frame, landmarks, width, height):
        self.spin_angle_1 = (self.spin_angle_1 + 2) % 360
        self.spin_angle_2 = (self.spin_angle_2 - 3) % 360
        self.pulse_val += 0.05 * self.pulse_dir
        if self.pulse_val > 1.0 or self.pulse_val < 0.0: self.pulse_dir *= -1

        cx, cy = int(width // 2), int(height * 0.85)
        base_w, base_h = 140, 50

        overlay = frame.copy()
        status = "SEARCHING..."
        color_base = (255, 200, 0)
        color_lock = (0, 215, 255)
        color_warn = (0, 0, 255)

        active_color = color_base
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
                active_color = color_lock
                status = "TARGET LOCKED"
            else:
                self.lock_anim = max(0.0, self.lock_anim - 0.1)
                if fy < (cy - 60):
                    active_color = color_warn; status = "MOVE BACK"
                elif fy > (cy + 60):
                    active_color = color_warn; status = "MOVE FWD"

            cv2.line(overlay, (lx, ly), (cx, cy), active_color, 1)
            cv2.line(overlay, (rx, ry), (cx, cy), active_color, 1)
            cv2.circle(overlay, (lx, ly), 4, active_color, -1)
            cv2.circle(overlay, (rx, ry), 4, active_color, -1)

        # Reactor Core
        core_size = int(10 + 5 * self.pulse_val)
        cv2.ellipse(overlay, (cx, cy), (core_size * 2, core_size), 0, 0, 360, active_color, -1)

        # Rings
        start_ang = self.spin_angle_2
        for i in range(3):
            s = start_ang + (i * 120)
            cv2.ellipse(overlay, (cx, cy), (base_w - 20, base_h - 10), 0, s, s + 80, active_color, 2)

        if self.lock_anim > 0.8:
            cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, 0, 360, active_color, 3)
        else:
            start_ang = self.spin_angle_1
            for i in range(4):
                s = start_ang + (i * 90)
                cv2.ellipse(overlay, (cx, cy), (base_w, base_h), 0, s, s + 40, active_color, 2)

        # Grid & Text
        grid_alpha = 0.3
        grid_overlay = overlay.copy()
        for i in range(0, 180, 20):
            rad = math.radians(i)
            x_off = int(math.cos(rad) * (base_w + 50))
            y_off = int(math.sin(rad) * (base_h + 30))
            cv2.line(grid_overlay, (cx, cy), (cx + x_off, cy + y_off), active_color, 1)
        cv2.addWeighted(grid_overlay, grid_alpha, overlay, 1 - grid_alpha, 0, overlay)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"STATUS: {status}", (cx - 80, cy + base_h + 40), font, 0.5, active_color, 1)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        return in_zone


hologram = HologramProjector()


# =============================================================================
#  PART 2: THE MATH ENGINE (Now Linked to Live State)
# =============================================================================

def calculate_angle_3d(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def is_profile_view(landmarks):
    l_sh = landmarks[11]
    r_sh = landmarks[12]
    return abs(l_sh.x - r_sh.x) < 0.20


def analyze_form_mechanics_3d(world_landmarks, stage, knee_angle):
    penalty = 0.0
    feedback = []

    def ext(idx):
        return np.array([world_landmarks[idx].x, world_landmarks[idx].y, world_landmarks[idx].z])

    def unit_vector(v):
        return v / np.linalg.norm(v)

    l_sh, r_sh = ext(11), ext(12)
    l_hip, r_hip = ext(23), ext(24)
    mid_sh, mid_hip = (l_sh + r_sh) / 2, (l_hip + r_hip) / 2

    # --- 1. DYNAMIC LEAN CHECK ---
    # We use state.PARAM_LEAN_WARN instead of hardcoded numbers
    lean_tolerance = state.PARAM_LEAN_WARN + (state.USER_HEIGHT_CM - 170) * 0.1

    spine_vec = mid_sh - mid_hip
    vertical_vec = np.array([0, 1, 0])
    lean_angle = np.degrees(np.arccos(np.clip(np.dot(unit_vector(spine_vec), vertical_vec), -1.0, 1.0)))
    lean_from_vertical = abs(180 - lean_angle)

    if stage == "DOWN" or knee_angle < state.PARAM_SQUAT_DEPTH:
        if lean_from_vertical > state.PARAM_LEAN_CRIT:
            penalty += 0.30
            feedback.append("CRITICAL LEAN")
        elif lean_from_vertical > lean_tolerance:
            penalty += 0.10
            feedback.append("Chest Up")

    # --- 2. DYNAMIC ROUNDING CHECK ---
    collarbone_vec = r_sh - l_sh
    dot_prod_round = np.abs(np.dot(unit_vector(spine_vec), unit_vector(collarbone_vec)))
    rounding_angle = np.degrees(np.arcsin(np.clip(dot_prod_round, 0.0, 1.0)))

    if stage == "DOWN" and rounding_angle > state.PARAM_ROUNDING:
        penalty += 0.20
        feedback.insert(0, "BACK ROUNDING")

    return penalty, feedback


# =============================================================================
#  PART 3: WORKER & AUDIO
# =============================================================================

def speak_async(text):
    if not state.VOICE_ON: return

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass

    threading.Thread(target=_speak, daemon=True).start()


class VisionWorker(QThread):
    frame_processed = Signal(QImage)
    stats_update = Signal(dict)
    system_status = Signal(str, str)
    session_finished = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.reset_session()
        self.STATE_CALIB, self.STATE_WARMUP, self.STATE_SESSION = 0, 1, 2
        self.current_state = 0

    def reset_session(self):
        self.reps = 0
        self.stage = "UP"
        self.max_rep_penalty = 0.0
        self.last_speech_time = 0
        self.calib_data = []
        self.session_log = []
        self.start_time = None
        self.ar_locked = False

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.current_state = self.STATE_CALIB
        self.reset_session()
        self.start_time = datetime.now()
        self.system_status.emit("INITIALIZING...", "#ffaa00")

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape

            results = self.pose.process(rgb_frame)

            if state.AR_MODE:
                landmarks_list = results.pose_landmarks.landmark if results.pose_landmarks else None
                self.ar_locked = hologram.draw(rgb_frame, landmarks_list, w, h)
                if not self.ar_locked and self.current_state == self.STATE_SESSION:
                    cv2.putText(rgb_frame, "ALIGN WITH TARGET", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                self.ar_locked = True

            if results.pose_landmarks and self.ar_locked:
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                self.process_logic(results)

            qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_processed.emit(qt_img)
            self.msleep(30)

        self.cap.release()

        final_score = self.calculate_avg_score()
        report = {
            "date": self.start_time.strftime("%Y-%m-%d %H:%M"),
            "reps": self.reps,
            "avg_score": final_score,
            "details": self.session_log
        }
        self.session_finished.emit(report)

    def calculate_avg_score(self):
        if not self.session_log: return 0
        total = sum([x['score'] for x in self.session_log])
        return int(total / len(self.session_log))

    def process_logic(self, results):
        landmarks_2d = results.pose_landmarks.landmark

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

        elif self.current_state == self.STATE_WARMUP:
            self.system_status.emit("START SQUATTING", "#00cc66")
            time.sleep(0.5)
            self.current_state = self.STATE_SESSION

        elif self.current_state == self.STATE_SESSION:
            landmarks_3d = results.pose_world_landmarks.landmark
            angle = calculate_angle_3d(landmarks_3d[23], landmarks_3d[25], landmarks_3d[27])

            penalty, issues = analyze_form_mechanics_3d(landmarks_3d, self.stage, angle)

            if self.stage == "DOWN" and penalty > self.max_rep_penalty:
                self.max_rep_penalty = penalty

            if issues and (time.time() - self.last_speech_time > 3.0):
                speak_async(issues[0])
                self.last_speech_time = time.time()

            # Dynamic Rep Counting
            if angle < state.PARAM_SQUAT_DEPTH and self.stage == "UP":
                self.stage = "DOWN"
                self.max_rep_penalty = 0.0

            if angle > state.PARAM_UP_THRESHOLD and self.stage == "DOWN":
                self.stage = "UP"
                self.reps += 1
                score = int(max(0.0, min(1.0, 1.0 * 1.15) - self.max_rep_penalty) * 100)

                log_entry = {"rep_num": self.reps, "score": score, "issue": issues[0] if issues else "Perfect Form"}
                self.session_log.append(log_entry)

                self.stats_update.emit({"reps": self.reps, "score": score, "feedback": log_entry['issue']})

                phrase = f"Rep {self.reps}"
                if issues: phrase += f". {issues[0]}"
                speak_async(phrase)

    def stop(self):
        self.running = False
        self.wait()


# =============================================================================
#  PART 4: WINDOWS (Developer, Reference, Settings)
# =============================================================================

class DeveloperToolsWindow(QWidget):
    """
    THE DEVELOPER CONSOLE: Live Parameter Tuning
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physio DevTools (Live)")
        self.resize(350, 450)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        layout.addWidget(StrongBodyLabel("Real-Time Mechanics Engine"))

        # Helper to create sliders
        def create_tuner(label, val, min_v, max_v, callback):
            l = QHBoxLayout()
            l.addWidget(BodyLabel(label))
            spin = DoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setValue(val)
            spin.valueChanged.connect(callback)
            l.addWidget(spin)
            layout.addLayout(l)

        # 1. Squat Depth
        create_tuner("Squat Depth (Deg)", state.PARAM_SQUAT_DEPTH, 90, 170,
                     lambda v: setattr(state, 'PARAM_SQUAT_DEPTH', v))

        # 2. Stand Up Threshold
        create_tuner("Stand Up (Deg)", state.PARAM_UP_THRESHOLD, 150, 180,
                     lambda v: setattr(state, 'PARAM_UP_THRESHOLD', v))

        # 3. Lean Warning
        create_tuner("Lean Warn (Deg)", state.PARAM_LEAN_WARN, 10, 80,
                     lambda v: setattr(state, 'PARAM_LEAN_WARN', v))

        # 4. Critical Lean
        create_tuner("Lean Crit (Deg)", state.PARAM_LEAN_CRIT, 20, 90,
                     lambda v: setattr(state, 'PARAM_LEAN_CRIT', v))

        # 5. Rounding
        create_tuner("Back Rounding (Deg)", state.PARAM_ROUNDING, 5, 45,
                     lambda v: setattr(state, 'PARAM_ROUNDING', v))

        layout.addStretch(1)
        layout.addWidget(BodyLabel("Changes apply immediately to next frame."))


class ReferenceWindow(QWidget):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Pro Form Guide")
        self.resize(400, 600)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Loading Guide...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.layout.addWidget(self.video_label)
        self.cap = cv2.VideoCapture(video_path)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(30)

    def next_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# =============================================================================
#  PART 5: DASHBOARD PAGES
# =============================================================================

class SettingsPage(ScrollArea):
    def __init__(self):
        super().__init__()
        self.view = QWidget()
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("settings_page")
        self.view.setStyleSheet("background-color: transparent;")
        self.setStyleSheet("background-color: transparent; border: none;")

        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setContentsMargins(30, 30, 30, 30)
        self.v_layout.setSpacing(20)

        self.v_layout.addWidget(TitleLabel("System Configuration"))

        # --- 1. BIOMETRICS (RESTORED WEIGHT) ---
        self.bio_card = CardWidget()
        bio_layout = QVBoxLayout(self.bio_card)
        bio_layout.addWidget(StrongBodyLabel("User Biometrics"))
        bio_layout.addWidget(BodyLabel("Height/Weight used for mechanics calibration."))

        # Height
        h_layout = QHBoxLayout()
        self.height_input = DoubleSpinBox()
        self.height_input.setRange(100, 250)
        self.height_input.setValue(state.USER_HEIGHT_CM)
        self.height_input.valueChanged.connect(lambda v: setattr(state, 'USER_HEIGHT_CM', v))
        h_layout.addWidget(QLabel("Height (cm):"))
        h_layout.addWidget(self.height_input)
        bio_layout.addLayout(h_layout)

        # Weight (FIXED: Added back)
        w_layout = QHBoxLayout()
        self.weight_input = DoubleSpinBox()
        self.weight_input.setRange(40, 200)
        self.weight_input.setValue(state.USER_WEIGHT_KG)
        self.weight_input.valueChanged.connect(lambda v: setattr(state, 'USER_WEIGHT_KG', v))
        w_layout.addWidget(QLabel("Weight (kg):"))
        w_layout.addWidget(self.weight_input)
        bio_layout.addLayout(w_layout)

        self.v_layout.addWidget(self.bio_card)

        # --- 2. AR GUIDANCE ---
        self.ar_card = CardWidget()
        ar_layout = QHBoxLayout(self.ar_card)
        icon_widget = IconWidget(FIF.CAMERA)
        icon_widget.setFixedSize(24, 24)
        ar_layout.addWidget(icon_widget)

        ar_text_layout = QVBoxLayout()
        ar_text_layout.addWidget(StrongBodyLabel("Holographic Guidance"))
        ar_text_layout.addWidget(BodyLabel("Enable projected floor target."))
        ar_layout.addLayout(ar_text_layout)

        ar_layout.addStretch(1)
        self.ar_switch = SwitchButton()
        self.ar_switch.setOnText("ON")
        self.ar_switch.setOffText("OFF")
        self.ar_switch.setChecked(state.AR_MODE)
        self.ar_switch.checkedChanged.connect(self.toggle_ar)
        ar_layout.addWidget(self.ar_switch)
        self.v_layout.addWidget(self.ar_card)

        # --- 3. VOICE ASSISTANT (FIXED: Added back) ---
        self.voice_card = CardWidget()
        voice_layout = QHBoxLayout(self.voice_card)

        voice_text_layout = QVBoxLayout()
        voice_text_layout.addWidget(StrongBodyLabel("AI Voice Assistant"))
        voice_text_layout.addWidget(BodyLabel("Enable audio feedback during reps."))
        voice_layout.addLayout(voice_text_layout)

        voice_layout.addStretch(1)
        self.voice_switch = SwitchButton()
        self.voice_switch.setOnText("ON")
        self.voice_switch.setOffText("OFF")
        self.voice_switch.setChecked(state.VOICE_ON)
        self.voice_switch.checkedChanged.connect(lambda v: setattr(state, 'VOICE_ON', v))
        voice_layout.addWidget(self.voice_switch)
        self.v_layout.addWidget(self.voice_card)

        # --- 4. DEV TOOLS ---
        self.dev_card = CardWidget()
        dev_layout = QHBoxLayout(self.dev_card)
        dev_layout.addWidget(StrongBodyLabel("Developer Mode"))
        dev_layout.addStretch(1)
        self.btn_dev = PrimaryPushButton("Open Console", self)
        self.btn_dev.clicked.connect(self.open_dev_console)
        dev_layout.addWidget(self.btn_dev)
        self.v_layout.addWidget(self.dev_card)

        self.v_layout.addStretch(1)

    def toggle_ar(self, val):
        state.AR_MODE = val
        mode = "Holo-Guide" if val else "Standard"
        if self.window():
            InfoBar.info(title='Tracking Mode', content=f"Switched to {mode}", parent=self.window())

    def open_dev_console(self):
        self.dev_window = DeveloperToolsWindow()
        self.dev_window.show()


class RecordsPage(ScrollArea):
    def __init__(self):
        super().__init__()
        self.view = QWidget()
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("records_page")
        self.view.setStyleSheet("background-color: transparent;")
        self.setStyleSheet("background-color: transparent; border: none;")
        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setContentsMargins(30, 30, 30, 30)
        self.v_layout.setSpacing(15)
        self.title = TitleLabel("Patient History")
        self.v_layout.addWidget(self.title)
        self.history_layout = QVBoxLayout()
        self.v_layout.addLayout(self.history_layout)
        self.v_layout.addStretch(1)

    def add_record(self, report):
        title = f"Session: {report['date']}"
        content = f"Reps: {report['reps']} | Score: {report['avg_score']}%"
        card = ExpandGroupSettingCard(icon=FIF.HISTORY, title=title, content=content)
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(20, 10, 20, 10)

        details = report.get('details', [])
        if details:
            for rep in details:
                row = QHBoxLayout()
                lbl_rep = StrongBodyLabel(f"Rep {rep['rep_num']}:")
                lbl_issue = BodyLabel(f"{rep['issue']}")
                lbl_score = BodyLabel(f"Score: {rep['score']}%")
                if rep['score'] < 70:
                    lbl_issue.setStyleSheet("color: #ff4444;")
                else:
                    lbl_issue.setStyleSheet("color: #00cc66;")
                row.addWidget(lbl_rep)
                row.addWidget(lbl_issue)
                row.addStretch(1)
                row.addWidget(lbl_score)
                detail_layout.addLayout(row)
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setStyleSheet("color: #333;")
                detail_layout.addWidget(line)
        else:
            # Cloud-loaded records have no per-rep breakdown
            lbl = BodyLabel("No per-rep breakdown available for this session.")
            lbl.setStyleSheet("color: #666;")
            detail_layout.addWidget(lbl)

        card.addGroupWidget(detail_widget)
        self.history_layout.insertWidget(0, card)

    # =============================================================================


#  PART 6: MAIN APP
# =============================================================================

class PhysioDashboard(FluentWindow):
    def __init__(self, username):
        super().__init__()
        self.current_user = username
        self.setWindowTitle(f"Physio-Vision Enterprise | User: {self.current_user}")
        self.resize(1200, 800)
        self.worker = VisionWorker()
        self.worker.frame_processed.connect(self.update_video)
        self.worker.stats_update.connect(self.update_metrics)
        self.worker.system_status.connect(self.update_status)
        self.worker.session_finished.connect(self.on_session_finish)
        self.home_interface = self.create_home_interface()
        self.home_interface.setObjectName("home_interface")
        self.records_interface = RecordsPage()
        self.settings_interface = SettingsPage()
        self.init_navigation()
        # Delay the network call by 200ms so the UI can safely render first
        QTimer.singleShot(200, self.fetch_cloud_history)





    def init_navigation(self):
        self.addSubInterface(self.home_interface, FIF.VIDEO, "Live Analysis")
        self.addSubInterface(self.records_interface, FIF.HEART, "Patient Records")
        self.addSubInterface(self.settings_interface, FIF.SETTING, "Settings", NavigationItemPosition.BOTTOM)

    def create_home_interface(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        video_card = CardWidget()
        v_layout = QVBoxLayout(video_card)
        header_layout = QHBoxLayout()
        self.lbl_status = StrongBodyLabel("SYSTEM OFFLINE")
        self.lbl_status.setStyleSheet("color: #666;")
        self.btn_guide = ToolButton(FIF.HELP, self)
        self.btn_guide.setToolTip("Watch Reference Video")
        self.btn_guide.clicked.connect(self.open_guide)
        header_layout.addWidget(self.lbl_status)
        header_layout.addStretch(1)
        header_layout.addWidget(self.btn_guide)
        v_layout.addLayout(header_layout)
        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 8px;")
        v_layout.addWidget(self.video_label)

        stats_card = CardWidget()
        stats_card.setFixedWidth(360)
        s_layout = QVBoxLayout(stats_card)
        s_layout.setSpacing(25)
        s_layout.addWidget(TitleLabel("Session Metrics"))
        s_layout.addWidget(BodyLabel("TOTAL REPS"))
        self.rep_val = TitleLabel("0")
        self.rep_val.setStyleSheet("font-size: 48px; color: #0099ff;")
        s_layout.addWidget(self.rep_val)
        s_layout.addWidget(BodyLabel("FORM QUALITY"))
        self.score_val = StrongBodyLabel("--")
        self.score_bar = ProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        s_layout.addWidget(self.score_val)
        s_layout.addWidget(self.score_bar)
        s_layout.addWidget(BodyLabel("AI FEEDBACK"))
        self.feedback_lbl = StrongBodyLabel("Waiting for start...")
        self.feedback_lbl.setStyleSheet("color: #999;")
        s_layout.addWidget(self.feedback_lbl)
        s_layout.addStretch(1)
        self.btn_action = PrimaryPushButton("INITIATE SYSTEM", self)
        self.btn_action.clicked.connect(self.toggle_session)
        self.btn_action.setMinimumHeight(50)
        s_layout.addWidget(self.btn_action)
        layout.addWidget(video_card, stretch=3)
        layout.addWidget(stats_card, stretch=1)
        return widget

    def open_guide(self):
        if not os.path.exists(state.GUIDE_PATH):
            InfoBar.warning(title="Error", content="Guide video file not found!", parent=self)
            return
        self.guide_window = ReferenceWindow(state.GUIDE_PATH)
        self.guide_window.show()

    @Slot(QImage)
    def update_video(self, img):
        pix = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    @Slot(dict)
    def update_metrics(self, data):
        self.rep_val.setText(str(data['reps']))
        self.score_val.setText(f"{data['score']}%")
        self.score_bar.setValue(data['score'])
        self.feedback_lbl.setText(data['feedback'])
        if data['score'] > 85:
            color = "#00cc66"
        elif data['score'] > 60:
            color = "#ffaa00"
        else:
            color = "#ff4444"
        self.score_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    @Slot(str, str)
    def update_status(self, text, color):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f"color: white; background-color: {color}; font-weight: bold; padding: 8px; border-radius: 4px;")


    @Slot(dict)
    def on_session_finish(self, report):
        # 1. Update the local UI history
        self.records_interface.add_record(report)

        # 2. Push to the Raspberry Pi Database!
        try:
            payload = {
                "username": self.current_user,
                "exercise": "Deep Squat",
                "reps": report['reps'],
                "score": report['avg_score'],
                "pain_level": 0  # We can build a pain popup next!
            }
            response = requests.post(f"{API_URL}/log_session", json=payload)

            if response.status_code == 200:
                InfoBar.success(title='Cloud Sync',
                                content="Session securely saved to database.",
                                orient=Qt.Horizontal, isClosable=True,
                                position=InfoBarPosition.TOP_RIGHT, parent=self)
            else:
                InfoBar.warning(title='Sync Failed',
                                content="Could not save to database.",
                                parent=self)
        except requests.exceptions.RequestException:
            InfoBar.error(title='Network Error',
                          content="Cannot reach the Raspberry Pi server.",
                          parent=self)

    def toggle_session(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.btn_action.setText("INITIATE SYSTEM")
            self.update_status("SYSTEM OFFLINE", "#333")
        else:
            self.worker.start()
            self.btn_action.setText("TERMINATE SESSION")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

    def fetch_cloud_history(self):
        try:
            response = requests.get(f"{API_URL}/get_history/{self.current_user}")

            if response.status_code == 200:
                data = response.json()
                cloud_records = data.get("history", [])

                # Reverse so oldest loads first → newest card ends up on top
                for row in reversed(cloud_records):
                    report = {
                        "date": row.get("date", "Unknown"),
                        "reps": row.get("reps", 0),
                        "avg_score": row.get("score", 0),
                        # This prevents the KeyError crash!
                        "details": []
                    }
                    self.records_interface.add_record(report)

        except requests.exceptions.RequestException:
            print("Could not fetch history from cloud. Booting in offline mode.")


if __name__ == '__main__':
    from PyQt5.QtCore import QTimer  # <--- Add this import here

    app = QApplication(sys.argv)
    setTheme(Theme.DARK)

    # 1. Prevent the app from crashing when switching windows
    app.setQuitOnLastWindowClosed(False)

    login_window = LoginWindow()


    def launch_dashboard(username):
        print(f"Building main dashboard for user: {username}...")
        global main_app  # Keep dashboard in memory
        main_app = PhysioDashboard(username)
        main_app.show()

        # Now that the dashboard is safe and visible, close the login window
        login_window.close()

        # Restore normal quit behavior
        app.setQuitOnLastWindowClosed(True)


    def trigger_transition(username):
        # This decouples the signal from the dashboard creation!
        print("Login successful. Decoupling thread...")
        QTimer.singleShot(100, lambda: launch_dashboard(username))


    # Connect to the decoupler instead of directly to the dashboard builder
    login_window.login_successful.connect(trigger_transition)

    login_window.show()
    sys.exit(app.exec())