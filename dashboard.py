# =============================================================================
#  dashboard.py — Physio-Vision GUI Layer
#  Professional medical-grade desktop interface.
#  Flow: Login → Hub → Exercise Analysis (per exercise) → back to Hub
#
#  Depends on: engine.py, auth.py
# =============================================================================

import sys
import os
import threading
import requests

from PyQt5.QtCore import (Qt, QTimer, QSize, pyqtSignal as Signal,
                           pyqtSlot as Slot)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QSizePolicy, QFrame, QGridLayout,
                              QGraphicsDropShadowEffect, QScrollArea)

from qfluentwidgets import (FluentWindow, NavigationItemPosition, TitleLabel,
                             PrimaryPushButton, PushButton, ProgressBar,
                             BodyLabel, StrongBodyLabel, CaptionLabel,
                             CardWidget, setTheme, Theme, SwitchButton,
                             InfoBar, InfoBarPosition, ScrollArea,
                             ExpandGroupSettingCard, DoubleSpinBox,
                             IconWidget, ToolButton, MessageBoxBase,
                             SubtitleLabel, Slider, TransparentPushButton,
                             LineEdit, SegmentedWidget)
from qfluentwidgets import FluentIcon as FIF

from auth import API_URL
from engine import state, VisionWorker, speak_async


# =============================================================================
#  STYLE CONSTANTS  (one place to change the look)
# =============================================================================

# Medical-grade palette: deep navy base, muted teal accent, clean white text
CLR_BG_DEEP    = "#0D1117"   # Window background
CLR_BG_CARD    = "#161B22"   # Card surface
CLR_BG_CARD2   = "#1C2230"   # Slightly lighter card
CLR_ACCENT     = "#1D7EC2"   # Primary action / highlight (medical blue)
CLR_ACCENT2    = "#15A589"   # Secondary accent (teal)
CLR_TEXT_PRI   = "#E6EDF3"   # Primary text
CLR_TEXT_SEC   = "#8B949E"   # Secondary / label text
CLR_GOOD       = "#2EA043"   # Good / success
CLR_WARN       = "#D29922"   # Warning
CLR_CRIT       = "#C93535"   # Critical / error
CLR_DIVIDER    = "#21262D"   # Separator lines
CLR_HOVER      = "#1F2937"   # Card hover state

CARD_RADIUS = "10px"
FONT_MONO   = "Consolas, 'Courier New', monospace"


def _shadow(widget, radius=18, alpha=80):
    """Attach a subtle drop-shadow to any QWidget."""
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(radius)
    fx.setOffset(0, 4)
    fx.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(fx)
    return fx


# =============================================================================
#  PART 1: REUSABLE METRIC CARD
# =============================================================================

class MetricCard(QFrame):
    """A compact KPI tile: icon line + big value + sub-label."""

    def __init__(self, icon_text: str, value: str, label: str,
                 accent: str = CLR_ACCENT, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CLR_BG_CARD};
                border-radius: {CARD_RADIUS};
                border: 1px solid {CLR_DIVIDER};
            }}
        """)
        self.setMinimumHeight(110)
        _shadow(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)

        icon_lbl = QLabel(icon_text)
        icon_lbl.setStyleSheet(f"color: {accent}; font-size: 18px; background: transparent; border: none;")
        layout.addWidget(icon_lbl)

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 28px; font-weight: 700;"
            f" background: transparent; border: none;"
        )
        layout.addWidget(self.value_lbl)

        sub = QLabel(label)
        sub.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(sub)

    def set_value(self, text: str):
        self.value_lbl.setText(text)


# =============================================================================
#  PART 2: HUB PAGE  (the landing screen after login)
# =============================================================================

class ExerciseCard(QFrame):
    """Clickable exercise tile shown on the Hub."""

    clicked = Signal(str)   # emits exercise_key

    def __init__(self, key: str, title: str, subtitle: str,
                 icon: str, status: str = "available", parent=None):
        super().__init__(parent)
        self.key = key
        self.status = status
        self._build(title, subtitle, icon, status)
        self.setCursor(Qt.PointingHandCursor if status == "available" else Qt.ForbiddenCursor)

    def _build(self, title, subtitle, icon, status):
        available = (status == "available")
        border_color = CLR_ACCENT if available else CLR_DIVIDER
        bg_color = CLR_BG_CARD if available else CLR_BG_CARD2
        text_color = CLR_TEXT_PRI if available else CLR_TEXT_SEC

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: {CARD_RADIUS};
                border: 1px solid {border_color};
            }}
            QFrame:hover {{
                background-color: {CLR_HOVER if available else bg_color};
                border: 1px solid {CLR_ACCENT if available else CLR_DIVIDER};
            }}
        """)
        self.setFixedHeight(140)
        _shadow(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        # Top row: icon + coming soon badge
        top = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(
            f"font-size: 26px; color: {CLR_ACCENT if available else CLR_TEXT_SEC};"
            " background: transparent; border: none;"
        )
        top.addWidget(icon_lbl)
        top.addStretch(1)
        if not available:
            badge = QLabel("COMING SOON")
            badge.setStyleSheet(
                f"color: {CLR_TEXT_SEC}; font-size: 9px; font-weight: 600;"
                f" background-color: {CLR_DIVIDER}; border-radius: 4px; padding: 2px 6px; border: none;"
            )
            top.addWidget(badge)
        layout.addLayout(top)

        # Title
        t = QLabel(title)
        t.setStyleSheet(
            f"color: {text_color}; font-size: 15px; font-weight: 600;"
            " background: transparent; border: none;"
        )
        layout.addWidget(t)

        # Subtitle
        s = QLabel(subtitle)
        s.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 11px;"
            " background: transparent; border: none;"
        )
        layout.addWidget(s)

    def mousePressEvent(self, event):
        if self.status == "available":
            self.clicked.emit(self.key)


class HubPage(QWidget):
    """The main hub: welcome header, KPI strip, exercise grid."""

    exercise_selected = Signal(str)   # forwards to PhysioDashboard

    EXERCISES = [
        ("squat",        "Deep Squat",        "Knee & hip mobility analysis",  "🦵", "available"),
        ("sts", "Sit to Stand", "Geriatric fall-risk assessment", "🪑", "available"),
        ("shoulder",     "Shoulder Press",    "Upper limb biomechanics",        "💪", "coming_soon"),
        ("balance",      "Balance Test",      "Static postural stability",      "⚖️", "coming_soon"),
        ("hip_flex",     "Hip Flexion",       "ROM measurement & logging",      "🔄", "coming_soon"),
        ("gait",         "Gait Analysis",     "Walking pattern & cadence",      "👣", "coming_soon"),
    ]

    def __init__(self, username: str, parent=None):
        super().__init__(parent)
        self.username = username
        self.setObjectName("hub_page")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 28, 30, 28)
        root.setSpacing(24)

        # ── Welcome header ────────────────────────────────────────────
        header = QHBoxLayout()
        greeting = QVBoxLayout()
        greeting.setSpacing(2)

        hi = QLabel(f"Welcome, {self.username}")
        hi.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 22px; font-weight: 700; background: transparent;"
        )
        greeting.addWidget(hi)

        sub = QLabel("Select a module below to begin your session")
        sub.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 13px; background: transparent;")
        greeting.addWidget(sub)

        header.addLayout(greeting)
        header.addStretch(1)

        # System status pill
        self.status_pill = QLabel("● SYSTEM READY")
        self.status_pill.setStyleSheet(
            f"color: {CLR_GOOD}; font-size: 11px; font-weight: 600;"
            f" background-color: #0D2818; border-radius: 12px; padding: 4px 12px; border: none;"
        )
        header.addWidget(self.status_pill)
        root.addLayout(header)

        # ── Divider ───────────────────────────────────────────────────
        root.addWidget(self._divider())

        # ── KPI strip ─────────────────────────────────────────────────
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.card_sessions = MetricCard("📋", "—", "Total Sessions",   CLR_ACCENT)
        self.card_avg_score = MetricCard("📊", "—", "Avg. Form Score",  CLR_ACCENT2)
        self.card_last_reps = MetricCard("🔁", "—", "Last Session Reps", CLR_WARN)
        self.card_last_pain  = MetricCard("💊", "—", "Last Pain Score",  CLR_CRIT)
        for c in [self.card_sessions, self.card_avg_score, self.card_last_reps, self.card_last_pain]:
            kpi_row.addWidget(c)
        root.addLayout(kpi_row)

        # ── Section label ─────────────────────────────────────────────
        mod_label = QLabel("Exercise Modules")
        mod_label.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 14px; font-weight: 600; background: transparent;"
        )
        root.addWidget(mod_label)

        # ── Exercise grid ─────────────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(14)
        for idx, (key, title, subtitle, icon, status) in enumerate(self.EXERCISES):
            card = ExerciseCard(key, title, subtitle, icon, status)
            card.clicked.connect(self.exercise_selected)
            grid.addWidget(card, idx // 3, idx % 3)
        root.addLayout(grid)
        root.addStretch(1)

    def update_kpis(self, records: list):
        """Refresh KPI tiles from a list of session dicts (from cloud)."""
        if not records:
            return
        total = len(records)
        avg_score = int(sum(r.get("score", 0) for r in records) / total) if total else 0
        last = records[0]
        self.card_sessions.set_value(str(total))
        self.card_avg_score.set_value(f"{avg_score}%")
        self.card_last_reps.set_value(str(last.get("reps", "—")))
        self.card_last_pain.set_value(str(last.get("pain_level", "—")))

    @staticmethod
    def _divider():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {CLR_DIVIDER};")
        line.setFixedHeight(1)
        return line


# =============================================================================
#  PART 3: EXERCISE ANALYSIS PAGE  (the camera + metrics view)
# =============================================================================

class AnalysisPage(QWidget):
    """Live camera feed + real-time metrics panel for one exercise."""

    back_requested = Signal()   # User clicked "← Back to Hub"

    def __init__(self, exercise_key: str, parent=None):
        super().__init__(parent)
        self.exercise_key = exercise_key
        self.setObjectName("analysis_page")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            f"background-color: {CLR_BG_DEEP};"
            f" border-bottom: 1px solid {CLR_DIVIDER};"
        )
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)

        btn_back = TransparentPushButton("← Back to Hub")
        btn_back.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 13px;")
        btn_back.clicked.connect(self.back_requested)
        tb_layout.addWidget(btn_back)
        tb_layout.addStretch(1)

        self.lbl_status = QLabel("● OFFLINE")
        self.lbl_status.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        tb_layout.addWidget(self.lbl_status)
        root.addWidget(topbar)

        # ── Content area ──────────────────────────────────────────────
        content = QHBoxLayout()
        content.setContentsMargins(20, 20, 20, 20)
        content.setSpacing(16)

        # --- Video panel ---
        video_card = QFrame()
        video_card.setStyleSheet(
            f"background-color: {CLR_BG_CARD}; border-radius: {CARD_RADIUS};"
            f" border: 1px solid {CLR_DIVIDER};"
        )
        _shadow(video_card)
        v_layout = QVBoxLayout(video_card)
        v_layout.setContentsMargins(12, 12, 12, 12)

        self.video_label = QLabel()
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 6px;")
        v_layout.addWidget(self.video_label)
        content.addWidget(video_card, stretch=3)

        # --- Stats panel ---
        stats_card = QFrame()
        stats_card.setFixedWidth(300)
        stats_card.setStyleSheet(
            f"background-color: {CLR_BG_CARD}; border-radius: {CARD_RADIUS};"
            f" border: 1px solid {CLR_DIVIDER};"
        )
        _shadow(stats_card)
        s_layout = QVBoxLayout(stats_card)
        s_layout.setContentsMargins(20, 22, 20, 22)
        s_layout.setSpacing(18)

        # Exercise title
        self.ex_title = QLabel(self._exercise_label())
        self.ex_title.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 16px; font-weight: 700; background: transparent;"
        )
        s_layout.addWidget(self.ex_title)

        s_layout.addWidget(self._stat_divider())

        # Rep counter
        s_layout.addWidget(self._stat_label("TOTAL REPS"))
        self.rep_val = QLabel("0")
        self.rep_val.setAlignment(Qt.AlignCenter)
        self.rep_val.setStyleSheet(
            f"color: {CLR_ACCENT}; font-size: 56px; font-weight: 800; background: transparent;"
        )
        s_layout.addWidget(self.rep_val)

        s_layout.addWidget(self._stat_divider())

        # Form quality
        s_layout.addWidget(self._stat_label("FORM QUALITY"))
        self.score_val = QLabel("—")
        self.score_val.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 18px; font-weight: 700; background: transparent;"
        )
        s_layout.addWidget(self.score_val)

        self.score_bar = ProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {CLR_DIVIDER}; border-radius: 4px; height: 8px; }}"
            f"QProgressBar::chunk {{ background-color: {CLR_ACCENT}; border-radius: 4px; }}"
        )
        s_layout.addWidget(self.score_bar)

        s_layout.addWidget(self._stat_divider())

        # AI Feedback
        s_layout.addWidget(self._stat_label("AI FEEDBACK"))
        self.feedback_lbl = QLabel("Waiting for analysis...")
        self.feedback_lbl.setWordWrap(True)
        self.feedback_lbl.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 13px; background: transparent;"
        )
        s_layout.addWidget(self.feedback_lbl)

        s_layout.addStretch(1)

        # Action button
        self.btn_action = PrimaryPushButton("▶  START SESSION")
        self.btn_action.setMinimumHeight(46)
        self.btn_action.setStyleSheet(
            f"QPushButton {{ background-color: {CLR_ACCENT}; color: white;"
            f" font-size: 13px; font-weight: 600; border-radius: 8px; }}"
            f"QPushButton:hover {{ background-color: #1A6FB0; }}"
        )
        s_layout.addWidget(self.btn_action)

        content.addWidget(stats_card, stretch=0)
        root.addLayout(content)

    # ── Helpers ───────────────────────────────────────────────────────

    def _exercise_label(self) -> str:
        return {
            "squat":    "Deep Squat Analysis",
            "sts": "Sit-to-Stand Analysis",
            "lunge":    "Lunge Analysis",
            "shoulder": "Shoulder Press Analysis",
        }.get(self.exercise_key, "Exercise Analysis")

    @staticmethod
    def _stat_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 10px; font-weight: 600;"
            " letter-spacing: 1px; background: transparent;"
        )
        return lbl

    @staticmethod
    def _stat_divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {CLR_DIVIDER}; border: none;")
        return line

    # ── Public update slots (called from PhysioDashboard) ─────────────

    @Slot(QImage)
    def update_video(self, img: QImage):
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    @Slot(dict)
    def update_metrics(self, data: dict):
        self.rep_val.setText(str(data["reps"]))
        score = data["score"]
        self.score_val.setText(f"{score}%")
        self.score_bar.setValue(score)
        self.feedback_lbl.setText(data["feedback"])

        if score > 85:
            chunk_color = CLR_GOOD
        elif score > 60:
            chunk_color = CLR_WARN
        else:
            chunk_color = CLR_CRIT

        self.score_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {CLR_DIVIDER}; border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background-color: {chunk_color}; border-radius: 4px; }}"
        )

    @Slot(str, str)
    def update_status(self, text: str, hex_color: str):
        dot = "●"
        self.lbl_status.setText(f"{dot} {text}")
        self.lbl_status.setStyleSheet(
            f"color: {hex_color}; font-size: 12px; font-weight: 600; background: transparent;"
        )


# =============================================================================
#  PART 4: PATIENT RECORDS PAGE
# =============================================================================

class RecordsPage(ScrollArea):
    """Scrollable history of all past sessions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget()
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("records_page")
        self.view.setStyleSheet("background-color: transparent;")
        self.setStyleSheet("background-color: transparent; border: none;")

        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setContentsMargins(30, 30, 30, 30)
        self.v_layout.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title = TitleLabel("Patient History")
        hdr.addWidget(title)
        hdr.addStretch(1)
        self.count_lbl = CaptionLabel("0 sessions")
        self.count_lbl.setStyleSheet(f"color: {CLR_TEXT_SEC};")
        hdr.addWidget(self.count_lbl)
        self.v_layout.addLayout(hdr)

        self.v_layout.addWidget(self._divider())

        self.history_layout = QVBoxLayout()
        self.history_layout.setSpacing(10)
        self.v_layout.addLayout(self.history_layout)
        self.v_layout.addStretch(1)

        self._count = 0

    def add_record(self, report: dict):
        title_txt  = f"Session — {report.get('date', 'Unknown')}"
        score      = report.get('avg_score', 0)
        reps       = report.get('reps', 0)
        pain       = report.get('pain_level', '—')
        content_txt = f"Reps: {reps}  |  Form Score: {score}%  |  Pain Level: {pain}"

        card = ExpandGroupSettingCard(icon=FIF.HISTORY, title=title_txt, content=content_txt)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setContentsMargins(20, 10, 20, 10)

        details = report.get("details", [])
        if details:
            for rep in details:
                row = QHBoxLayout()
                lbl_rep   = StrongBodyLabel(f"Rep {rep['rep_num']}:")
                lbl_issue = BodyLabel(rep["issue"])
                lbl_score = BodyLabel(f"{rep['score']}%")
                color = CLR_GOOD if rep["score"] >= 70 else CLR_CRIT
                lbl_issue.setStyleSheet(f"color: {color};")
                row.addWidget(lbl_rep)
                row.addWidget(lbl_issue)
                row.addStretch(1)
                row.addWidget(lbl_score)
                detail_layout.addLayout(row)

                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet(f"color: {CLR_DIVIDER};")
                detail_layout.addWidget(sep)
        else:
            lbl = BodyLabel("No per-rep breakdown available for this session.")
            lbl.setStyleSheet(f"color: {CLR_TEXT_SEC};")
            detail_layout.addWidget(lbl)

        card.addGroupWidget(detail_widget)
        self.history_layout.insertWidget(0, card)

        self._count += 1
        self.count_lbl.setText(f"{self._count} session{'s' if self._count != 1 else ''}")

    @staticmethod
    def _divider():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {CLR_DIVIDER};")
        return line


# =============================================================================
#  PART 5: SETTINGS PAGE
# =============================================================================

class SettingsPage(ScrollArea):
    """User biometrics, system toggles, and developer console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget()
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setObjectName("settings_page")
        self.view.setStyleSheet("background-color: transparent;")
        self.setStyleSheet("background-color: transparent; border: none;")

        layout = QVBoxLayout(self.view)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        layout.addWidget(TitleLabel("System Configuration"))

        # ── Biometrics ────────────────────────────────────────────────
        bio_card = CardWidget()
        bio_layout = QVBoxLayout(bio_card)
        bio_layout.addWidget(StrongBodyLabel("User Biometrics"))
        bio_layout.addWidget(BodyLabel("Used for biomechanical calibration and form tolerance."))

        h_row = QHBoxLayout()
        self.height_input = DoubleSpinBox()
        self.height_input.setRange(100, 250)
        self.height_input.setValue(state.USER_HEIGHT_CM)
        self.height_input.valueChanged.connect(lambda v: setattr(state, "USER_HEIGHT_CM", v))
        h_row.addWidget(QLabel("Height (cm):"))
        h_row.addWidget(self.height_input)
        bio_layout.addLayout(h_row)

        w_row = QHBoxLayout()
        self.weight_input = DoubleSpinBox()
        self.weight_input.setRange(40, 200)
        self.weight_input.setValue(state.USER_WEIGHT_KG)
        self.weight_input.valueChanged.connect(lambda v: setattr(state, "USER_WEIGHT_KG", v))
        w_row.addWidget(QLabel("Weight (kg):"))
        w_row.addWidget(self.weight_input)
        bio_layout.addLayout(w_row)
        layout.addWidget(bio_card)

        # ── AR / Voice toggles ────────────────────────────────────────
        toggle_card = CardWidget()
        tog_layout = QVBoxLayout(toggle_card)

        def toggle_row(label, sublabel, initial, callback):
            row = QHBoxLayout()
            txt = QVBoxLayout()
            txt.addWidget(StrongBodyLabel(label))
            txt.addWidget(BodyLabel(sublabel))
            row.addLayout(txt)
            row.addStretch(1)
            sw = SwitchButton()
            sw.setOnText("ON");  sw.setOffText("OFF")
            sw.setChecked(initial)
            sw.checkedChanged.connect(callback)
            row.addWidget(sw)
            tog_layout.addLayout(row)
            return sw

        toggle_row("Holographic AR Guidance",
                   "Projects a floor target for foot placement.",
                   state.AR_MODE,
                   lambda v: setattr(state, "AR_MODE", v))

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {CLR_DIVIDER};"); tog_layout.addWidget(sep)

        toggle_row("AI Voice Assistant",
                   "Speaks real-time coaching cues during reps.",
                   state.VOICE_ON,
                   lambda v: setattr(state, "VOICE_ON", v))

        layout.addWidget(toggle_card)

        # ── Developer console ─────────────────────────────────────────
        dev_card = CardWidget()
        dev_layout = QHBoxLayout(dev_card)
        txt = QVBoxLayout()
        txt.addWidget(StrongBodyLabel("Developer Console"))
        txt.addWidget(BodyLabel("Live parameter tuning for pose thresholds."))
        dev_layout.addLayout(txt)
        dev_layout.addStretch(1)
        btn_dev = PrimaryPushButton("Open Console")
        btn_dev.clicked.connect(self._open_dev_console)
        dev_layout.addWidget(btn_dev)
        layout.addWidget(dev_card)

        layout.addStretch(1)

    def _open_dev_console(self):
        self._dev_win = DeveloperToolsWindow()
        self._dev_win.show()


# =============================================================================
#  PART 6: DEVELOPER TOOLS WINDOW
# =============================================================================

class DeveloperToolsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Physio-Vision — Developer Console")
        self.resize(360, 420)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Window)
        self.setStyleSheet(f"background-color: {CLR_BG_DEEP}; color: {CLR_TEXT_PRI};")

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(StrongBodyLabel("Real-Time Threshold Tuning"))

        def add_tuner(label, val, lo, hi, attr):
            row = QHBoxLayout()
            row.addWidget(BodyLabel(label))
            spin = DoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(val)
            spin.valueChanged.connect(lambda v: setattr(state, attr, v))
            row.addWidget(spin)
            layout.addLayout(row)

        add_tuner("Squat Depth (°)",       state.PARAM_SQUAT_DEPTH,  90,  170, "PARAM_SQUAT_DEPTH")
        add_tuner("Stand-Up Threshold (°)", state.PARAM_UP_THRESHOLD, 150, 180, "PARAM_UP_THRESHOLD")
        add_tuner("Lean Warning (°)",       state.PARAM_LEAN_WARN,    10,  80,  "PARAM_LEAN_WARN")
        add_tuner("Lean Critical (°)",      state.PARAM_LEAN_CRIT,    20,  90,  "PARAM_LEAN_CRIT")
        add_tuner("Back Rounding (°)",      state.PARAM_ROUNDING,     5,   45,  "PARAM_ROUNDING")

        layout.addStretch(1)
        note = BodyLabel("⚡ Changes apply to the next processed frame.")
        note.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 11px;")
        layout.addWidget(note)


# =============================================================================
#  PART 7: PAIN SCALE DIALOG
# =============================================================================

class PainScaleDialog(MessageBoxBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel("Session Complete — Pain Assessment", self)

        self.pain_map = {
            0:  ("01.PNG",  "0: No pain. Completely normal."),
            1:  ("01.PNG",  "1: Very mild. Barely noticeable."),
            2:  ("23.PNG",  "2: Minor. A nagging annoyance."),
            3:  ("23.PNG",  "3: Mild. Noticeable during movement."),
            4:  ("45.PNG",  "4: Moderate. Interferes with form."),
            5:  ("45.PNG",  "5: Distracting. Significant discomfort."),
            6:  ("67.PNG",  "6: Severe. Limits rep completion."),
            7:  ("67.PNG",  "7: Very severe. Intense and disabling."),
            8:  ("89.PNG",  "8: Extreme. Hard to bear weight."),
            9:  ("89.PNG",  "9: Unbearable. Crying out."),
            10: ("10.PNG",  "10: Worst imaginable. STOP NOW."),
        }

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(160, 160)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            f"background-color: {CLR_BG_CARD}; border-radius: 8px; border: 1px solid {CLR_DIVIDER};"
        )

        self.guide_label = BodyLabel(self.pain_map[0][1], self)
        self.guide_label.setWordWrap(True)
        self.guide_label.setMinimumHeight(45)

        self.slider = Slider(Qt.Horizontal, self)
        self.slider.setRange(0, 10)
        self.slider.setValue(0)

        self.valLabel = TitleLabel("0", self)
        self.valLabel.setAlignment(Qt.AlignCenter)

        content = QHBoxLayout()
        content.setSpacing(20)
        img_col = QVBoxLayout()
        img_col.addWidget(self.image_label)
        img_col.addStretch(1)
        right_col = QVBoxLayout()
        right_col.addWidget(self.guide_label)
        right_col.addSpacing(15)
        right_col.addWidget(self.slider)
        right_col.addWidget(self.valLabel)
        right_col.addStretch(1)
        content.addLayout(img_col)
        content.addLayout(right_col)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addSpacing(15)
        self.viewLayout.addLayout(content)

        self.slider.valueChanged.connect(self._update)
        self._update(0)

        self.yesButton.setText("Save Session")
        self.cancelButton.setText("Discard")
        self.widget.setMinimumWidth(550)

    def _update(self, val: int):
        if val <= 3:
            color = CLR_GOOD
        elif val <= 6:
            color = CLR_WARN
        else:
            color = CLR_CRIT
        self.valLabel.setText(str(val))
        self.valLabel.setStyleSheet(
            f"color: {color}; font-size: 54px; font-weight: bold;"
        )
        img_name, text = self.pain_map[val]
        self.guide_label.setText(text)
        img_path = os.path.join("pain_imgs", "Squat", img_name)
        if os.path.exists(img_path):
            pix = QPixmap(img_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pix)
        else:
            self.image_label.setText("Image\nNot Found")
            self.image_label.setStyleSheet("color: red; background: transparent;")


# =============================================================================
#  PART 8: MAIN DASHBOARD WINDOW
# =============================================================================

class PhysioDashboard(FluentWindow):
    """Top-level application window.
    Manages navigation between Hub, Analysis, Records, and Settings."""
    history_loaded = Signal(list)

    def __init__(self, username: str):
        super().__init__()
        self.current_user = username
        self.setWindowTitle(f"Physio-Vision  |  {self.current_user}")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 700)

        # ── Background worker ──────────────────────────────────────────
        self.worker = VisionWorker()
        self.worker.frame_processed.connect(self._on_frame)
        self.worker.stats_update.connect(self._on_stats)
        self.worker.system_status.connect(self._on_status)
        self.worker.session_finished.connect(self._on_session_finish)
        self.history_loaded.connect(self._on_history_loaded)

        # ── Pages ──────────────────────────────────────────────────────
        self.hub_page      = HubPage(username)
        self.hub_page.setObjectName("hub_page")
        self.hub_page.exercise_selected.connect(self._launch_exercise)

        self.analysis_page = AnalysisPage("squat")   # default; replaced on launch
        self.analysis_page.setObjectName("analysis_page")
        self.analysis_page.back_requested.connect(self._return_to_hub)
        self.analysis_page.btn_action.clicked.connect(self._toggle_session)

        self.records_page  = RecordsPage()
        self.settings_page = SettingsPage()

        # ── Navigation ─────────────────────────────────────────────────
        self.addSubInterface(self.hub_page,      FIF.HOME,    "Hub")
        self.addSubInterface(self.analysis_page, FIF.VIDEO,   "Live Analysis")
        self.addSubInterface(self.records_page,  FIF.HEART,   "Patient Records")
        self.addSubInterface(self.settings_page, FIF.SETTING, "Settings",
                             NavigationItemPosition.BOTTOM)

        # ── Load cloud history after a short delay ─────────────────────
        QTimer.singleShot(300, self._fetch_cloud_history)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _launch_exercise(self, exercise_key: str):
        """Hub tile clicked → switch to analysis page for that exercise."""
        # 1. Update the background key
        self.analysis_page.exercise_key = exercise_key

        # 2. FORCE the UI label to update its text
        self.analysis_page.ex_title.setText(self.analysis_page._exercise_label())

        # 3. Switch nav to the analysis tab
        self.switchTo(self.analysis_page)

    def _return_to_hub(self):
        """Back button on analysis page → stop any running session, go to hub."""
        if self.worker.isRunning():
            self.worker.stop()
            self.analysis_page.btn_action.setText("▶  START SESSION")
        self.switchTo(self.hub_page)

    def _toggle_session(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.analysis_page.btn_action.setText("▶  START SESSION")
            self.analysis_page.update_status("OFFLINE", CLR_TEXT_SEC)
        else:
            # Tell the engine exactly which logic track to run
            self.worker.exercise_mode = self.analysis_page.exercise_key
            self.worker.start()
            self.analysis_page.btn_action.setText("⬛  STOP SESSION")

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @Slot(QImage)
    def _on_frame(self, img: QImage):
        self.analysis_page.update_video(img)

    @Slot(dict)
    def _on_stats(self, data: dict):
        self.analysis_page.update_metrics(data)

    @Slot(str, str)
    def _on_status(self, text: str, color: str):
        self.analysis_page.update_status(text, color)

    @Slot(dict)
    def _on_session_finish(self, report: dict):
        """Session ended → show pain dialog → save locally + to cloud."""
        dialog = PainScaleDialog(self)
        if dialog.exec():
            pain_score = dialog.slider.value()
            report["pain_level"] = pain_score
            self.records_page.add_record(report)

            # Cloud save
            try:
                payload = {
                    "username":  self.current_user,
                    "exercise":  self.analysis_page.exercise_key,
                    "reps":      report["reps"],
                    "score":     report["avg_score"],
                    "pain_level": pain_score,
                }
                resp = requests.post(f"{API_URL}/log_session", json=payload)
                if resp.status_code == 200:
                    InfoBar.success(
                        title="Cloud Sync",
                        content="Session saved to database.",
                        orient=Qt.Horizontal, isClosable=True,
                        position=InfoBarPosition.TOP_RIGHT, parent=self
                    )
                else:
                    InfoBar.warning(title="Sync Failed",
                                    content="Could not save to database.", parent=self)
            except requests.exceptions.RequestException:
                InfoBar.error(title="Network Error",
                              content="Cannot reach the server.", parent=self)

        # Reset analysis UI
        self.analysis_page.btn_action.setText("▶  START SESSION")
        self.analysis_page.update_status("OFFLINE", CLR_TEXT_SEC)

    # ------------------------------------------------------------------
    # Cloud history loader
    # ------------------------------------------------------------------

    def _fetch_cloud_history(self):
        def _fetch():
            try:
                resp = requests.get(f"{API_URL}/get_history/{self.current_user}")
                if resp.status_code == 200:
                    records = resp.json().get("history", [])
                    # SAFELY emit the data over the bridge to the main thread
                    self.history_loaded.emit(records)
            except requests.exceptions.RequestException:
                print("[Physio-Vision] Offline mode — could not reach server.")

        threading.Thread(target=_fetch, daemon=True).start()

    @Slot(list)
    def _on_history_loaded(self, records: list):
        """This safely receives the data on the main UI thread and updates the screen."""
        self.hub_page.update_kpis(records)

        # Populate records page (oldest first so newest ends on top)
        for row in reversed(records):
            report = {
                "date": row.get("date", "Unknown"),
                "reps": row.get("reps", 0),
                "avg_score": row.get("score", 0),
                "pain_level": row.get("pain_level", "—"),
                "details": [],
            }
            self.records_page.add_record(report)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# =============================================================================
#  ENTRY POINT  (called from main.py / test.py)
# =============================================================================

def run_application():
    """Bootstrap: Splash (subprocess) → Login → PhysioDashboard.

    play_splash() runs BEFORE QApplication is created so cv2 and Qt
    never share the same process at startup — eliminating all crashes.
    """
    # 1. Play the splash video in a completely separate process.
    #    This call blocks until the video finishes, then returns.
    from auth import play_splash
    play_splash()

    # 2. Now it is safe to start Qt.
    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    app.setQuitOnLastWindowClosed(False)

    from auth import LoginWindow
    login_window = LoginWindow()
    _dashboard_ref = []   # list used as mutable closure cell

    def launch_dashboard(username: str):
        print(f"[Physio-Vision] Building dashboard for: {username}")
        setTheme(Theme.DARK)
        dashboard = PhysioDashboard(username)
        _dashboard_ref.append(dashboard)   # keep alive
        dashboard.show()
        login_window.close()
        app.setQuitOnLastWindowClosed(True)

    def on_login(username: str):
        QTimer.singleShot(100, lambda: launch_dashboard(username))

    login_window.login_successful.connect(on_login)
    login_window.show()
    login_window.setWindowState(login_window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    login_window.raise_()
    login_window.activateWindow()
    sys.exit(app.exec())