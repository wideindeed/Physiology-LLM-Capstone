import sys
import os
import subprocess
import requests

import socket
import webbrowser
from urllib.parse import urlparse, parse_qs
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFrame, QStackedWidget,
                             QGraphicsOpacityEffect)

from qfluentwidgets import (LineEdit, PrimaryPushButton, PushButton,
                            InfoBar, setTheme, Theme)

# ---------------------------------------------------------------------------
# API URL
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL")
if API_URL is None:
    raise EnvironmentError(
        "API_URL is not set.\n"
        "Windows CMD:   set API_URL=https://your-tunnel-url.com/\n"
        "PyCharm:       Run > Edit Configurations > Environment variables"
    )


class GoogleAuthThread(QThread):
    """An invisible background server that waits for the Google tokens."""
    auth_success = pyqtSignal(dict)

    def run(self):
        try:
            # 1. Spin up a local server on port 54321
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 54321))
            server.listen(1)

            # 2. Open the user's web browser to your API
            webbrowser.open("https://api.physiovision.app/auth/google/login?client=desktop")

            # 3. Wait for the browser to redirect back to us (halts this thread until it happens)
            client, addr = server.accept()
            request = client.recv(2048).decode()

            # 4. Parse the URL parameters the API sent us
            first_line = request.split('\n')[0]
            url = first_line.split(' ')[1]
            parsed_url = urlparse(url)
            params = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}

            # 5. Send a friendly message to the browser so it doesn't look broken
            http_response = """HTTP/1.1 200 OK\nContent-Type: text/html\n\n
            <html><body style="background:#18181A; color:#3DAA6E; font-family:sans-serif; text-align:center; padding-top:100px;">
            <h2>Authentication successful!</h2><p>You can close this tab and return to the Physio-Vision app.</p>
            <script>window.close();</script>
            </body></html>"""
            client.sendall(http_response.encode())

            # Shut down the invisible server
            client.close()
            server.close()

            # Pass the data back to the main UI
            self.auth_success.emit(params)

        except Exception as e:
            print(f"Google Auth Error: {e}")


# ---------------------------------------------------------------------------
# Asset paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
SPLASH_VIDEO = os.path.join(_HERE, "startup_content", "eye.mp4")
LOGO_PATH = os.path.join(_HERE, "startup_content", "Physiologo.PNG")
PLAYER_SCRIPT = os.path.join(_HERE, "splash_player.py")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
CLR_BG = "#FFFFFF"
CLR_BG_INPUT = "#F5F7FA"
CLR_BORDER = "#DDE1E7"
CLR_BORDER_FOCUS = "#1D7EC2"
CLR_TEXT_PRI = "#0D1117"
CLR_TEXT_SEC = "#6B7280"
CLR_ACCENT = "#1D7EC2"
CLR_ACCENT_DARK = "#155E9A"
CLR_TAB_ACTIVE = "#0D1117"
CLR_TAB_INACTIVE = "#9CA3AF"


# =============================================================================
#  SPLASH — runs BEFORE Qt starts, purely via subprocess
# =============================================================================

def play_splash():
    """
    Plays eye.mp4 in a plain cv2 window via a subprocess.
    Blocks until the video is done. Safe to call before QApplication exists.
    If the player script or video is missing, returns instantly.
    """
    if not os.path.exists(PLAYER_SCRIPT):
        return
    if not os.path.exists(SPLASH_VIDEO):
        return

    try:
        subprocess.run(
            [sys.executable, PLAYER_SCRIPT, SPLASH_VIDEO],
            timeout=60  # safety cap: never block more than 60 s
        )
    except Exception:
        pass  # if anything goes wrong, just continue to login


# =============================================================================
#  LOGIN WINDOW
# =============================================================================

class _TabBar(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._tabs = []
        for idx, text in enumerate(("Sign In", "Register")):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setFixedHeight(38)
            lbl._idx = idx
            lbl.installEventFilter(self)
            layout.addWidget(lbl)
            self._tabs.append(lbl)
        self._refresh()

    def _refresh(self):
        for lbl in self._tabs:
            active = (lbl._idx == self._current)
            lbl.setStyleSheet(
                f"color: {CLR_TAB_ACTIVE if active else CLR_TAB_INACTIVE};"
                f" font-size: 14px; font-weight: {'700' if active else '400'};"
                " font-family: 'Segoe UI', sans-serif;"
                f" border-bottom: {'2px solid ' + CLR_TEXT_PRI if active else '2px solid transparent'};"
                " padding-bottom: 4px; background: transparent;"
            )

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.MouseButtonPress and hasattr(obj, "_idx"):
            if obj._idx != self._current:
                self._current = obj._idx
                self._refresh()
                self.tab_changed.emit(self._current)
        return False


class _Input(LineEdit):
    def __init__(self, placeholder: str, parent=None, password: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        if password:
            self.setEchoMode(LineEdit.Password)
        self.setFixedHeight(42)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {CLR_BG_INPUT};
                border: 1px solid {CLR_BORDER};
                border-radius: 8px;
                padding: 0 14px;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
                color: {CLR_TEXT_PRI};
            }}
            QLineEdit:focus {{
                border: 1.5px solid {CLR_BORDER_FOCUS};
                background-color: #FFFFFF;
            }}
        """)


class _PrimaryBtn(PrimaryPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {CLR_ACCENT};
                color: white;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Segoe UI', sans-serif;
                border: none;
            }}
            QPushButton:hover   {{ background-color: {CLR_ACCENT_DARK}; }}
            QPushButton:pressed {{ background-color: #0F4F7A; }}
        """)


class LoginWindow(QWidget):
    login_successful = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physio-Vision  |  Authentication")
        self.setFixedSize(440, 620)
        self.setStyleSheet(f"background-color: {CLR_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(48, 44, 48, 36)
        root.setSpacing(0)

        # Logo
        logo_lbl = QLabel()
        logo_lbl.setAlignment(Qt.AlignCenter)
        logo_lbl.setFixedHeight(80)
        logo_lbl.setStyleSheet("background: transparent;")
        if os.path.exists(LOGO_PATH):
            pix = QPixmap(LOGO_PATH).scaledToHeight(70, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pix)
        else:
            logo_lbl.setText("PV")
            logo_lbl.setStyleSheet(
                f"color: {CLR_ACCENT}; font-size: 36px; font-weight: 900;"
                " background: transparent;"
            )
        root.addWidget(logo_lbl)
        root.addSpacing(16)

        # Title
        title = QLabel("Physio-Vision")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 22px; font-weight: 700;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
        )
        root.addWidget(title)

        subtitle = QLabel("Motion Intelligence Platform")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px; letter-spacing: 1px;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
        )
        root.addWidget(subtitle)
        root.addSpacing(24)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {CLR_BORDER}; border: none;")
        root.addWidget(div)
        root.addSpacing(18)

        # Tabs
        self._tab_bar = _TabBar()
        root.addWidget(self._tab_bar)
        root.addSpacing(20)

        # Stacked forms
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background: transparent;")
        self._stack.addWidget(self._build_login_form())
        self._stack.addWidget(self._build_register_form())
        root.addWidget(self._stack)
        self._tab_bar.tab_changed.connect(self._stack.setCurrentIndex)

        root.addStretch(1)

        footer = QLabel("Secure connection  ·  Data encrypted in transit")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 10px; background: transparent;"
        )
        root.addWidget(footer)

    def _build_login_form(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.login_user = _Input("Username")
        self.login_pass = _Input("Password", password=True)
        self.login_pass.returnPressed.connect(self.attempt_login)

        btn = _PrimaryBtn("Sign In")
        btn.clicked.connect(self.attempt_login)

        # --- NEW: Google Login Button ---
        self.google_btn = PushButton("Continue with Google")
        self.google_btn.setFixedHeight(44)
        self.google_btn.setStyleSheet("""
            PushButton {
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px;
            }
        """)
        self.google_btn.clicked.connect(self.start_google_login)

        lay.addWidget(self.login_user)
        lay.addWidget(self.login_pass)
        lay.addSpacing(6)
        lay.addWidget(btn)

        # Add a subtle "or" label
        or_lbl = QLabel("or")
        or_lbl.setAlignment(Qt.AlignCenter)
        or_lbl.setStyleSheet(f"color: {CLR_TEXT_SEC}; font-size: 12px; margin: 4px 0;")
        lay.addWidget(or_lbl)

        lay.addWidget(self.google_btn)

        return w

    def _build_register_form(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.reg_user = _Input("Choose a username")
        self.reg_email = _Input("Email address")
        self.reg_pass = _Input("Choose a password", password=True)
        self.reg_confirm = _Input("Confirm password", password=True)

        btn = _PrimaryBtn("Create Account")
        btn.clicked.connect(self.attempt_register)

        note = QLabel("Credentials are stored securely on the Physio server.")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 11px; background: transparent;"
        )

        lay.addWidget(self.reg_user)
        lay.addWidget(self.reg_email)
        lay.addWidget(self.reg_pass)
        lay.addWidget(self.reg_confirm)
        lay.addSpacing(4)
        lay.addWidget(btn)
        lay.addSpacing(8)
        lay.addWidget(note)
        return w

    def attempt_login(self):
        user = self.login_user.text().strip()
        pw = self.login_pass.text().strip()

        if not user or not pw:
            InfoBar.error("", "Please enter both username and password.", parent=self)
            return

        try:
            # 1. VIP Badge Header for Cloudflare Bypass
            headers = {
                "Content-Type": "application/json",
                "X-Desktop-Key": "my_secret_desktop_key_2026"
            }

            # 2. Add empty cf_token to satisfy Pydantic
            payload = {
                "username": user,
                "password": pw,
                "cf_token": ""
            }

            resp = requests.post(f"{API_URL}/login", json=payload, headers=headers)

            if resp.status_code == 200:
                self.login_successful.emit(user)
            else:
                # 3. Crash Fix: Safely parse FastAPI lists into a readable string
                error_detail = resp.json().get("detail", "Invalid credentials.")
                if isinstance(error_detail, list):
                    error_detail = "\n".join([f"• {err.get('loc')[-1]}: {err.get('msg')}" for err in error_detail])

                InfoBar.error("Login Failed", str(error_detail), parent=self)

        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not reach the server.", parent=self)

    def attempt_register(self):
        user = self.reg_user.text().strip()
        email = self.reg_email.text().strip()
        pw = self.reg_pass.text().strip()
        confirm = self.reg_confirm.text().strip()

        if not user or not email or not pw:
            InfoBar.error("", "Please fill in all fields.", parent=self)
            return

        if "@" not in email or "." not in email:
            InfoBar.error("", "Please enter a valid email address.", parent=self)
            return

        if pw != confirm:
            InfoBar.error("", "Passwords do not match.", parent=self)
            return

        try:
            # 1. VIP Badge Header
            headers = {
                "Content-Type": "application/json",
                "X-Desktop-Key": "my_secret_desktop_key_2026"
            }

            # 2. The Strict Payload (Added dummy biometrics & cf_token so it doesn't crash)
            payload = {
                "username": user,
                "email": email,
                "password": pw,
                "first_name": "Desktop",
                "last_name": "User",
                "country": "United States",
                "fitness_level": "beginner",
                "height_cm": 170.0,
                "weight_kg": 70.0,
                "cf_token": ""
            }

            resp = requests.post(f"{API_URL}/register", json=payload, headers=headers)

            if resp.status_code == 201 or resp.status_code == 200:
                InfoBar.success("Verification Sent", "Check your email to activate your account.", parent=self)
                self._tab_bar._current = 0
                self._tab_bar._refresh()
                self._stack.setCurrentIndex(0)
                self.login_user.setText(user)
            else:
                error_detail = resp.json().get("detail", "Registration Failed.")
                if isinstance(error_detail, list):
                    error_detail = "\n".join([f"• {err.get('loc')[-1]}: {err.get('msg')}" for err in error_detail])

                InfoBar.error("Registration Failed", str(error_detail), parent=self)

        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not reach the server.", parent=self)

    # =========================================================================
    # GOOGLE LOGIN HANDLERS
    # =========================================================================
    def start_google_login(self):
        """Triggered when the user clicks the Google Button."""
        InfoBar.info("Browser Opened", "Please complete the login in your web browser.", parent=self)

        # Start the invisible background server
        self.google_thread = GoogleAuthThread()
        self.google_thread.auth_success.connect(self.handle_google_result)
        self.google_thread.start()

    def handle_google_result(self, params):
        """Triggered when the background server catches the data from the Raspberry Pi."""
        if "access" in params:
            # SCENARIO 1: Existing User (Login Successful)
            username = params.get("username", "User")
            InfoBar.success("Success", f"Logged in as {username} via Google!", parent=self)
            self.login_successful.emit(username)

        elif "temp_token" in params:
            # SCENARIO 2: Brand New User (Needs Biometrics)
            InfoBar.warning(
                "Almost there!",
                "Google auth successful. However, you need to complete your profile on the website first.",
                parent=self,
                duration=5000
            )


# =============================================================================
#  STANDALONE TESTING
# =============================================================================

if __name__ == "__main__":
    # Play splash BEFORE Qt starts — no conflict possible
    play_splash()

    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)

    w = LoginWindow()
    w.login_successful.connect(lambda u: (print(f"OK: {u}"), sys.exit()))
    w.show()
    sys.exit(app.exec())