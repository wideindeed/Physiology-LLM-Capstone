import sys
import requests
import os
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QHBoxLayout
from qfluentwidgets import (LineEdit, PrimaryPushButton, PushButton,
                            TitleLabel, InfoBar, setTheme, Theme, CardWidget)


API_URL = os.environ.get("API_URL")

if API_URL is None:
    raise EnvironmentError(
        "API_URL environment variable is not set!\n"
        "Please set it before running the app:\n"
        "   Windows Command Prompt:   set API_URL=https://your-tunnel-url.com/\n"
        "   PyCharm: Run → Edit Configurations → Environment variables"
    )

class LoginWindow(QWidget):
    # This signal acts as a "wire" to send the username back to your main app upon success
    login_successful = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physio-Vision | Authentication")
        self.resize(400, 350)
        self.setStyleSheet("background-color: #2b2b2b;")  # Matches your dark theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        # The central card that holds the login form
        self.card = CardWidget(self)
        card_layout = QVBoxLayout(self.card)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(30, 40, 30, 40)

        self.title = TitleLabel("Physio-Vision", self)
        self.title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.title)

        self.username_input = LineEdit(self)
        self.username_input.setPlaceholderText("Username")
        card_layout.addWidget(self.username_input)

        self.password_input = LineEdit(self)
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(LineEdit.Password)  # Hides the typing
        card_layout.addWidget(self.password_input)

        button_layout = QHBoxLayout()
        self.btn_register = PushButton("Register Account", self)
        self.btn_register.clicked.connect(self.attempt_register)

        self.btn_login = PrimaryPushButton("Login", self)
        self.btn_login.clicked.connect(self.attempt_login)

        button_layout.addWidget(self.btn_register)
        button_layout.addWidget(self.btn_login)
        card_layout.addLayout(button_layout)

        layout.addWidget(self.card)

    def attempt_login(self):
        user = self.username_input.text().strip()
        pw = self.password_input.text().strip()

        if not user or not pw:
            InfoBar.error("Input Error", "Please enter both username and password.", parent=self)
            return

        try:
            response = requests.post(f"{API_URL}/login", json={"username": user, "password": pw})
            if response.status_code == 200:
                self.login_successful.emit(user)  # Send the username out!

            else:
                InfoBar.error("Login Failed", response.json().get("detail", "Invalid credentials."), parent=self)
        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not connect to the Raspberry Pi server.", parent=self)

    def attempt_register(self):
        user = self.username_input.text().strip()
        pw = self.password_input.text().strip()

        if not user or not pw:
            InfoBar.error("Input Error", "Please enter both username and password.", parent=self)
            return

        try:
            response = requests.post(f"{API_URL}/register", json={"username": user, "password": pw})
            if response.status_code == 200:
                InfoBar.success("Success", "Account created! You can now click Login.", parent=self)
            else:
                InfoBar.error("Registration Failed", response.json().get("detail", "Username might be taken."),
                              parent=self)
        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not connect to the Raspberry Pi server.", parent=self)


# --- STANDALONE TESTING ---
# This block allows you to run this file by itself just to test the database connection
if __name__ == '__main__':
    app = QApplication(sys.argv)
    setTheme(Theme.DARK)


    # Simple function to print success when testing standalone
    def on_success(username):
        print(f"SUCCESS! Logged in as: {username}")
        sys.exit()


    w = LoginWindow()
    w.login_successful.connect(on_success)
    w.show()
    sys.exit(app.exec())