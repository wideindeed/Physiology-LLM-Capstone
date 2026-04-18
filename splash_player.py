import cv2
import sys
import ctypes


def get_screen_resolution():
    """Grab the resolution of the primary monitor using Windows API."""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def play_video():
    if len(sys.argv) < 2:
        return

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    # --- 1. SET YOUR DIMENSIONS ---
    POPUP_WIDTH = 500
    POPUP_HEIGHT = 500

    # --- 2. CALCULATE CENTER ---
    screen_w, screen_h = get_screen_resolution()
    center_x = (screen_w // 2) - (POPUP_WIDTH // 2)
    center_y = (screen_h // 2) - (POPUP_HEIGHT // 2)

    # --- 3. CREATE & TELEPORT WINDOW ---
    win_name = "Physio-Vision Startup"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Optional: Remove borders for a cleaner splash feel
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Move it to the calculated center
    cv2.moveWindow(win_name, center_x, center_y)
    cv2.resizeWindow(win_name, POPUP_WIDTH, POPUP_HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (POPUP_WIDTH, POPUP_HEIGHT))
        cv2.imshow(win_name, frame)

        # Plays at ~30 FPS. Press 'ESC' to skip.
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    play_video()