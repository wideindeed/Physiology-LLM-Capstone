import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from keras.callbacks import ModelCheckpoint
import os

# --- 1. CONFIG: Updated for STS ---
DATA_DIR = "UIPRMD_STS/fold0"
TIME_STEPS = 88  # From the UI-PRMD JSON for STS
NUM_FEATURES = 66  # 22 joints * 3 dimensions (X, Y, Z)


# --- 2. THE UPGRADED NORMALIZATION (Pelvis Anchor) ---
def normalize_skeleton_sts(data):
    """
    1. Reshape to (Batch, Time, Joints, 3)
    2. Center at Mid-Hip (Joint 0)
    3. Scale so Pelvis Width = 1.0 (Rigid Anchor)
    4. Flatten back
    """
    B, T, F = data.shape
    J = F // 3
    data = data.reshape(B, T, J, 3)

    # Center at Mid-Hip (Joint 0 in UI-PRMD)
    root = data[:, :, 0:1, :]
    data = data - root

    # UI-PRMD Mapping: Joint 14 is Right Hip, Joint 18 is Left Hip
    # We scale by the distance between them (a rigid, unchanging bone)
    left_hip = data[:, :, 18:19, :]
    right_hip = data[:, :, 14:15, :]
    pelvis_width = np.linalg.norm(left_hip - right_hip, axis=3, keepdims=True)

    # Avoid division by zero
    pelvis_width = np.maximum(pelvis_width, 0.0001)

    # Scale the entire skeleton
    data = data / pelvis_width

    return data.reshape(B, T, F)


# --- 3. LOAD & PROCESS ---
def load_data(fold_path):
    x_train = np.load(os.path.join(fold_path, "x_train_fold0.npy"))
    y_train = np.load(os.path.join(fold_path, "y_train_fold0.npy"))
    x_test = np.load(os.path.join(fold_path, "x_test_fold0.npy"))
    y_test = np.load(os.path.join(fold_path, "y_test_fold0.npy"))
    return x_train, y_train, x_test, y_test


print("Loading STS data...")
x_train, y_train, x_test, y_test = load_data(DATA_DIR)

# Ensure Shape is (Batch, Time, Features)
if x_train.shape[1] == NUM_FEATURES:
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

print(f"X_Train Shape: {x_train.shape} | Expected: (Batch, {TIME_STEPS}, {NUM_FEATURES})")

print("Normalizing Training Data with Pelvis Anchor...")
x_train = normalize_skeleton_sts(x_train)
x_test = normalize_skeleton_sts(x_test)

# --- 4. THE NEURAL NETWORK ---
model = Sequential([
    Input(shape=(TIME_STEPS, NUM_FEATURES)),

    # Widened first layer to capture the complex Center of Gravity shift
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Outputs final score between 0 and 1
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print("Starting STS Robust Training...")
checkpoint_path = os.path.join(os.path.dirname(__file__), 'sit_to_stand_robust.keras')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mae', save_best_only=True, mode='min')

model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint]
)

print("Training Complete. Model saved as 'sit_to_stand_robust.keras'")