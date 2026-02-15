import traceback
import cv2
import mediapipe as mp
import numpy as np
import joblib   # ✅ USE JOBLIB (IMPORTANT)
import sys

print("Starting program...")

# ================= LOAD MODEL =================
print("Loading model...")

try:
    model = joblib.load("sign_model.pkl")  # ✅ FIXED
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed")
    traceback.print_exc()
    sys.exit(1)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= CAMERA =================
print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera NOT opened")
    sys.exit(1)

print("✅ Camera opened")

# ================= MAIN LOOP =================
while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_letter = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ===== EXTRACT 63 LANDMARK VALUES =====
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data = np.array(landmarks).reshape(1, -1)

                prediction = model.predict(data)
                predicted_letter = prediction[0]

                print("Predicted:", predicted_letter)

    # ===== DISPLAY LETTER ON SCREEN =====
    cv2.putText(
        frame,
        f"Prediction: {predicted_letter}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("ESC pressed, exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended cleanly")
