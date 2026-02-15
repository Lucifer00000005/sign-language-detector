import cv2
import mediapipe as mp
import csv

print("Dataset collection started")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

with open("dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(
                frame,
                "Press A-Z or 0-9 to save | ESC to quit",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Collect Dataset", frame)

        key = cv2.waitKey(1) & 0xFF

        if key != 255 and result.multi_hand_landmarks:
            label = chr(key).upper()
            writer.writerow(landmarks + [label])
            print("Saved:", label)

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection ended")
