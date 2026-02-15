import cv2
import mediapipe as mp
import os
import csv

DATA_DIR = DATA_DIR = "Data/processed_combine_asl_dataset"
OUTPUT_CSV = "dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

print("Starting landmark extraction...")

count = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (224, 224))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    row.append(label)
                    writer.writerow(row)
                    count += 1

print("Extraction done")
print("Total samples saved:", count)
