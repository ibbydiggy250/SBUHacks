import mediapipe as mp
import cv2
import time
import csv
import numpy as np

class HandDetector:
    def __init__(self, max_hands=2, detection_conf=0.5, tracking_conf=0.5, debug=True):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawer = mp.solutions.drawing_utils
        self.prev_time = 0
        self.debug = debug

    # ---------------------- HAND DETECTION ----------------------
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        all_hands = []

        if results.multi_hand_landmarks:
            for idx, hand in enumerate(results.multi_hand_landmarks):
                # Identify left/right hand
                label = results.multi_handedness[idx].classification[0].label
                score = results.multi_handedness[idx].classification[0].score

                # Draw landmarks
                color = (0, 255, 0) if label == "Left" else (0, 0, 255)
                self.drawer.draw_landmarks(
                    frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                    self.drawer.DrawingSpec(color=color, thickness=2, circle_radius=2)
                )

                # Extract hand coordinates
                h, w, _ = frame.shape
                hand_points = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
                all_hands.append({"label": label, "score": score, "points": hand_points})

        # Add FPS display
        frame = self.display_fps(frame)

        if self.debug:
            print(f"Detected {len(all_hands)} hand(s)")

        return all_hands, frame

    # ---------------------- FPS DISPLAY ----------------------
    def display_fps(self, frame):
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    

    # ---------------------- SAVE LANDMARKS ----------------------
    def save_landmarks(self, hand_points, label="A", filename="dataset.csv"):
        """Save one frame of landmarks for training."""
        flat = [coord for point in hand_points for coord in point]
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + flat)
        print(f"✅ Saved landmarks for label '{label}' to {filename}")


# ---------------------- MAIN LOOP ----------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(debug=False)

    print("🖐️ ASL Mediapipe Detector Running")
    print("➡️  Press any letter key (A–Z) to save that sign")
    print("➡️  Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera feed detected.")
            break

        # Mirror the webcam so it feels natural
        frame = cv2.flip(frame, 1)

        hands, frame = detector.detect(frame)
        cv2.imshow("ASL Hand Detection", frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # ESC to quit
        if key == 27:
            break

        # Detect letter keys (A–Z or a–z)
        elif 65 <= key <= 90 or 97 <= key <= 122:
            if hands:
                label = chr(key).upper()
                detector.save_landmarks(hands[0]["points"], label=label)
                print(f"✅ Captured {label}")
            else:
                print("❌ No hand detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()
