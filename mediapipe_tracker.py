import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional

class MediapipeTracker:
    """
    Tracks up to 2 hands (Left/Right) and returns a fixed-size feature vector per frame.
    Feature layout (per frame): [LeftHand(21*2), RightHand(21*2)] = 84 values.
    Missing hands are zero-padded. Coordinates are normalized to [0,1] relative to frame size.
    """

    def __init__(self, max_hands: int = 2, detection_conf: float = 0.5, tracking_conf: float = 0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawer = mp.solutions.drawing_utils

    def _extract_hand_xy(self, hand_landmarks, w, h) -> np.ndarray:
        # 21 landmarks, each (x, y), normalized to [0,1]
        pts = []
        for lm in hand_landmarks.landmark:
            pts.append(lm.x)  # normalized by mediapipe already (0..1)
            pts.append(lm.y)
        return np.array(pts, dtype=np.float32)

    def features_from_frame(self, frame_bgr: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Returns:
            features: np.ndarray shape (84,) = [Left(42), Right(42)]
            frame_out: frame with drawings (if draw)
            meta: dict with handedness & confidences
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        meta = {"hands": []}
        left = np.zeros(42, dtype=np.float32)
        right = np.zeros(42, dtype=np.float32)

        if result.multi_hand_landmarks:
            for idx, hand_lms in enumerate(result.multi_hand_landmarks):
                handed = result.multi_handedness[idx].classification[0].label  # "Left" or "Right"
                score = float(result.multi_handedness[idx].classification[0].score)
                meta["hands"].append({"handedness": handed, "score": score})

                if draw:
                    self.drawer.draw_landmarks(
                        frame_bgr, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
                        self.drawer.DrawingSpec(color=(0,255,0) if handed=="Left" else (0,0,255),
                                                thickness=2, circle_radius=2)
                    )

                hand_xy = self._extract_hand_xy(hand_lms, frame_bgr.shape[1], frame_bgr.shape[0])
                if handed == "Left":
                    left = hand_xy
                else:
                    right = hand_xy

        features = np.concatenate([left, right], axis=0)  # (84,)
        return features, frame_bgr, meta

def webcam_demo():
    cap = cv2.VideoCapture(0)
    tracker = MediapipeTracker()
    print("Mediapipe tracker demo — press ESC to exit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats, vis, meta = tracker.features_from_frame(frame, draw=True)
        cv2.putText(vis, f"feat shape: {feats.shape}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Mediapipe Tracker", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_demo()

