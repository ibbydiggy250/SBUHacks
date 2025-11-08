import os
import time
import json
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from .mediapipe_tracker import MediapipeTracker

# Map number keys to phrases for easy labeling during collection
PHRASES = {
    ord('1'): "hello",
    ord('2'): "yes",
    ord('3'): "thank_you",
    ord('4'): "im_happy",
    ord('5'): "see_you_soon",
}

SEQ_LEN = 30  # frames per sequence (~1 sec at ~30fps)
SAVE_DIR = os.path.join("data", "samples")

def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)

def collect():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    tracker = MediapipeTracker()
    buffer = deque(maxlen=SEQ_LEN)

    print("Phrase Collector")
    print("Controls:")
    print("  Press 1: hello")
    print("  Press 2: yes")
    print("  Press 3: thank_you")
    print("  Press 4: im_happy")
    print("  Press 5: see_you_soon")
    print("  Press ESC to quit.")
    print("Hold the gesture for ~1 second before pressing the key.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        feats, vis, meta = tracker.features_from_frame(frame, draw=True)
        buffer.append(feats)

        # UI
        cv2.putText(vis, "Press 1-5 to save sequence", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        for k, p in PHRASES.items():
            cv2.putText(vis, f"{chr(k)}:{p}", (10, 60 + 25*list(PHRASES.keys()).index(k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

        cv2.imshow("Phrase Collector", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key in PHRASES and len(buffer) == SEQ_LEN:
            label = PHRASES[key]
            seq = np.stack(list(buffer), axis=0)  # (T, 84)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = os.path.join(SAVE_DIR, f"{label}_{timestamp}.npz")
            np.savez_compressed(out_path, seq=seq, label=label)
            print(f"✅ Saved {label} sequence -> {out_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect()

