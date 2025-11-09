"""
Main ASL Translator Application - Orchestrates all components
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from typing import Optional, Tuple, List

from asl_database import ASLDatabase
from similarity_matcher import SimilarityMatcher
from api_integrations import ElevenLabsAPI

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class ASLTranslatorApp:
    def __init__(
        self,
        similarity_metric: str = 'combined',
        confidence_threshold: float = 0.3,
        elevenlabs_api_key: Optional[str] = None
    ):
        """
        Initialize ASL Translator Application
        Args:
            similarity_metric: 'euclidean', 'cosine', 'dtw', or 'combined'
            confidence_threshold: Minimum confidence for sign matching
            elevenlabs_api_key: ElevenLabs API key (optional, for TTS)
        """
        # MediaPipe setup
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Database and matcher
        self.database = ASLDatabase(db_path="asl_database_gloss.json")
        self.matcher = SimilarityMatcher(metric=similarity_metric)
        self.confidence_threshold = confidence_threshold

        self.np_database = self.database.get_numpy_database()
        print(f"[INFO] Cached {len(self.np_database)} signs for fast matching")
        
        # API client (optional TTS)
        self.elevenlabs = ElevenLabsAPI(api_key=elevenlabs_api_key) if elevenlabs_api_key else None
        
        # State management
        self.current_phrase = []
        self.phrase_history = []
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.prediction_queue = deque(maxlen=10)
        self.stability_threshold = 0.7
        
        # Audio cache
        self.audio_cache = {}
    
    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """Extract normalized landmarks from hand"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks)
    
    def process_frame(self, frame) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Process a video frame and detect ASL sign
        Returns: (sign, confidence, landmarks) or (None, 0.0, None)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            # Use first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = self.extract_landmarks(hand_landmarks)

            # Normalize landmarks same way as database
            normalized_landmarks = self.database.normalize_landmarks(landmarks)

            # Match against database using normalized input
            match = self.matcher.match_sign(
                normalized_landmarks,
                self.database.database,
                threshold=self.confidence_threshold
            )

            if match:
                print(f"[DEBUG] Match found: {match}")
                sign, confidence = match
                return sign, confidence, landmarks
            else:
                # Try with lower threshold for best match
                matches = self.matcher.find_best_match(
                    normalized_landmarks,
                    self.database.database,
                    top_k=1
                )
                if matches:
                    sign, confidence = matches[0]
                    if confidence >= self.confidence_threshold * 0.8:
                        return sign, confidence, landmarks

        return None, 0.0, None

    def update_prediction(self, sign: Optional[str], confidence: float) -> Optional[str]:
        """
        Update prediction with stability checking
        Returns: Stable sign prediction or None
        """
        current_time = time.time()
        
        if sign is None:
            # Reset if no prediction for 2 seconds
            if current_time - self.last_prediction_time > 2.0:
                self.prediction_queue.clear()
            return None
        
        self.prediction_queue.append((sign, confidence, current_time))
        self.last_prediction_time = current_time
        
        # Check if prediction is stable
        if len(self.prediction_queue) >= 5:
            recent_signs = [p[0] for p in list(self.prediction_queue)[-5:]]
            recent_confidences = [p[1] for p in list(self.prediction_queue)[-5:]]
            
            # Check if same sign appears consistently
            if len(set(recent_signs)) == 1 and np.mean(recent_confidences) >= self.stability_threshold:
                print(f"[DETECTED] Stable: {recent_signs[0]} ({np.mean(recent_confidences):.2f})")
                stable_sign = recent_signs[0]
                if stable_sign != self.last_prediction:
                    self.last_prediction = stable_sign
                    return stable_sign
        
        return self.last_prediction
    
    def add_sign_to_phrase(self, sign: str):
        """Add a sign to the current phrase"""
        if sign and (not self.current_phrase or self.current_phrase[-1] != sign):
            self.current_phrase.append(sign)
            print(f"Added sign: {sign}. Current phrase: {' '.join(self.current_phrase)}")
    
    def finish_phrase(self) -> Optional[str]:
        """Finish current phrase and return the text"""
        if not self.current_phrase:
            return None
        
        phrase_text = ' '.join(self.current_phrase)
        self.phrase_history.append(phrase_text)
        self.current_phrase = []
        return phrase_text
    
    def format_text(self, asl_text: str) -> str:
        """Simple text formatting (capitalization, punctuation)"""
        if not asl_text:
            return asl_text
        
        # Capitalize first letter
        text = asl_text.strip()
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Add period if no punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using ElevenLabs API"""
        if not self.elevenlabs:
            return None
        
        if text in self.audio_cache:
            return self.audio_cache[text]
        
        audio = self.elevenlabs.text_to_speech(text)
        if audio:
            self.audio_cache[text] = audio
        return audio
    
    def collect_training_data(self, sign: str, num_samples: int = 20):
        """
        Collect training data for a specific sign from webcam
        Note: For better results, use load_dataset.py to load from pre-existing ASL datasets
        """
        print(f"Collecting {num_samples} samples for sign: {sign}")
        print("Press SPACE to capture sample, ESC to cancel")
        print("Note: Consider using pre-existing ASL datasets for better accuracy!")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0

        frame_skip = 2  # Capture every 2nd frame
        frame_count = 0
        while cap.isOpened() and samples_collected < num_samples:
            ret, frame = cap.read()
            frame_count +=1 
            if frame_count % frame_skip != 0:
                continue
            if not ret:
                break
            frame = cv2.resize(frame, (480, 360))
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    landmarks = self.extract_landmarks(hand_landmarks)
                    if self.np_database.add_landmark_sample(sign, landmarks):
                        samples_collected += 1
                        print(f"Collected {samples_collected}/{num_samples} samples")
            
            cv2.putText(frame, f"Collecting: {sign} ({samples_collected}/{num_samples})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | ESC: Cancel", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - already capturing continuously
                pass
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save database
        self.np_database.save_database()
        print(f"Collected {samples_collected} samples for {sign}")
        print("Tip: For better results, use 'python load_dataset.py' to load from pre-existing ASL datasets!")
    
    def run_console_mode(self):
        cap = cv2.VideoCapture(0)

        print("ASL Translator - Console Mode")
        print("Controls:")
        print("  ENTER: Finish phrase and refine")
        print("  BACKSPACE: Remove last sign")
        print("  's': Speak current phrase")
        print("  'c': Clear current phrase")
        print("  't': Training mode")
        print("  ESC: Exit")

        # Cooldown timer for auto-add
        self.last_auto_add_ts = 0.0

        frame_skip = 2  # every 2nd frame; adjust to 3 if still laggy
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Process frame and update prediction
            sign, confidence, landmarks = self.process_frame(frame)
            stable_sign = self.update_prediction(sign, confidence)

            # --- AUTO ADD MODE with cooldown ---
            now = time.time()
            if stable_sign \
                and (not self.current_phrase or self.current_phrase[-1] != stable_sign) \
                and (now - self.last_auto_add_ts) > 1.0:
                self.add_sign_to_phrase(stable_sign)
                self.last_auto_add_ts = now

            # Display info overlay
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            if sign:
                color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
                cv2.putText(frame, f"Detected: {sign} ({confidence:.2f})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if stable_sign:
                cv2.putText(frame, f"Stable: {stable_sign}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            current_phrase = ' '.join(self.current_phrase) if self.current_phrase else "None"
            cv2.putText(frame, f"Phrase: {current_phrase}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("ASL Translator", frame)

            # --- Keyboard controls ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            elif key == 13:  # ENTER
                phrase = self.finish_phrase()
                if phrase:
                    print(f"\nASL Phrase: {phrase}")
                    formatted = self.format_text(phrase)
                    print(f"Formatted: {formatted}")

            elif key == 8:  # BACKSPACE
                if self.current_phrase:
                    removed = self.current_phrase.pop()
                    print(f"Removed last sign: {removed}")

            elif key == ord('s'):  # Speak
                if self.current_phrase:
                    phrase = ' '.join(self.current_phrase)
                    formatted = self.format_text(phrase)
                    audio = self.text_to_speech(formatted)
                    if audio:
                        with open('output.mp3', 'wb') as f:
                            f.write(audio)
                        print("Audio saved to output.mp3")
                    else:
                        print("TTS not available. Set ELEVENLABS_API_KEY for text-to-speech.")

            elif key == ord('c'):  # Clear phrase
                self.current_phrase = []

            elif key == ord('t'):  # Training mode
                sign_name = input("Enter sign name to collect data for: ").strip()
                if sign_name:
                    self.collect_training_data(sign_name)

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    import os
    
    # Initialize with API keys from environment (TTS is optional)
    translator = ASLTranslatorApp(
        similarity_metric='combined',
        confidence_threshold=0.45,
        elevenlabs_api_key=os.getenv('ELEVENLABS_API_KEY')  # Optional
    )
    
    translator.run_console_mode()

