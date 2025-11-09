"""
Streamlit Web Interface for ASL Translator
Zoom-style interface with webcam feed, text display, and audio playback
"""

import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
import base64

from asl_translator_app import ASLTranslatorApp
from asl_database import ASLDatabase

# Page configuration
st.set_page_config(
    page_title="ASL Translator",
    page_icon="🤟",
    layout="wide"
)

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'current_phrase' not in st.session_state:
    st.session_state.current_phrase = []
if 'refined_text' not in st.session_state:
    st.session_state.refined_text = ""
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

def initialize_translator():
    """Initialize the ASL translator"""
    if st.session_state.translator is None:
        # Get API key from secrets or environment (optional, for TTS only)
        elevenlabs_key = None
        if hasattr(st, 'secrets'):
            elevenlabs_key = st.secrets.get('ELEVENLABS_API_KEY', None)
        elif hasattr(st, 'session_state'):
            elevenlabs_key = st.session_state.get('elevenlabs_api_key', None)
        
        st.session_state.translator = ASLTranslatorApp(
            similarity_metric=st.session_state.get('similarity_metric', 'combined'),
            confidence_threshold=st.session_state.get('confidence_threshold', 0.6),
            elevenlabs_api_key=elevenlabs_key  # Optional, for TTS
        )

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    similarity_metric = st.selectbox(
        "Similarity Metric",
        options=['euclidean', 'cosine', 'dtw', 'combined'],
        index=3,
        help="Method for comparing hand landmarks"
    )
    st.session_state.similarity_metric = similarity_metric
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Minimum confidence for sign detection"
    )
    st.session_state.confidence_threshold = confidence_threshold
    
    if st.session_state.translator:
        from similarity_matcher import SimilarityMatcher
        st.session_state.translator.matcher = SimilarityMatcher(metric=similarity_metric)
        st.session_state.translator.confidence_threshold = confidence_threshold
    
    st.divider()
    
    st.header("📚 Database")
    db_stats = None
    if st.session_state.translator:
        db_stats = st.session_state.translator.database.get_statistics()
        st.metric("Signs in Database", db_stats['num_signs'])
        st.metric("Total Samples", db_stats['total_samples'])
    
    if st.button("🔄 Refresh Database"):
        if st.session_state.translator:
            st.session_state.translator.database.load_database()
            st.rerun()
    
    st.divider()
    
    st.header("📚 Load Dataset")
    st.markdown("**Recommended**: Load from pre-existing ASL datasets for better accuracy")
    
    dataset_path = st.text_input("Dataset Path", help="Path to dataset folder or JSON file", key="dataset_path")
    dataset_format = st.selectbox(
        "Dataset Format",
        options=["auto", "sign_name", "flat", "wlasl", "msasl", "json"],
        index=0,
        help="Auto-detect or specify format"
    )
    
    if st.button("📥 Load Dataset", use_container_width=True):
        if dataset_path and st.session_state.translator:
            try:
                from dataset_loader import ASLDatasetLoader
                loader = ASLDatasetLoader()
                
                with st.spinner("Loading dataset... This may take a while..."):
                    loader.process_dataset(dataset_path, dataset_format, "asl_database.json")
                    # Reload database
                    st.session_state.translator.database.load_database()
                
                st.success("Dataset loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.code(str(e))
    
    st.markdown("**See [DATASET_LOADING.md](DATASET_LOADING.md) for detailed instructions**")
    
    st.divider()
    
    st.header("🎤 Training Mode (Alternative)")
    st.markdown("*Collect training data from your webcam*")
    training_sign = st.text_input("Sign Name", key="training_sign")
    num_samples = st.number_input("Number of Samples", min_value=1, max_value=100, value=20)
    
    if st.button("📝 Collect Training Data"):
        if training_sign and st.session_state.translator:
            st.info(f"Collecting {num_samples} samples for '{training_sign}'. Check the popup window.")
            st.session_state.translator.collect_training_data(training_sign, int(num_samples))
            st.success("Training data collected!")
            st.rerun()

# Main content
st.title("🤟 ASL Translator")
st.markdown("Real-time American Sign Language translation with grammar refinement and text-to-speech")

# Initialize translator
initialize_translator()

# Main columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Webcam Feed")
    
    # Webcam feed
    camera_feed = st.camera_input("Camera", key="camera")
    
    if camera_feed is not None:
        # Convert to OpenCV format
        bytes_data = camera_feed.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        if st.session_state.translator:
            # Process frame
            sign, confidence, landmarks = st.session_state.translator.process_frame(cv_image)
            stable_sign = st.session_state.translator.update_prediction(sign, confidence)
            
            # Display detection
            if sign:
                confidence_color = "🟢" if confidence >= confidence_threshold else "🟡"
                st.write(f"{confidence_color} **Detected:** {sign} (Confidence: {confidence:.2f})")
            
            if stable_sign:
                st.success(f"✅ **Stable Sign:** {stable_sign}")
                
                # Auto-add to phrase if button pressed
                if st.button("➕ Add Sign to Phrase", key="add_sign"):
                    st.session_state.translator.add_sign_to_phrase(stable_sign)
                    st.session_state.current_phrase = st.session_state.translator.current_phrase
                    st.rerun()
        
        # Draw landmarks on image
        if st.session_state.translator and cv_image is not None:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = st.session_state.translator.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                import mediapipe as mp
                mp_draw = mp.solutions.drawing_utils
                mp_hands = mp.solutions.hands
                mp_drawing_styles = mp.solutions.drawing_styles
                
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        rgb_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                st.image(rgb_image, use_container_width=True)

with col2:
    st.header("📝 Translation")
    
    # Current phrase
    st.subheader("Current Phrase")
    current_phrase_text = ' '.join(st.session_state.current_phrase) if st.session_state.current_phrase else "No signs yet"
    st.text_area("Raw ASL Text", value=current_phrase_text, height=100, disabled=True, key="raw_text")
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("✅ Finish Phrase", use_container_width=True):
            if st.session_state.translator:
                phrase = st.session_state.translator.finish_phrase()
                if phrase:
                    st.session_state.current_phrase = []
                    # Format text (simple capitalization/punctuation)
                    formatted = st.session_state.translator.format_text(phrase)
                    st.session_state.refined_text = formatted
                    st.session_state.translator.current_phrase = []
                    st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            if st.session_state.translator:
                st.session_state.translator.current_phrase = []
                st.session_state.current_phrase = []
                st.session_state.refined_text = ""
                st.rerun()
    
    st.divider()
    
    # Formatted text
    st.subheader("Formatted Text")
    refined_text = st.text_area(
        "Formatted ASL Text",
        value=st.session_state.refined_text,
        height=100,
        key="refined_text_area",
        help="Simple formatting: capitalization and punctuation"
    )
    
    if refined_text:
        st.session_state.refined_text = refined_text
        
        # Text-to-speech button
        if st.button("🔊 Generate Speech", use_container_width=True):
            if st.session_state.translator:
                with st.spinner("Generating speech..."):
                    audio_data = st.session_state.translator.text_to_speech(refined_text)
                    if audio_data:
                        st.session_state.audio_data = audio_data
                        st.audio(audio_data, format='audio/mpeg')
                        st.success("Audio generated!")
                    else:
                        st.error("Failed to generate audio. Check API key.")
    
    # Display audio if available
    if st.session_state.audio_data:
        st.audio(st.session_state.audio_data, format='audio/mpeg')
    
    st.divider()
    
    # History
    st.subheader("📜 History")
    if st.session_state.translator:
        history = st.session_state.translator.phrase_history[-5:]
        for i, phrase in enumerate(reversed(history)):
            st.text(f"{len(history)-i}. {phrase}")

# Footer
st.divider()
st.markdown("""
### 📖 Instructions
1. **Position your hand** in front of the webcam
2. **Make an ASL sign** - the system will detect it
3. **Click "Add Sign to Phrase"** to add detected signs
4. **Click "Finish Phrase"** to format the text
5. **Click "Generate Speech"** to hear the translation

### 🔑 API Keys (Optional)
For text-to-speech, set your ElevenLabs API key in `.streamlit/secrets.toml`:
```toml
ELEVENLABS_API_KEY = "your_key_here"
```
Note: TTS is optional. The ASL detection system works without it.

### 📚 Training
Use the sidebar to collect training data for new signs.
""")

