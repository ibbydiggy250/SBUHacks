# ASL Translator - Detection System

A comprehensive American Sign Language (ASL) detection and translation system using MediaPipe, similarity matching, and optional text-to-speech.

## Features

- **Real-time ASL Detection**: Uses MediaPipe to extract hand landmarks from live video
- **Landmark Database**: Stores normalized hand landmarks for known ASL signs
- **Similarity Matching**: Compares live landmarks to database using multiple metrics:
  - Euclidean distance
  - Cosine similarity
  - Dynamic Time Warping (DTW)
  - Combined metric (weighted combination)
- **Phrase Building**: Constructs phrases from detected signs
- **Text Formatting**: Simple capitalization and punctuation
- **Optional TTS**: Text-to-speech using ElevenLabs API (optional)
- **Streamlit Interface**: Web-based interface with webcam feed
- **Training Mode**: Collect training data for new signs

## Installation

### Windows (if pip is not recognized)

**Option 1: Use Python's built-in pip (Recommended)**
```bash
python -m pip install -r requirements.txt
```

**Option 2: Use installation scripts**
- Run `install_windows.bat` (Command Prompt)
- Or run `install_windows.ps1` (PowerShell)

**Option 3: Use Python launcher**
```bash
py -m pip install -r requirements.txt
```

See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows installation instructions.

### Linux/Mac

```bash
pip install -r requirements.txt
```

Or if you have both Python 2 and 3:
```bash
pip3 install -r requirements.txt
```

## Usage

### Console Mode

Run the main application:

```bash
python asl_translator_app.py
```

**Controls:**
- **SPACE**: Add detected sign to phrase
- **ENTER**: Finish phrase and format text
- **'s'**: Generate speech (if TTS enabled)
- **'c'**: Clear current phrase
- **'t'**: Enter training mode

### Streamlit Web Interface

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

This opens a web interface with:
- Live webcam feed
- Real-time sign detection
- Phrase building
- Text formatting
- Optional text-to-speech
- Training mode in sidebar

## Components

### 1. ASL Database (`asl_database.py`)
- Stores normalized hand landmarks for ASL signs
- JSON-based storage
- Normalizes landmarks for scale and position invariance

### 2. Similarity Matcher (`similarity_matcher.py`)
- Compares live landmarks to database
- Supports multiple similarity metrics
- Returns confidence scores

### 3. ASL Translator App (`asl_translator_app.py`)
- Main application orchestrator
- Processes video frames
- Manages phrase building
- Handles training data collection

### 4. Streamlit Interface (`streamlit_app.py`)
- Web-based user interface
- Real-time webcam processing
- Interactive controls

### 5. API Integrations (`api_integrations.py`)
- ElevenLabs TTS API (optional)

## Loading Pre-existing ASL Datasets

The system can learn from pre-existing ASL images and videos (recommended):

```bash
python load_dataset.py
```

See [DATASET_LOADING.md](DATASET_LOADING.md) for detailed instructions.

**Supported formats:**
- Folder structure (each folder = sign name)
- WLASL dataset format
- MS-ASL dataset format
- Flat structure (all files in one folder)
- JSON file (custom mapping)

## Training Data Collection (Alternative)

If you prefer to collect data from your webcam:

1. **Via Console**: Press 't' and enter sign name
2. **Via Streamlit**: Use sidebar training mode
3. **Collect Samples**: Show the sign to camera (20+ samples recommended)
4. **Save**: Database automatically saves after collection

## Database Structure

The database stores landmarks in `asl_database.json`:
```json
{
  "HELLO": [
    [landmark_coordinates...],
    [landmark_coordinates...]
  ],
  "THANK YOU": [
    [landmark_coordinates...]
  ]
}
```

## Configuration

### Similarity Metrics

- **Euclidean**: Distance-based matching
- **Cosine**: Angle-based similarity
- **DTW**: Dynamic Time Warping for sequences
- **Combined**: Weighted combination (recommended)

### Confidence Threshold

Adjust the minimum confidence for sign detection (default: 0.6)

## API Keys (Optional)

For text-to-speech, set your ElevenLabs API key:

**Environment Variable:**
```bash
export ELEVENLABS_API_KEY="your_key_here"
```

**Streamlit Secrets** (`.streamlit/secrets.toml`):
```toml
ELEVENLABS_API_KEY = "your_key_here"
```

**Note**: TTS is optional. The ASL detection system works without it.

## Example Workflow

1. **Collect Training Data**:
   - Run training mode
   - Show signs to camera
   - Collect 20+ samples per sign

2. **Detect Signs**:
   - Position hand in front of camera
   - System detects and displays sign
   - Add signs to phrase

3. **Build Phrases**:
   - Add multiple signs
   - Finish phrase to format text
   - Optional: Generate speech

## File Structure

```
SBUHacks/
├── asl_database.py          # Database management
├── similarity_matcher.py    # Similarity matching algorithms
├── asl_translator_app.py    # Main application
├── streamlit_app.py         # Web interface
├── api_integrations.py      # TTS API integration
├── requirements.txt         # Dependencies
├── asl_database.json        # Sign database (created automatically)
└── README.md               # This file
```

## Dependencies

- `opencv-python`: Video processing
- `mediapipe`: Hand landmark detection
- `numpy`: Numerical operations
- `scipy`: Scientific computing (for DTW)
- `requests`: API calls
- `streamlit`: Web interface
- `Pillow`: Image processing

## Troubleshooting

**No signs detected:**
- Ensure good lighting
- Keep hand in frame
- Collect more training data
- Lower confidence threshold

**Low accuracy:**
- Collect more training samples (20+ per sign)
- Use 'combined' similarity metric
- Adjust confidence threshold
- Ensure consistent hand positioning

**TTS not working:**
- Check API key is set correctly
- Verify internet connection
- Check API quota/limits

## Future Enhancements

- Support for two-handed signs
- Temporal sequence recognition
- Integration with ASL datasets (WLASL, MS-ASL)
- Real-time phrase completion
- Mobile app version

## License

This project is open source and available for educational and research purposes.
