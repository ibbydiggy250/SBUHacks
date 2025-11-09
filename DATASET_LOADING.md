# Loading ASL Datasets

This guide explains how to load pre-existing ASL images and videos into the database.

## Quick Start

Run the dataset loader script:

```bash
python load_dataset.py
```

The script will guide you through the process.

## Supported Dataset Formats

### 1. Folder Structure (Recommended)

Each folder represents a sign name, containing images/videos:

```
dataset/
├── HELLO/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── video1.mp4
├── THANK YOU/
│   ├── image1.jpg
│   └── video1.mp4
└── YES/
    └── image1.jpg
```

**Usage:**
```bash
python load_dataset.py
# Enter path: dataset/
# Format: sign_name (or auto-detect)
```

### 2. WLASL Format

WLASL (Word-Level American Sign Language) dataset structure:

```
wlasl_dataset/
└── videos/
    ├── hello/
    │   ├── video1.mp4
    │   └── video2.mp4
    ├── thank_you/
    │   └── video1.mp4
    └── yes/
        └── video1.mp4
```

**Usage:**
```bash
python load_dataset.py
# Enter path: wlasl_dataset/
# Format: wlasl (or auto-detect)
```

### 3. MS-ASL Format

MS-ASL dataset structure:

```
msasl_dataset/
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── MSASL_train.json
```

**Usage:**
```bash
python load_dataset.py
# Enter path: msasl_dataset/
# Format: msasl (or auto-detect)
```

### 4. Flat Structure

All files in one folder, named with sign prefix:

```
dataset/
├── HELLO_001.jpg
├── HELLO_002.jpg
├── THANK_YOU_001.mp4
└── YES_001.jpg
```

**Usage:**
```bash
python load_dataset.py
# Enter path: dataset/
# Format: flat
```

### 5. JSON Format

Custom JSON file mapping signs to file paths:

```json
{
    "HELLO": [
        "path/to/hello1.jpg",
        "path/to/hello2.mp4"
    ],
    "THANK YOU": [
        "path/to/thank_you1.jpg"
    ]
}
```

**Usage:**
```bash
python load_dataset.py
# Enter path: dataset.json
# Format: json
# Media base path: /path/to/media/files (if paths are relative)
```

## Command Line Usage

You can also use the dataset loader directly:

```bash
python dataset_loader.py dataset_path --format auto --output asl_database.json
```

**Arguments:**
- `dataset_path`: Path to dataset folder or JSON file
- `--format`: Format type (auto, sign_name, flat, wlasl, msasl, json)
- `--output`: Output database path (default: asl_database.json)
- `--media-base`: Base path for media files (for JSON format)

## Examples

### Example 1: Load from folder structure
```bash
python dataset_loader.py ./my_asl_dataset --format sign_name --output asl_database.json
```

### Example 2: Load WLASL dataset
```bash
python dataset_loader.py ./wlasl_dataset --format wlasl --output asl_database.json
```

### Example 3: Load MS-ASL dataset
```bash
python dataset_loader.py ./msasl_dataset --format msasl --output asl_database.json
```

### Example 4: Load from JSON
```bash
python dataset_loader.py ./dataset.json --format json --media-base ./media --output asl_database.json
```

## Supported File Formats

### Images:
- JPG/JPEG
- PNG
- BMP

### Videos:
- MP4
- AVI
- MOV
- MKV

## Processing Details

- **Images**: Extracts landmarks from the entire image
- **Videos**: Samples 5-10 frames per video (configurable)
- **Hand Detection**: Uses MediaPipe to detect hands
- **Normalization**: Landmarks are normalized for scale and position invariance

## Dataset Sources

### Recommended ASL Datasets:

1. **WLASL (Word-Level American Sign Language)**
   - ~12,000 videos covering 2,000 ASL words
   - Download: https://github.com/dxli94/WLASL

2. **MS-ASL (Microsoft American Sign Language)**
   - 1,000 ASL signs performed by 200+ signers
   - Download: https://www.microsoft.com/en-us/research/project/ms-asl/

3. **ASL Citizen**
   - 84,000 videos of 2,731 ASL signs
   - Download: https://www.microsoft.com/en-us/research/project/asl-citizen/

4. **Custom Datasets**
   - Organize your own images/videos in folder structure
   - Or create a JSON file mapping signs to files

## Tips

1. **Quality**: Use clear images/videos with visible hands
2. **Lighting**: Good lighting improves detection accuracy
3. **Multiple Samples**: More samples per sign = better accuracy
4. **Diversity**: Include different angles, backgrounds, signers
5. **Naming**: Use consistent naming (uppercase, no spaces or use underscores)

## Troubleshooting

### No landmarks detected:
- Check if hands are visible in images/videos
- Ensure good lighting
- Try adjusting MediaPipe confidence thresholds

### Low accuracy:
- Collect more samples per sign (20+ recommended)
- Use diverse samples (different angles, backgrounds)
- Ensure consistent hand positioning

### Processing is slow:
- Videos take longer than images
- Reduce `sample_frames` parameter for videos
- Process in batches if dataset is very large

## Next Steps

After loading the dataset:

1. **Test the database:**
   ```bash
   python test_installation.py
   ```

2. **Run the translator:**
   ```bash
   python asl_translator_app.py
   ```

3. **Or use Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

