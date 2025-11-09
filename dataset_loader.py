"""
Dataset Loader - Load ASL signs from pre-existing images and videos
Supports common ASL datasets and custom image/video collections
"""

import cv2
import os
import glob
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mediapipe as mp
from tqdm import tqdm

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class ASLDatasetLoader:
    def __init__(self):
        """Initialize dataset loader with MediaPipe hands"""
        self.hands = mp_hands.Hands(
            static_image_mode=True,  # True for images, can be False for videos
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """Extract hand landmarks from a single image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Use first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return np.array(landmarks)
            
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_landmarks_from_video(self, video_path: str, sample_frames: int = 10) -> List[np.ndarray]:
        """Extract hand landmarks from video frames"""
        landmarks_list = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return landmarks_list
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count in frame_indices:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    results = self.hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        # Use first hand detected
                        hand_landmarks = results.multi_hand_landmarks[0]
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks_list.append(np.array(landmarks))
                
                frame_count += 1
            
            cap.release()
            return landmarks_list
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return landmarks_list
    
    def load_from_folder_structure(self, base_path: str, structure: str = "sign_name") -> Dict[str, List[np.ndarray]]:
        """
        Load dataset from folder structure
        Args:
            base_path: Base directory containing the dataset
            structure: Folder structure type
                - "sign_name": Each folder is a sign name, contains images/videos
                - "flat": All files in one folder, named like "SIGN_NAME_001.jpg"
        Returns:
            Dictionary mapping sign names to lists of landmarks
        """
        dataset = {}
        
        if structure == "sign_name":
            # Structure: base_path/SIGN_NAME/image1.jpg, video1.mp4, etc.
            sign_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            
            for sign_folder in tqdm(sign_folders, desc="Loading signs"):
                sign_name = sign_folder.upper()
                sign_path = os.path.join(base_path, sign_folder)
                
                landmarks_list = []
                
                # Load images
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                for ext in image_extensions:
                    images = glob.glob(os.path.join(sign_path, ext))
                    images.extend(glob.glob(os.path.join(sign_path, ext.upper())))
                    
                    for image_path in images:
                        landmarks = self.extract_landmarks_from_image(image_path)
                        if landmarks is not None:
                            landmarks_list.append(landmarks)
                
                # Load videos
                video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
                for ext in video_extensions:
                    videos = glob.glob(os.path.join(sign_path, ext))
                    videos.extend(glob.glob(os.path.join(sign_path, ext.upper())))
                    
                    for video_path in videos:
                        video_landmarks = self.extract_landmarks_from_video(video_path)
                        landmarks_list.extend(video_landmarks)
                
                if landmarks_list:
                    dataset[sign_name] = landmarks_list
                    print(f"Loaded {len(landmarks_list)} samples for {sign_name}")
        
        elif structure == "flat":
            # Structure: base_path/SIGN_NAME_001.jpg, SIGN_NAME_002.mp4, etc.
            all_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.mp4', '*.avi', '*.mov']:
                all_files.extend(glob.glob(os.path.join(base_path, ext)))
                all_files.extend(glob.glob(os.path.join(base_path, ext.upper())))
            
            for file_path in tqdm(all_files, desc="Processing files"):
                # Extract sign name from filename
                filename = os.path.basename(file_path)
                # Assume format: SIGN_NAME_001.jpg or SIGN-NAME_001.jpg
                sign_name = filename.split('_')[0].split('-')[0].upper()
                
                if sign_name not in dataset:
                    dataset[sign_name] = []
                
                # Process file
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    landmarks = self.extract_landmarks_from_image(file_path)
                    if landmarks is not None:
                        dataset[sign_name].append(landmarks)
                else:
                    video_landmarks = self.extract_landmarks_from_video(file_path)
                    dataset[sign_name].extend(video_landmarks)
        
        return dataset
    
    def load_from_wlasl_format(self, dataset_path: str) -> Dict[str, List[np.ndarray]]:
        """
        Load from WLASL (Word-Level American Sign Language) dataset format
        Structure: dataset_path/videos/SIGN_NAME/video_files
        """
        dataset = {}
        videos_path = os.path.join(dataset_path, 'videos')
        
        if not os.path.exists(videos_path):
            print(f"WLASL format: videos folder not found at {videos_path}")
            return dataset
        
        sign_folders = [d for d in os.listdir(videos_path) if os.path.isdir(os.path.join(videos_path, d))]
        
        for sign_folder in tqdm(sign_folders, desc="Loading WLASL dataset"):
            sign_name = sign_folder.upper()
            sign_path = os.path.join(videos_path, sign_folder)
            
            video_files = glob.glob(os.path.join(sign_path, '*.mp4'))
            video_files.extend(glob.glob(os.path.join(sign_path, '*.avi')))
            
            landmarks_list = []
            for video_path in video_files:
                video_landmarks = self.extract_landmarks_from_video(video_path, sample_frames=5)
                landmarks_list.extend(video_landmarks)
            
            if landmarks_list:
                dataset[sign_name] = landmarks_list
                print(f"Loaded {len(landmarks_list)} samples for {sign_name}")
        
        return dataset
    
    def load_from_msasl_format(self, dataset_path: str) -> Dict[str, List[np.ndarray]]:
        """
        Load from MS-ASL dataset format
        Structure: dataset_path/videos/video_files (with metadata JSON)
        """
        dataset = {}
        videos_path = os.path.join(dataset_path, 'videos')
        
        if not os.path.exists(videos_path):
            print(f"MS-ASL format: videos folder not found at {videos_path}")
            return dataset
        
        # Try to load metadata if available
        metadata_path = os.path.join(dataset_path, 'MSASL_train.json')
        video_labels = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                for item in metadata:
                    video_labels[item['url'].split('/')[-1]] = item['label']
        
        # Process videos
        video_files = glob.glob(os.path.join(videos_path, '*.mp4'))
        video_files.extend(glob.glob(os.path.join(videos_path, '*.avi')))
        
        for video_path in tqdm(video_files, desc="Loading MS-ASL dataset"):
            filename = os.path.basename(video_path)
            
            # Get sign name from metadata or filename
            if filename in video_labels:
                sign_name = video_labels[filename].upper()
            else:
                # Extract from filename
                sign_name = filename.split('_')[0].upper()
            
            if sign_name not in dataset:
                dataset[sign_name] = []
            
            video_landmarks = self.extract_landmarks_from_video(video_path, sample_frames=5)
            dataset[sign_name].extend(video_landmarks)
        
        return dataset
    
    def load_from_custom_json(self, json_path: str, media_base_path: str = None) -> Dict[str, List[np.ndarray]]:
        """
        Load from custom JSON file with structure:
        {
            "SIGN_NAME": ["path/to/image1.jpg", "path/to/video1.mp4", ...],
            ...
        }
        """
        dataset = {}
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for sign_name, file_paths in tqdm(data.items(), desc="Loading from JSON"):
            sign_name_upper = sign_name.upper()
            dataset[sign_name_upper] = []
            
            for file_path in file_paths:
                # Handle relative paths
                if media_base_path:
                    full_path = os.path.join(media_base_path, file_path)
                else:
                    full_path = file_path
                
                if not os.path.exists(full_path):
                    continue
                
                # Process file based on extension
                if full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    landmarks = self.extract_landmarks_from_image(full_path)
                    if landmarks is not None:
                        dataset[sign_name_upper].append(landmarks)
                elif full_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_landmarks = self.extract_landmarks_from_video(full_path)
                    dataset[sign_name_upper].extend(video_landmarks)
        
        return dataset
    
    def process_dataset(
        self,
        dataset_path: str,
        format_type: str = "auto",
        output_db_path: str = "asl_database.json"
    ) -> Dict[str, List[np.ndarray]]:
        """
        Process a dataset and save to database
        Args:
            dataset_path: Path to dataset
            format_type: "auto", "sign_name", "flat", "wlasl", "msasl", "json"
            output_db_path: Path to save the database
        """
        print(f"Loading dataset from: {dataset_path}")
        
        # Auto-detect format
        if format_type == "auto":
            if os.path.exists(os.path.join(dataset_path, 'videos')):
                # Check for WLASL or MS-ASL format
                if os.path.exists(os.path.join(dataset_path, 'MSASL_train.json')):
                    format_type = "msasl"
                else:
                    format_type = "wlasl"
            elif os.path.isfile(dataset_path) and dataset_path.endswith('.json'):
                format_type = "json"
            else:
                format_type = "sign_name"
        
        # Load dataset
        if format_type == "wlasl":
            dataset = self.load_from_wlasl_format(dataset_path)
        elif format_type == "msasl":
            dataset = self.load_from_msasl_format(dataset_path)
        elif format_type == "json":
            dataset = self.load_from_custom_json(dataset_path)
        elif format_type == "flat":
            dataset = self.load_from_folder_structure(dataset_path, structure="flat")
        else:  # sign_name
            dataset = self.load_from_folder_structure(dataset_path, structure="sign_name")
        
        # Save to database
        if dataset:
            self.save_to_database(dataset, output_db_path)
            print(f"\nDataset loaded: {len(dataset)} signs")
            total_samples = sum(len(samples) for samples in dataset.values())
            print(f"Total samples: {total_samples}")
        else:
            print("No data loaded from dataset")
        
        return dataset
    
    def save_to_database(self, dataset: Dict[str, List[np.ndarray]], db_path: str):
        """Save dataset to database JSON file"""
        from asl_database import ASLDatabase
        
        db = ASLDatabase(db_path=db_path)
        
        for sign_name, landmarks_list in dataset.items():
            for landmarks in landmarks_list:
                db.add_landmark_sample(sign_name, landmarks)
        
        db.save_database()
        print(f"Database saved to {db_path}")
    
    def close(self):
        """Close MediaPipe hands"""
        self.hands.close()


def main():
    """Command-line interface for loading datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load ASL dataset from images/videos")
    parser.add_argument("dataset_path", help="Path to dataset folder or JSON file")
    parser.add_argument("--format", choices=["auto", "sign_name", "flat", "wlasl", "msasl", "json"],
                       default="auto", help="Dataset format type")
    parser.add_argument("--output", default="asl_database.json", help="Output database path")
    parser.add_argument("--media-base", help="Base path for media files (for JSON format)")
    
    args = parser.parse_args()
    
    loader = ASLDatasetLoader()
    
    try:
        if args.format == "json":
            dataset = loader.load_from_custom_json(args.dataset_path, args.media_base)
            loader.save_to_database(dataset, args.output)
        else:
            loader.process_dataset(args.dataset_path, args.format, args.output)
    finally:
        loader.close()


if __name__ == "__main__":
    main()

