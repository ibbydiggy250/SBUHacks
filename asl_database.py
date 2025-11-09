"""
ASL Landmark Database - Stores and manages normalized hand landmarks for ASL signs
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class ASLDatabase:

    def get_numpy_database(self):
        """Convert stored samples to numpy arrays once for faster matching."""
        import numpy as np
        np_db = {}
        for sign, samples in self.database.items():
            np_db[sign] = [np.array(s, dtype=np.float32) for s in samples]
        return np_db

    def __init__(self, db_path: str = "asl_database.json"):
        self.db_path = db_path
        self.database: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.load_database()
    
    def load_database(self):
        """Load database from JSON file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Convert back to numpy arrays
                    for sign, landmarks_list in data.items():
                        self.database[sign] = [np.array(lm) for lm in landmarks_list]
                print(f"Loaded {len(self.database)} signs from database")
            except Exception as e:
                print(f"Error loading database: {e}")
        else:
            print("No existing database found. Creating new one.")
    
    def save_database(self):
        """Save database to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            data = {}
            for sign, landmarks_list in self.database.items():
                data[sign] = [lm.tolist() for lm in landmarks_list]
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Database saved with {len(self.database)} signs")
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def add_landmark_sample(self, sign: str, landmarks: np.ndarray):
        """Add a landmark sample for a specific sign"""
        if landmarks is None or len(landmarks) == 0:
            return False
        
        # Normalize landmarks
        normalized = self.normalize_landmarks(landmarks)
        self.database[sign.upper()].append(normalized)
        return True
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to be scale and position invariant"""
        if len(landmarks) == 0:
            return landmarks
        
        # Reshape if needed (21 landmarks * 3 coordinates = 63 values)
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 3)
        
        # Center around wrist (landmark 0)
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        
        # Normalize by scale (use distance from wrist to middle finger MCP as reference)
        if len(centered) > 9:
            scale = np.linalg.norm(centered[9])  # Middle finger MCP
            if scale > 0:
                centered = centered / scale
        
        return centered.flatten()
    
    def get_sign_samples(self, sign: str) -> List[np.ndarray]:
        """Get all samples for a specific sign"""
        return self.database.get(sign.upper(), [])
    
    def get_all_signs(self) -> List[str]:
        """Get list of all signs in database"""
        return list(self.database.keys())
    
    def remove_sign(self, sign: str):
        """Remove a sign from the database"""
        if sign.upper() in self.database:
            del self.database[sign.upper()]
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        total_samples = sum(len(samples) for samples in self.database.values())
        return {
            'num_signs': len(self.database),
            'total_samples': total_samples,
            'signs': {sign: len(samples) for sign, samples in self.database.items()}
        }

