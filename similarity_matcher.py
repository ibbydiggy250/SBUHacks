"""
Similarity Matcher - Compares live landmarks to database using various metrics
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean, cosine
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class SimilarityMatcher:
    def __init__(self, metric: str = 'euclidean'):
        """
        Initialize similarity matcher
        Args:
            metric: 'euclidean', 'cosine', 'dtw', or 'combined'
        """
        self.metric = metric.lower()
        if self.metric not in ['euclidean', 'cosine', 'dtw', 'combined']:
            raise ValueError(f"Unknown metric: {metric}")
    
    def euclidean_distance(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Calculate Euclidean distance between two landmark sets"""
        if landmarks1.shape != landmarks2.shape:
            # Pad or truncate to match
            min_len = min(len(landmarks1), len(landmarks2))
            landmarks1 = landmarks1[:min_len]
            landmarks2 = landmarks2[:min_len]
        
        return float(np.linalg.norm(landmarks1 - landmarks2))
    
    def cosine_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Calculate cosine similarity between two landmark sets"""
        if landmarks1.shape != landmarks2.shape:
            min_len = min(len(landmarks1), len(landmarks2))
            landmarks1 = landmarks1[:min_len]
            landmarks2 = landmarks2[:min_len]
        
        # Flatten for cosine similarity
        v1 = landmarks1.flatten()
        v2 = landmarks2.flatten()
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def dtw_distance(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """
        Dynamic Time Warping distance for sequence alignment
        Simplified version for static landmarks
        """
        # Reshape to sequences if needed
        if landmarks1.ndim == 1:
            landmarks1 = landmarks1.reshape(-1, 3)
        if landmarks2.ndim == 1:
            landmarks2 = landmarks2.reshape(-1, 3)
        
        n, m = len(landmarks1), len(landmarks2)
        
        # Initialize DTW matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(landmarks1[i-1] - landmarks2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return float(dtw_matrix[n, m])
    
    def calculate_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """Calculate similarity based on selected metric"""
        # --- Safety checks to prevent shape or value errors ---
        if landmarks1 is None or landmarks2 is None:
            return 0.0
        if len(landmarks1) == 0 or len(landmarks2) == 0:
            return 0.0

        # Flatten both to 1D arrays
        landmarks1 = np.array(landmarks1).flatten()
        landmarks2 = np.array(landmarks2).flatten()

        # Truncate or pad so both have the same length
        min_len = min(len(landmarks1), len(landmarks2))
        landmarks1 = landmarks1[:min_len]
        landmarks2 = landmarks2[:min_len]

        if self.metric == 'euclidean':
            distance = self.euclidean_distance(landmarks1, landmarks2)
            # Convert distance to similarity (inverse, normalized)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            return similarity
        elif self.metric == 'cosine':
            return self.cosine_similarity(landmarks1, landmarks2)
        elif self.metric == 'dtw':
            distance = self.dtw_distance(landmarks1, landmarks2)
            similarity = 1 / (1 + distance)
            return similarity
        elif self.metric == 'combined':
            # Weighted combination of all metrics
            euclidean_sim = 1 / (1 + self.euclidean_distance(landmarks1, landmarks2))
            cosine_sim = self.cosine_similarity(landmarks1, landmarks2)
            dtw_sim = 1 / (1 + self.dtw_distance(landmarks1, landmarks2))
            
            # Normalize and combine
            combined = (euclidean_sim * 0.4 + cosine_sim * 0.3 + dtw_sim * 0.3)
            return combined
        
        return 0.0
    
    def find_best_match(
        self, 
        query_landmarks: np.ndarray, 
        database: Dict[str, List[np.ndarray]],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find best matching signs in database
        Returns: List of (sign, confidence) tuples, sorted by confidence (highest first)
        """
        matches = []
        
        for sign, samples in database.items():
            if len(samples) == 0:
                continue
            
            # Calculate similarity with all samples for this sign
            similarities = []
            for sample in samples:
                try:
                # Skip invalid or empty samples
                    if sample is None or len(sample) == 0:
                        continue
                    if query_landmarks.shape != sample.shape:
                        # Try to flatten or reshape if needed
                        sample = sample.flatten()
                        query_landmarks = query_landmarks.flatten()
                        if sample.shape != query_landmarks.shape:
                            continue  # skip bad sample entirely

                    sim = self.calculate_similarity(query_landmarks, sample)
                    similarities.append(sim)
                except Exception as e:
                    print(f"[WARN] Skipped sample for {sign}: {e}")
                    continue

            
            if similarities:
                # Use average similarity, or max similarity
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                # Use weighted average favoring max (more robust to outliers)
                confidence = 0.7 * max_similarity + 0.3 * avg_similarity
                matches.append((sign, float(confidence)))
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def match_sign(
        self,
        query_landmarks: np.ndarray,
        database: Dict[str, List[np.ndarray]],
        threshold: float = 0.5
    ) -> Optional[Tuple[str, float]]:
        """
        Match a sign with confidence threshold
        Returns: (sign, confidence) or None if no match above threshold
        """
        matches = self.find_best_match(query_landmarks, database, top_k=1)
        
        if matches and matches[0][1] >= threshold:
            return matches[0]
        
        return None

