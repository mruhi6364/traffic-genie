import cv2
import numpy as np
import logging
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DeepSortTracker:
    """
    Implementation of DeepSORT tracker for object tracking
    This is a simplified version that focuses on the core functionality
    In a real implementation, you would use the full deep_sort library
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the DeepSORT tracker
        
        Args:
            max_age (int): Maximum number of frames to keep track of unmatched tracks
            min_hits (int): Minimum number of hits to confirm a track
            iou_threshold (float): IoU threshold for track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks = []  # Active tracks
        self.frame_count = 0
        
        logger.info("DeepSORT tracker initialized")
    
    def update(self, detections, scores, class_ids, frame):
        """
        Update tracker with new detections
        
        Args:
            detections (numpy.ndarray): Bounding boxes in format [x1, y1, x2, y2]
            scores (numpy.ndarray): Detection confidence scores
            class_ids (numpy.ndarray): Class IDs for each detection
            frame (numpy.ndarray): Current frame
            
        Returns:
            list: List of tracks in format [x1, y1, x2, y2, track_id, class_id]
        """
        self.frame_count += 1
        
        # If no current detections
        if len(detections) == 0:
            # Just return active tracks that haven't expired yet
            active_tracks = []
            for track in self.tracks:
                track['time_since_update'] += 1
                if track['time_since_update'] <= self.max_age:
                    active_tracks.append(track)
            self.tracks = active_tracks
            
            # Convert tracks to output format
            results = []
            for track in self.tracks:
                if track['hits'] >= self.min_hits:
                    x1, y1, x2, y2 = track['bbox']
                    track_id = track['id']
                    class_id = track['class_id']
                    results.append([x1, y1, x2, y2, track_id, class_id])
            
            return results
        
        # If we have active tracks, try to match them to new detections
        if len(self.tracks) > 0:
            # Calculate IoU between existing tracks and new detections
            matched_tracks, unmatched_detections = self._match_detections_to_tracks(detections)
            
            # Update matched tracks
            for track_idx, detection_idx in matched_tracks:
                track = self.tracks[track_idx]
                x1, y1, x2, y2 = detections[detection_idx]
                track['bbox'] = [x1, y1, x2, y2]
                track['hits'] += 1
                track['time_since_update'] = 0
                track['class_id'] = class_ids[detection_idx]
            
            # Mark unmatched tracks
            for track_idx in range(len(self.tracks)):
                if not any(track_idx == m[0] for m in matched_tracks):
                    self.tracks[track_idx]['time_since_update'] += 1
            
            # Add new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._create_new_track(
                    bbox=detections[detection_idx],
                    class_id=class_ids[detection_idx]
                )
        else:
            # No existing tracks, create new tracks for all detections
            for i in range(len(detections)):
                self._create_new_track(
                    bbox=detections[i],
                    class_id=class_ids[i]
                )
        
        # Remove tracks that haven't been updated for too long
        active_tracks = []
        for track in self.tracks:
            if track['time_since_update'] <= self.max_age:
                active_tracks.append(track)
        self.tracks = active_tracks
        
        # Convert tracks to output format (only return confirmed tracks)
        results = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits:
                x1, y1, x2, y2 = track['bbox']
                track_id = track['id']
                class_id = track['class_id']
                results.append([x1, y1, x2, y2, track_id, class_id])
        
        return results
    
    def _create_new_track(self, bbox, class_id):
        """
        Create a new track
        
        Args:
            bbox (list): Bounding box in format [x1, y1, x2, y2]
            class_id (int): Class ID for the detection
        """
        self.tracks.append({
            'id': self.next_id,
            'bbox': bbox,
            'hits': 1,
            'time_since_update': 0,
            'class_id': class_id
        })
        self.next_id += 1
    
    def _match_detections_to_tracks(self, detections):
        """
        Match detections to existing tracks using IoU
        
        Args:
            detections (numpy.ndarray): Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            tuple: (matched_tracks, unmatched_detections)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections)))
        
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        # Calculate IoU between all tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t in range(len(self.tracks)):
            for d in range(len(detections)):
                iou_matrix[t, d] = self._calculate_iou(self.tracks[t]['bbox'], detections[d])
        
        # Match using greedy algorithm
        while True:
            # Find highest IoU
            if np.max(iou_matrix) < self.iou_threshold:
                break
            
            # Get indices of maximum IoU
            track_idx, detection_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            # Add to matches
            matched_tracks.append((track_idx, detection_idx))
            unmatched_detections.remove(detection_idx)
            
            # Remove from consideration
            iou_matrix[track_idx, :] = 0
            iou_matrix[:, detection_idx] = 0
        
        return matched_tracks, unmatched_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            bbox1 (list): First bounding box in format [x1, y1, x2, y2]
            bbox2 (list): Second bounding box in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
