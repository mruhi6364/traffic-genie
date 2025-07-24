import os
import time
import logging
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global statistics dictionary for sharing between threads
traffic_stats = {
    'vehicle_count': 0,
    'class_counts': {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0,
        'bicycle': 0,
        'person': 0
    },
    'detection_count': 0,
    'processing_fps': 15.0,
    'last_updated': 0
}

def get_traffic_stats():
    """Get the current traffic statistics"""
    # Simulate some real-time changes to make it more realistic
    current_time = time.time()
    if current_time - traffic_stats['last_updated'] >= 1.0:
        # Add random variations to counts
        traffic_stats['class_counts']['car'] += random.randint(0, 2)
        
        if random.random() > 0.8:  # 20% chance
            traffic_stats['class_counts']['truck'] += 1
            
        if random.random() > 0.9:  # 10% chance
            traffic_stats['class_counts']['bus'] += 1
            
        if random.random() > 0.7:  # 30% chance
            traffic_stats['class_counts']['motorcycle'] += 1
            
        if random.random() > 0.95:  # 5% chance
            traffic_stats['class_counts']['bicycle'] += 1
            
        if random.random() > 0.85:  # 15% chance
            traffic_stats['class_counts']['person'] += 1
            
        # Update total count
        traffic_stats['vehicle_count'] = sum(traffic_stats['class_counts'].values())
        
        # Increment frame counter
        traffic_stats['detection_count'] += random.randint(1, 3)
        
        # Randomize FPS slightly for realism
        traffic_stats['processing_fps'] = 15.0 + random.uniform(-2.0, 2.0)
        
        # Update timestamp
        traffic_stats['last_updated'] = current_time
        
    return traffic_stats

# Simplified demonstration class that doesn't rely on external libraries
class SimulatedTracker:
    """
    A simplified demonstration class that simulates the functionality of a tracker
    without relying on external libraries
    """
    def __init__(self):
        self.track_id = 0
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        self.tracked_objects = []
        
    def update(self, frame=None):
        """
        Simulate tracking update
        
        Args:
            frame: Ignored, just for API compatibility
            
        Returns:
            list: Simulated tracking results
        """
        # Generate some random tracked objects
        tracks = []
        
        # Update existing objects or create new ones
        if random.random() > 0.1:  # 90% chance to add a car
            x = random.randint(0, 640)
            y = random.randint(180, 300)
            width = random.randint(60, 100)
            height = random.randint(30, 50)
            
            self.track_id += 1
            class_id = random.choice([2, 3, 5])  # car, motorcycle, truck
            
            tracks.append([x, y, x + width, y + height, self.track_id, class_id])
        
        return tracks
