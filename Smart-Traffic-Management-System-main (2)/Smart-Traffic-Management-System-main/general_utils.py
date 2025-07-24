import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_directory(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def calculate_fps(start_time, num_frames=1):
    """
    Calculate FPS
    
    Args:
        start_time (float): Start time
        num_frames (int): Number of frames processed
        
    Returns:
        float: FPS
    """
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time if elapsed_time > 0 else 0
    return fps

def resize_frame(frame, max_width=1280):
    """
    Resize frame keeping aspect ratio
    
    Args:
        frame (numpy.ndarray): Input frame
        max_width (int): Maximum width
        
    Returns:
        numpy.ndarray: Resized frame
    """
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
    return frame

def get_video_info(video_path):
    """
    Get video information
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        tuple: (width, height, fps, total_frames)
    """
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None, None, None, None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None, None, None, None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        return width, height, fps, total_frames
    
    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return None, None, None, None

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1 (tuple): First point (x, y)
        point2 (tuple): Second point (x, y)
        
    Returns:
        float: Distance
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def estimate_speed(points, fps, pixels_per_meter=10):
    """
    Estimate speed from tracking points
    
    Args:
        points (list): List of tracking points
        fps (float): Frames per second
        pixels_per_meter (float): Conversion factor from pixels to meters
        
    Returns:
        float: Speed in km/h
    """
    if len(points) < 2:
        return 0
    
    # Calculate distance traveled
    total_distance = 0
    for i in range(1, len(points)):
        total_distance += calculate_distance(points[i-1], points[i])
    
    # Convert to meters
    distance_meters = total_distance / pixels_per_meter
    
    # Calculate time elapsed
    time_seconds = len(points) / fps
    
    # Calculate speed in m/s
    if time_seconds > 0:
        speed_ms = distance_meters / time_seconds
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        return speed_kmh
    
    return 0
