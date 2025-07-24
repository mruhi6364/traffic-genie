import cv2
import numpy as np
import random
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define colors for visualization
COLORS = {
    'car': (0, 255, 0),       # Green
    'truck': (0, 0, 255),     # Blue
    'bus': (255, 0, 0),       # Red
    'motorcycle': (255, 255, 0), # Yellow
    'bicycle': (255, 0, 255), # Magenta
    'person': (0, 255, 255),  # Cyan
    'default': (200, 200, 200), # Light gray
}

def get_color(class_id, class_names):
    """
    Get color for a class ID
    
    Args:
        class_id (int): Class ID
        class_names (list): List of class names
        
    Returns:
        tuple: RGB color tuple
    """
    try:
        class_name = class_names[class_id]
        return COLORS.get(class_name, COLORS['default'])
    except:
        # Generate a random color if not found
        return tuple(random.randint(0, 255) for _ in range(3))

def draw_boxes(frame, tracks, class_ids, class_names):
    """
    Draw bounding boxes with tracking IDs
    
    Args:
        frame (numpy.ndarray): Input frame
        tracks (list): List of tracks [x1, y1, x2, y2, track_id, class_id]
        class_ids (numpy.ndarray): Class IDs
        class_names (list): List of class names
        
    Returns:
        numpy.ndarray: Frame with drawn boxes
    """
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        
        # Get color for this class
        color = get_color(int(class_id), class_names)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add class name and track ID
        class_name = class_names[int(class_id)]
        label = f"{class_name} #{int(track_id)}"
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (int(x1), int(y1)-20), (int(x1) + text_width, int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def draw_trails(frame, track_history):
    """
    Draw tracking trails
    
    Args:
        frame (numpy.ndarray): Input frame
        track_history (dict): Track history dictionary {track_id: deque(points)}
        
    Returns:
        numpy.ndarray: Frame with drawn trails
    """
    # Draw the tracking lines
    for track_id, points in track_history.items():
        # Skip if there are not enough points
        if len(points) < 2:
            continue
        
        # Random color for each track
        color = tuple(map(int, np.random.randint(0, 255, size=3)))
        
        # Draw lines connecting consecutive points
        for i in range(1, len(points)):
            # Get start and end points
            start_point = (int(points[i-1][0]), int(points[i-1][1]))
            end_point = (int(points[i][0]), int(points[i][1]))
            
            # Draw line - thicker at the newer end
            thickness = int(np.sqrt(64 / float(i + 1)) * 1.5)
            cv2.line(frame, start_point, end_point, color, thickness)
    
    return frame

def draw_segments(frame, masks, class_ids, class_names, alpha=0.3):
    """
    Draw segmentation masks
    
    Args:
        frame (numpy.ndarray): Input frame
        masks (numpy.ndarray): Segmentation masks
        class_ids (numpy.ndarray): Class IDs
        class_names (list): List of class names
        alpha (float): Transparency of the overlay
        
    Returns:
        numpy.ndarray: Frame with drawn segmentation masks
    """
    try:
        # Create a copy of the frame
        overlay = frame.copy()
        
        # Draw each mask
        for i, mask in enumerate(masks):
            if i >= len(class_ids):
                continue
                
            class_id = class_ids[i]
            color = get_color(class_id, class_names)
            
            # Create binary mask
            binary_mask = mask.astype(bool)
            
            # Apply mask to the overlay
            overlay[binary_mask] = color
        
        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    except Exception as e:
        logger.error(f"Error drawing segments: {str(e)}")
        return frame

def draw_stats(frame, class_counts, fps):
    """
    Draw statistics on the frame
    
    Args:
        frame (numpy.ndarray): Input frame
        class_counts (dict): Dictionary of class counts
        fps (float): Current FPS
        
    Returns:
        numpy.ndarray: Frame with drawn statistics
    """
    # Draw rectangle for stats background
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (250, 120), (255, 255, 255), 1)
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw vehicle counts
    y_pos = 60
    cv2.putText(frame, "Vehicle Counts:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for i, (class_name, count) in enumerate(class_counts.items()):
        y_pos += 25
        cv2.putText(frame, f"{class_name}: {count}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
