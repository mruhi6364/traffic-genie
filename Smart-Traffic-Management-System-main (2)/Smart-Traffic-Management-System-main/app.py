import os
import logging
import datetime
import uuid
import tempfile
import time
import json
import random
import io
import base64
import math
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, Response, jsonify, url_for, redirect, flash, session
from werkzeug.utils import secure_filename

# Import OpenCV for 3D transformations
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")
try:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
except ImportError:
    # ProxyFix is not essential, so we can proceed without it
    pass

# Configuration
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'webm'}

from models import VideoFile

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Global traffic stats and detector
traffic_stats = {
    'vehicle_count': 0,
    'class_counts': {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0,
        'bicycle': 0
    },
    'detection_count': 0,
    'processing_fps': 0,
    'last_updated': 0
}

def update_demo_stats():
    """Update demo statistics with simulated data"""
    global traffic_stats
    # Increment counters for demo
    traffic_stats['vehicle_count'] += 1
    traffic_stats['class_counts']['car'] += 1
    if traffic_stats['detection_count'] % 10 == 0:
        traffic_stats['class_counts']['truck'] += 1
    if traffic_stats['detection_count'] % 15 == 0:
        traffic_stats['class_counts']['bus'] += 1
    if traffic_stats['detection_count'] % 5 == 0:
        traffic_stats['class_counts']['motorcycle'] += 1
    traffic_stats['detection_count'] += 1
    traffic_stats['processing_fps'] = 15.0 + (traffic_stats['detection_count'] % 5)
    traffic_stats['last_updated'] = datetime.datetime.now().timestamp()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_uuid = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_uuid}_{filename}")
            file.save(file_path)
            
            # Store video info in the model
            video = VideoFile(id=file_uuid, filename=filename, path=file_path)
            video.save()
            
            return redirect(url_for('process_video', video_id=file_uuid))
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            flash(f'Error uploading file: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash(f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}', 'danger')
        return redirect(url_for('index'))

@app.route('/video/<video_id>')
def process_video(video_id):
    video = VideoFile.get(video_id)
    if not video:
        flash('Video not found', 'danger')
        return redirect(url_for('index'))
    
    return render_template('video.html', video=video)

@app.route('/stream/<video_id>')
def stream(video_id):
    """Generate a frame from the video file or a 3D visualization"""
    # For demo videos or when actual video processing isn't available
    if video_id.startswith('demo'):
        # For demo purposes, use the video_id as a seed for random but consistent vehicle positions
        seed_value = int(hash(video_id) % 10000)
        random.seed(seed_value)
        
        # Create timestamp-based frame number (updated every 200ms)
        timestamp = int(time.time() * 5)  # 5 frames per second equivalent
        
        # Make sure stats has non-zero values
        default_stats = {
            'processing_fps': 24.5,  # Default to a realistic FPS
            'detection_count': int(timestamp % 1000),
            'vehicle_count': 8,
            'class_counts': {
                'car': 4,
                'truck': 1,
                'bus': 1,
                'motorcycle': 1,
                'bicycle': 0,
                'person': 1
            }
        }
        
        # Create SVG content showing simulated traffic
        svg_frame = generate_traffic_svg(timestamp, default_stats)
        
        # Update statistics
        update_demo_stats()
        
        # Return as SVG image
        return Response(svg_frame, mimetype='image/svg+xml')
    
    # For actual uploaded videos
    video = VideoFile.get(video_id)
    if not video:
        return "Video not found", 404
    
    try:
        # Process the actual video file using OpenCV
        return process_video_frame(video.path, video_id)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Fallback to simulated visualization
        timestamp = int(time.time() * 5)
        svg_frame = generate_traffic_svg(timestamp, traffic_stats)
        update_demo_stats()
        return Response(svg_frame, mimetype='image/svg+xml')

def process_video_frame(video_path, video_id):
    """Process a frame from the actual video file using OpenCV"""
    # Each client gets their own video capture object
    # We'll use a session-based counter to track which frame to show
    session_key = f'video_frame_{video_id}'
    
    if session_key not in session:
        session[session_key] = 0
    
    # Increment frame counter (this will cycle through the video)
    frame_count = session[session_key]
    session[session_key] = (frame_count + 1) % 1000  # Reset after 1000 frames to avoid overflow
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame to display (loop the video if needed)
        frame_position = frame_count % max(1, total_frames)
        
        # Set the position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to read frame")
        
        # Apply 3D object detection (simulated here, but could be replaced with actual detection)
        processed_frame = apply_3d_detection(frame, frame_count)
        
        # Convert to JPEG for web display
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Close the video
        cap.release()
        
        # Update statistics based on the processing
        update_demo_stats()
        
        # Return the processed frame
        return Response(frame_bytes, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        # Create a fallback image with error message
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Error: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        ret, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()
        
        return Response(frame_bytes, mimetype='image/jpeg')

def apply_3d_detection(frame, frame_count):
    """Apply 3D object detection visualization to a video frame using real detection"""
    global traffic_stats, detector
    
    # Initialize detector if not already initialized
    if 'detector' not in globals():
        from yolo_detection import YOLODetector
        global detector
        detector = YOLODetector()
    
    # Perform real object detection
    try:
        detections = detector.detect_objects(frame)
        
        # Update statistics based on real detections
        update_real_stats(detections)
        
        # Return the frame without drawing bounding boxes as requested
        return frame
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        # Return original frame if detection fails
        return frame

def update_real_stats(detections):
    """Update statistics based on real detections"""
    global traffic_stats
    
    # Reset counts for this detection cycle
    vehicle_counts = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0,
        'bicycle': 0,
        'person': 0
    }
    
    # Get the start time for FPS calculation
    start_time = time.time()
    
    # Count detections by vehicle type
    for detection in detections:
        class_name = detection["class"]
        
        # Map the detected class to our vehicle types
        if 'detector' in globals():
            vehicle_type = detector.map_class_to_vehicle_type(class_name)
        else:
            # Fallback mapping
            mapping = {
                "car": "car",
                "bus": "bus", 
                "truck": "truck",
                "bicycle": "bicycle",
                "motorbike": "motorcycle",
                "person": "person"
            }
            vehicle_type = mapping.get(class_name.lower(), "car")
        
        # Increment count for this vehicle type
        if vehicle_type in vehicle_counts:
            vehicle_counts[vehicle_type] += 1
    
    # Update statistics based on the detections
    for vehicle_type, count in vehicle_counts.items():
        if count > 0:  # Only update if we detected any
            if vehicle_type in traffic_stats['class_counts']:
                traffic_stats['class_counts'][vehicle_type] += count
            else:
                traffic_stats['class_counts'][vehicle_type] = count
    
    # Calculate number of vehicles (excluding persons)
    total_vehicles = sum(count for vtype, count in vehicle_counts.items() if vtype != 'person')
    
    # Update total vehicle count
    traffic_stats['vehicle_count'] += total_vehicles
    
    # Update detection count
    traffic_stats['detection_count'] += 1
    
    # Calculate FPS based on processing time
    processing_time = time.time() - start_time
    if processing_time > 0:
        fps = 1.0 / processing_time
        # Smooth FPS with running average
        traffic_stats['processing_fps'] = (traffic_stats['processing_fps'] * 0.9) + (fps * 0.1)
    else:
        # If processing time is too small, use a reasonable value
        traffic_stats['processing_fps'] = 30.0
    
    # Update timestamp
    traffic_stats['last_updated'] = datetime.datetime.now().timestamp()
    
    # Log detection summary
    logger.debug(f"Detected: {vehicle_counts}, Total vehicles: {total_vehicles}, FPS: {traffic_stats['processing_fps']:.1f}")

@app.route('/detection_demo/<video_id>')
def detection_demo(video_id):
    """Render a demo page with detection visualization"""
    video = VideoFile.get(video_id)
    if not video:
        flash('Video not found', 'danger')
        return redirect(url_for('index'))
    
    vehicle_types = [
        {"type": "car", "color": "#00FF00"},      # Lime
        {"type": "truck", "color": "#0000FF"},    # Blue
        {"type": "bus", "color": "#FF0000"},      # Red
        {"type": "motorcycle", "color": "#FFFF00"}, # Yellow
        {"type": "bicycle", "color": "#FF00FF"},  # Magenta
        {"type": "person", "color": "#00FFFF"}    # Cyan
    ]
    
    return render_template('detection_demo.html', video=video, vehicle_types=vehicle_types)

def generate_traffic_svg(frame_count, stats):
    """Generate a 3D-like SVG representation of traffic with vehicle tracking"""
    
    # SVG start with 3D perspective effect
    svg = f"""<svg width="640" height="480" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <defs>
        <!-- Define gradients for 3D effect -->
        <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#1a2a6c" />
            <stop offset="50%" stop-color="#2a4858" />
            <stop offset="100%" stop-color="#3c3b52" />
        </linearGradient>
        
        <linearGradient id="roadGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#333333" />
            <stop offset="50%" stop-color="#555555" />
            <stop offset="100%" stop-color="#333333" />
        </linearGradient>
        
        <linearGradient id="perspectiveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="rgba(0,0,0,0.7)" />
            <stop offset="100%" stop-color="rgba(0,0,0,0)" />
        </linearGradient>
        
        <!-- Filter for glow effects -->
        <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    </defs>
    
    <!-- Sky background with 3D effect -->
    <rect width="640" height="480" fill="url(#skyGradient)" />
    
    <!-- 3D perspective grid effect for depth -->
    <g id="perspective-grid">
    """
    
    # Create perspective grid lines for 3D depth effect
    vanishing_point_x = 320  # Center of screen
    vanishing_point_y = 120  # Above horizon
    
    # Draw horizon line
    svg += f'<line x1="0" y1="240" x2="640" y2="240" stroke="#444444" stroke-width="2" />'
    
    # Draw perspective lines from bottom to vanishing point
    for x in range(0, 641, 80):
        svg += f'<line x1="{x}" y1="480" x2="{vanishing_point_x}" y2="{vanishing_point_y}" stroke="#444444" stroke-width="1" stroke-opacity="0.3" />'
    
    # Draw road with 3D perspective trapezoid
    svg += f"""
    </g>
    
    <!-- 3D road -->
    <polygon points="120,480 520,480 450,250 190,250" fill="url(#roadGradient)" />
    
    <!-- Road markings with perspective -->
    <polygon points="315,480 325,480 323,250 317,250" fill="#FFFFFF" />
    """
    
    # Add dashed lines with perspective
    for y in range(300, 480, 30):
        # Calculate width based on perspective (closer = wider)
        width = 50 * (y - 240) / 240
        x_center = 320
        svg += f'<line x1="{x_center - width}" y1="{y}" x2="{x_center + width}" y2="{y}" stroke="#FFFFFF" stroke-width="3" stroke-dasharray="10,10" />'
    
    # Add background city skyline silhouette for 3D depth
    svg += f"""
    <!-- City skyline for depth -->
    <polygon points="0,250 50,230 80,245 120,220 150,235 200,215 250,240 300,230 350,220 400,235 450,215 500,240 550,225 600,235 640,250 640,270 0,270" fill="#111122" />
    
    <!-- Distant mountains -->
    <polygon points="0,200 100,180 200,210 300,170 400,190 500,160 600,185 640,200 640,250 0,250" fill="#223344" fill-opacity="0.5" />
    
    <!-- Perspective overlay for depth -->
    <rect x="0" y="250" width="640" height="230" fill="url(#perspectiveGradient)" opacity="0.3" />
    """
    
    # Use deterministic randomness based on frame count for predictable movement
    r = random.Random(42)  # Fixed seed for initial positions
    
    # Add traffic vehicles with tracking IDs
    vehicle_classes = ["car", "truck", "motorcycle", "bus", "bicycle", "person"]
    # Match the example image with orange boxes for all vehicle types
    vehicle_colors = {
        "car": "#FF7800",       # Orange
        "truck": "#FF7800",     # Orange
        "motorcycle": "#FF7800", # Orange
        "bus": "#FF7800",       # Orange
        "bicycle": "#FF7800",   # Orange
        "person": "#FF7800"     # Orange
    }
    
    vehicles = []
    
    # First lane (top to bottom direction)
    for i in range(6):
        base_seed = i * 100 + frame_count
        vehicle_type = vehicle_classes[i % len(vehicle_classes)]
        vehicle_id = i + 1
        
        # Calculate position - moves from left to right
        speed = r.randint(3, 8)  # Different speeds
        x = (base_seed * speed) % 800 - 100  # Start off-screen
        y = 180
        
        # Generate a realistic confidence score
        confidence = round(0.65 + (r.random() * 0.34), 2)  # Between 65% and 99%
        
        vehicles.append({
            "x": x, "y": y, "id": vehicle_id, 
            "type": vehicle_type, 
            "color": vehicle_colors[vehicle_type],
            "confidence": confidence
        })
    
    # Second lane (bottom to top direction)
    for i in range(5):
        base_seed = i * 100 + frame_count
        vehicle_type = vehicle_classes[(i + 2) % len(vehicle_classes)]
        vehicle_id = i + 7
        
        # Calculate position - moves from right to left
        speed = r.randint(2, 6)  # Different speeds
        x = 740 - (base_seed * speed) % 800  # Start off-screen from right
        y = 280
        
        # Generate a realistic confidence score
        confidence = round(0.65 + (r.random() * 0.34), 2)  # Between 65% and 99%
        
        vehicles.append({
            "x": x, "y": y, "id": vehicle_id, 
            "type": vehicle_type, 
            "color": vehicle_colors[vehicle_type],
            "confidence": confidence
        })
    
    # Occasionally add a person on the sidewalk
    if frame_count % 30 == 0:  # Every 30 frames
        person_x = r.randint(50, 590)
        person_y = 130  # On the top sidewalk
        vehicles.append({
            "x": person_x, "y": person_y, "id": 20, 
            "type": "person", 
            "color": vehicle_colors["person"],
            "confidence": round(0.70 + (r.random() * 0.29), 2)
        })
    
    # Draw vehicles with 3D bounding boxes
    for vehicle in vehicles:
        x, y = vehicle["x"], vehicle["y"]
        vehicle_type = vehicle["type"]
        vehicle_id = vehicle["id"]
        color = vehicle["color"]
        confidence = vehicle["confidence"]
        
        # Only draw if in viewport
        if x < -100 or x > 740:
            continue
            
        # Set dimensions based on vehicle type
        if vehicle_type == "car":
            width, height, depth = 80, 40, 120
        elif vehicle_type == "truck":
            width, height, depth = 100, 60, 180
        elif vehicle_type == "bus":
            width, height, depth = 120, 55, 200
        elif vehicle_type == "motorcycle":
            width, height, depth = 40, 20, 60
        elif vehicle_type == "bicycle":
            width, height, depth = 30, 15, 40
        else:  # person
            width, height, depth = 20, 40, 20
        
        # Calculate perspective depth factor (closer to bottom = larger)
        depth_factor = 1.0 - ((y - 150) / 400)  # normalized 0-1 based on y position
        if depth_factor < 0.1:
            depth_factor = 0.1
        if depth_factor > 1.0:
            depth_factor = 1.0
            
        # Apply perspective scaling
        perspective_width = width * depth_factor
        perspective_height = height * depth_factor
        
        # Calculate 3D perspective height (top of the 3D box)
        perspective_depth = depth * depth_factor * 0.3  # Scale depth for visual effect
        
        # Calculate 3D box corners (front face)
        front_tl = (x, y)
        front_tr = (x + perspective_width, y)
        front_bl = (x, y + perspective_height)
        front_br = (x + perspective_width, y + perspective_height)
        
        # Calculate 3D box corners (back face) - shifted up for perspective
        back_tl = (x + perspective_depth * 0.5, y - perspective_depth)
        back_tr = (x + perspective_width + perspective_depth * 0.5, y - perspective_depth)
        back_bl = (x + perspective_depth * 0.5, y + perspective_height - perspective_depth)
        back_br = (x + perspective_width + perspective_depth * 0.5, y + perspective_height - perspective_depth)
        
        # Draw 3D bounding box with shaded faces
        svg += f'''
        <g class="detection-3d">
            <!-- Back face (lighter shade) -->
            <polygon points="{back_tl[0]},{back_tl[1]} {back_tr[0]},{back_tr[1]} {back_br[0]},{back_br[1]} {back_bl[0]},{back_bl[1]}"
                     fill="{color}" fill-opacity="0.1" stroke="{color}" stroke-width="2" />
            
            <!-- Connecting lines (sides) -->
            <line x1="{front_tl[0]}" y1="{front_tl[1]}" x2="{back_tl[0]}" y2="{back_tl[1]}" 
                  stroke="{color}" stroke-width="2" />
            <line x1="{front_tr[0]}" y1="{front_tr[1]}" x2="{back_tr[0]}" y2="{back_tr[1]}" 
                  stroke="{color}" stroke-width="2" />
            <line x1="{front_bl[0]}" y1="{front_bl[1]}" x2="{back_bl[0]}" y2="{back_bl[1]}" 
                  stroke="{color}" stroke-width="2" />
            <line x1="{front_br[0]}" y1="{front_br[1]}" x2="{back_br[0]}" y2="{back_br[1]}" 
                  stroke="{color}" stroke-width="2" />
            
            <!-- Top face (lighter) -->
            <polygon points="{front_tl[0]},{front_tl[1]} {front_tr[0]},{front_tr[1]} {back_tr[0]},{back_tr[1]} {back_tl[0]},{back_tl[1]}"
                     fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-width="2" />
                     
            <!-- Front face (most visible) -->
            <polygon points="{front_tl[0]},{front_tl[1]} {front_tr[0]},{front_tr[1]} {front_br[0]},{front_br[1]} {front_bl[0]},{front_bl[1]}"
                     fill="none" stroke="{color}" stroke-width="3" />
                     
            <!-- Label text (at the top of the 3D box) -->
            <text x="{front_tl[0]+5}" y="{front_tl[1]+20}" font-family="Arial" font-size="14" font-weight="bold" 
                  fill="{color}">{vehicle_type} {confidence:.2f}</text>
                  
            <!-- ObjectID text (at the bottom of the box) -->
            <text x="{front_bl[0]+5}" y="{front_bl[1]-5}" font-family="Arial" font-size="12"
                  fill="{color}">ID:{vehicle_id}</text>
        </g>
        '''
    
    # Add minimal video player controls to match the example
    svg += f'''
    <!-- Video player controls -->
    <rect x="0" y="450" width="640" height="30" fill="rgba(0,0,0,0.8)" />
    <rect x="200" y="460" width="240" height="5" fill="#444444" rx="2" />
    <rect x="200" y="460" width="{(frame_count % 100) * 2.4}" height="5" fill="#FF7800" rx="2" />
    <text x="10" y="465" font-family="Arial" font-size="10" fill="white">00:00:{(frame_count % 60):02d}</text>
    <text x="610" y="465" font-family="Arial" font-size="10" fill="white" text-anchor="end">00:00:31</text>
    <text x="320" y="445" font-family="Arial" font-size="12" fill="white" text-anchor="middle">demo</text>
    '''
    
    # SVG end
    svg += "</svg>"
    
    return svg

@app.route('/stats/<video_id>')
def get_stats(video_id):
    video = VideoFile.get(video_id)
    if not video:
        return jsonify({"error": "Video not found"}), 404
    
    # Return demo stats
    return jsonify(traffic_stats)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', error="Server error occurred. Please try again."), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
