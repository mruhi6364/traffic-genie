/**
 * Main JavaScript for Smart Traffic Management System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
    
    // Add event listener for the upload form if it exists
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('video');
            
            // Check if a file is selected
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a video file first');
                return false;
            }
            
            // Validate file size
            if (fileInput && fileInput.files[0] && fileInput.files[0].size > 500 * 1024 * 1024) {
                e.preventDefault();
                alert('File size exceeds 500MB limit');
                return false;
            }
            
            // Display loading spinner
            const uploadBtn = document.getElementById('upload-btn');
            if (uploadBtn) {
                uploadBtn.disabled = true;
                uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Uploading...';
            }
            
            return true;
        });
    }
    
    // Handle video feed errors
    const videoFeed = document.getElementById('videoFeed');
    if (videoFeed) {
        videoFeed.onerror = function() {
            console.error('Error loading video feed');
            videoFeed.style.display = 'none';
            
            // Create error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger m-3';
            errorDiv.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i> Error loading video feed. Please try again.';
            
            // Insert error message before the video
            videoFeed.parentNode.insertBefore(errorDiv, videoFeed);
        };
    }
});

/**
 * Format number with commas for thousands
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Update the statistics display
 * @param {Object} stats - Statistics object
 */
function updateStatistics(stats) {
    if (!stats) return;
    
    // Update count displays if they exist
    if (document.getElementById('totalVehicles')) {
        document.getElementById('totalVehicles').textContent = formatNumber(stats.vehicle_count);
    }
    
    if (document.getElementById('framesProcessed')) {
        document.getElementById('framesProcessed').textContent = formatNumber(stats.detection_count);
    }
    
    if (document.getElementById('processingFps')) {
        document.getElementById('processingFps').textContent = `${stats.processing_fps.toFixed(1)} FPS`;
    }
    
    // Update vehicle type counts
    const vehicleTypes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle'];
    vehicleTypes.forEach(type => {
        const countElement = document.getElementById(`${type}Count`);
        if (countElement && stats.class_counts) {
            countElement.textContent = formatNumber(stats.class_counts[type] || 0);
        }
    });
}
