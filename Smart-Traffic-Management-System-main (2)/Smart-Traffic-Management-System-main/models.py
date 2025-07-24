import os
import json
import uuid
from datetime import datetime

# Simple model storage using files since we're not using an actual database
class VideoFile:
    """
    Model to store video file information
    """
    _storage_path = "video_data.json"
    _data = {}
    
    @classmethod
    def _load_data(cls):
        """
        Load data from storage file
        """
        if os.path.exists(cls._storage_path):
            try:
                with open(cls._storage_path, 'r') as f:
                    cls._data = json.load(f)
            except:
                cls._data = {}
        return cls._data
    
    @classmethod
    def _save_data(cls):
        """
        Save data to storage file
        """
        with open(cls._storage_path, 'w') as f:
            json.dump(cls._data, f)
    
    @classmethod
    def get(cls, video_id):
        """
        Get video by ID
        """
        cls._load_data()
        if video_id in cls._data:
            data = cls._data[video_id]
            return cls(
                id=video_id,
                filename=data['filename'],
                path=data['path'],
                uploaded_at=data.get('uploaded_at')
            )
        return None
    
    @classmethod
    def get_all(cls):
        """
        Get all videos
        """
        cls._load_data()
        return [cls(
            id=video_id,
            filename=data['filename'],
            path=data['path'],
            uploaded_at=data.get('uploaded_at')
        ) for video_id, data in cls._data.items()]
    
    def __init__(self, id=None, filename=None, path=None, uploaded_at=None):
        """
        Initialize a video file
        """
        self.id = id or str(uuid.uuid4())
        self.filename = filename
        self.path = path
        self.uploaded_at = uploaded_at or datetime.now().isoformat()
    
    def save(self):
        """
        Save the video information
        """
        VideoFile._load_data()
        VideoFile._data[self.id] = {
            'filename': self.filename,
            'path': self.path,
            'uploaded_at': self.uploaded_at
        }
        VideoFile._save_data()
        return self
    
    def delete(self):
        """
        Delete the video information and file
        """
        VideoFile._load_data()
        if self.id in VideoFile._data:
            del VideoFile._data[self.id]
            VideoFile._save_data()
        
        # Delete the file
        if os.path.exists(self.path):
            os.remove(self.path)
