"""
Real-time frame buffer for immediate image processing
Eliminates Cloud Storage retrieval delays
"""
import asyncio
import time
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FrameBuffer:
    """Thread-safe in-memory frame buffer for real-time processing"""
    
    def __init__(self, max_age_seconds: int = 5):
        self.frames: Dict[str, Tuple[bytes, float]] = {}  # user_id -> (image_bytes, timestamp)
        self.max_age = max_age_seconds
        self._lock = asyncio.Lock()
    
    async def store_frame(self, user_id: str, image_bytes: bytes) -> None:
        """Store latest frame for user"""
        async with self._lock:
            timestamp = time.time()
            self.frames[user_id] = (image_bytes, timestamp)
            logger.info(f"📸 Stored fresh frame for user {user_id} ({len(image_bytes)} bytes)")
    
    async def get_latest_frame(self, user_id: str) -> Optional[bytes]:
        """Get latest frame if recent enough"""
        async with self._lock:
            if user_id not in self.frames:
                logger.warning(f"No frame available for user {user_id}")
                return None
            
            image_bytes, timestamp = self.frames[user_id]
            age = time.time() - timestamp
            
            if age > self.max_age:
                logger.warning(f"Frame too old for user {user_id} (age: {age:.1f}s)")
                return None
            
            logger.info(f"✅ Retrieved fresh frame for user {user_id} (age: {age:.1f}s)")
            return image_bytes
    
    async def cleanup_old_frames(self) -> None:
        """Remove stale frames"""
        async with self._lock:
            current_time = time.time()
            expired_users = [
                user_id for user_id, (_, timestamp) in self.frames.items()
                if current_time - timestamp > self.max_age
            ]
            
            for user_id in expired_users:
                del self.frames[user_id]
                logger.info(f"🗑️ Removed expired frame for user {user_id}")

# Global frame buffer instance
frame_buffer = FrameBuffer(max_age_seconds=5)