"""
Real-time frame buffer for immediate image processing
Maintains rolling buffer of 3 most recent frames per user
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FrameBuffer:
    """Thread-safe rolling frame buffer - keeps 3 most recent frames per user"""
    
    def __init__(self, max_frames_per_user: int = 3, max_age_seconds: int = 10):
        self.frames: Dict[str, List[Tuple[bytes, float]]] = {}  # user_id -> [(image_bytes, timestamp), ...]
        self.max_frames = max_frames_per_user
        self.max_age = max_age_seconds
        self._lock = asyncio.Lock()
    
    async def store_frame(self, user_id: str, image_bytes: bytes) -> None:
        """Store frame in rolling buffer (keeps only 3 most recent)"""
        async with self._lock:
            timestamp = time.time()
            
            # Initialize user's frame list if doesn't exist
            if user_id not in self.frames:
                self.frames[user_id] = []
            
            # Add new frame to the end
            self.frames[user_id].append((image_bytes, timestamp))
            
            # Keep only the most recent 3 frames
            if len(self.frames[user_id]) > self.max_frames:
                old_frame = self.frames[user_id].pop(0)  # Remove oldest
                logger.debug(f"🔄 Removed oldest frame for user {user_id}")
            
            logger.info(f"📸 Stored frame for user {user_id} ({len(image_bytes)} bytes) - Buffer: {len(self.frames[user_id])}/3")
    
    async def get_latest_frame(self, user_id: str) -> Optional[bytes]:
        """Get most recent frame if available and fresh"""
        async with self._lock:
            if user_id not in self.frames or not self.frames[user_id]:
                logger.warning(f"No frames available for user {user_id}")
                return None
            
            # Get the most recent frame (last in list)
            image_bytes, timestamp = self.frames[user_id][-1]
            age = time.time() - timestamp
            
            if age > self.max_age:
                logger.warning(f"Latest frame too old for user {user_id} (age: {age:.1f}s)")
                return None
            
            logger.info(f"✅ Retrieved latest frame for user {user_id} (age: {age:.1f}s, buffer: {len(self.frames[user_id])}/3)")
            return image_bytes
    
    async def get_frame_by_index(self, user_id: str, index: int = -1) -> Optional[bytes]:
        """Get frame by index (-1 = latest, -2 = second latest, -3 = oldest)"""
        async with self._lock:
            if user_id not in self.frames or not self.frames[user_id]:
                return None
            
            try:
                image_bytes, timestamp = self.frames[user_id][index]
                age = time.time() - timestamp
                
                if age > self.max_age:
                    return None
                    
                return image_bytes
            except IndexError:
                return None
    
    async def get_buffer_status(self, user_id: str) -> Dict:
        """Get detailed buffer status for debugging"""
        async with self._lock:
            if user_id not in self.frames:
                return {"frames_count": 0, "frames": []}
            
            current_time = time.time()
            frame_info = []
            
            for i, (image_bytes, timestamp) in enumerate(self.frames[user_id]):
                age = current_time - timestamp
                frame_info.append({
                    "index": i,
                    "size_bytes": len(image_bytes),
                    "age_seconds": round(age, 2),
                    "fresh": age <= self.max_age
                })
            
            return {
                "frames_count": len(self.frames[user_id]),
                "frames": frame_info
            }
    
    async def cleanup_old_frames(self) -> None:
        """Remove frames older than max_age"""
        async with self._lock:
            current_time = time.time()
            
            for user_id in list(self.frames.keys()):
                # Filter out old frames
                fresh_frames = [
                    (image_bytes, timestamp)
                    for image_bytes, timestamp in self.frames[user_id]
                    if current_time - timestamp <= self.max_age
                ]
                
                removed_count = len(self.frames[user_id]) - len(fresh_frames)
                
                if removed_count > 0:
                    self.frames[user_id] = fresh_frames
                    logger.info(f"🗑️ Removed {removed_count} expired frames for user {user_id}")
                
                # Remove user entirely if no frames left
                if not self.frames[user_id]:
                    del self.frames[user_id]


# Global frame buffer instance - rolling buffer of 3 frames
frame_buffer = FrameBuffer(max_frames_per_user=3, max_age_seconds=10)

# Periodic cleanup coroutine for use in FastAPI lifespan or background tasks
async def periodic_frame_cleanup(interval: int = 5):
    """Periodically clean up old frames from the buffer every `interval` seconds."""
    while True:
        await frame_buffer.cleanup_old_frames()
        await asyncio.sleep(interval)