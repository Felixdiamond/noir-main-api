"""
Real-time HTTP streaming processor for Project Noir
Alternative to WebSocket for Cloud Run compatibility
Uses Server-Sent Events (SSE) for real-time communication
"""
import asyncio
import json
import base64
import time
import logging
from typing import Dict, Optional
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
import httpx

logger = logging.getLogger(__name__)

class RealTimeHTTPProcessor:
    """Handles real-time image processing via HTTP streaming"""
    
    def __init__(self, yolo_url: str, depth_url: str):
        self.yolo_url = yolo_url
        self.depth_url = depth_url
        self.processing_queue = asyncio.Queue()
        
    async def process_frame_immediate(self, user_id: str, image_bytes: bytes) -> dict:
        """Process frame immediately and return results"""
        try:
            start_time = time.time()
            
            # Process with YOLO and Depth in parallel (fastest possible)
            async def quick_yolo():
                try:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                        response = await client.post(f"{self.yolo_url}/detect", files=files)
                        if response.status_code == 200:
                            return response.json()
                except Exception as e:
                    logger.warning(f"Quick YOLO failed: {e}")
                return {"detections": []}
            
            async def quick_depth():
                try:
                    async with httpx.AsyncClient(timeout=12.0) as client:
                        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                        response = await client.post(f"{self.depth_url}/estimate-depth", files=files)
                        if response.status_code == 200:
                            return response.json()
                except Exception as e:
                    logger.warning(f"Quick depth failed: {e}")
                return {}
            
            # Execute in parallel with timeout protection
            try:
                yolo_result, depth_result = await asyncio.wait_for(
                    asyncio.gather(quick_yolo(), quick_depth(), return_exceptions=True),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.warning("Processing timeout, returning partial results")
                yolo_result = {"detections": []}
                depth_result = {}
            
            processing_time = time.time() - start_time
            
            # Format results
            objects = [d.get("class_name", "") for d in yolo_result.get("detections", [])]
            
            result = {
                "success": True,
                "user_id": user_id,
                "timestamp": time.time(),
                "processing_time": processing_time,
                "results": {
                    "objects": objects,
                    "object_count": len(objects),
                    "depth_range": {
                        "min": depth_result.get("min_depth", 0),
                        "max": depth_result.get("max_depth", 0)
                    } if depth_result else None
                }
            }
            
            logger.info(f"⚡ HTTP real-time processing completed in {processing_time:.2f}s for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Real-time HTTP processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

# Global HTTP processor instance
realtime_http_processor = None