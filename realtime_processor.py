"""
Real-time image processing stream for Project Noir
Processes images immediately as they arrive via WebSocket
"""
import asyncio
import json
import base64
import time
import logging
from typing import Dict, Optional
from fastapi import WebSocket
import httpx

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Handles real-time image processing via WebSocket streams"""
    
    def __init__(self, yolo_url: str, depth_url: str):
        self.yolo_url = yolo_url
        self.depth_url = depth_url
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect_stream(self, websocket: WebSocket, user_id: str):
        """Connect a user to the real-time processing stream"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"🔴 Real-time stream connected for user {user_id}")
        
        try:
            while True:
                # Receive frame data via WebSocket
                data = await websocket.receive_text()
                frame_data = json.loads(data)
                
                # Process frame immediately
                await self.process_frame_realtime(user_id, frame_data, websocket)
                
        except Exception as e:
            logger.error(f"❌ WebSocket error for user {user_id}: {e}")
        finally:
            self.disconnect_stream(user_id)
    
    def disconnect_stream(self, user_id: str):
        """Disconnect user from real-time stream"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"🔴 Real-time stream disconnected for user {user_id}")
    
    async def process_frame_realtime(self, user_id: str, frame_data: dict, websocket: WebSocket):
        """Process frame in real-time and send results back immediately"""
        try:
            start_time = time.time()
            
            # Extract image data
            image_base64 = frame_data.get("image_data", "")
            if not image_base64:
                await websocket.send_json({"error": "No image data provided"})
                return
            
            # Decode image
            image_bytes = base64.b64decode(image_base64)
            
            # Send immediate acknowledgment
            await websocket.send_json({
                "type": "frame_received",
                "timestamp": time.time(),
                "status": "processing"
            })
            
            # Process with YOLO and Depth in parallel (fastest possible)
            async def quick_yolo():
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                        response = await client.post(f"{self.yolo_url}/detect", files=files)
                        if response.status_code == 200:
                            return response.json()
                except Exception as e:
                    logger.warning(f"Quick YOLO failed: {e}")
                return {"detections": []}
            
            async def quick_depth():
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                        response = await client.post(f"{self.depth_url}/estimate-depth", files=files)
                        if response.status_code == 200:
                            return response.json()
                except Exception as e:
                    logger.warning(f"Quick depth failed: {e}")
                return {}
            
            # Execute in parallel
            yolo_result, depth_result = await asyncio.gather(
                quick_yolo(), 
                quick_depth(), 
                return_exceptions=True
            )
            
            processing_time = time.time() - start_time
            
            # Send results immediately
            await websocket.send_json({
                "type": "processing_complete",
                "user_id": user_id,
                "timestamp": time.time(),
                "processing_time": processing_time,
                "results": {
                    "objects": [d.get("class_name", "") for d in yolo_result.get("detections", [])],
                    "object_count": len(yolo_result.get("detections", [])),
                    "depth_range": {
                        "min": depth_result.get("min_depth", 0),
                        "max": depth_result.get("max_depth", 0)
                    } if depth_result else None
                },
                "status": "success"
            })
            
            logger.info(f"⚡ Real-time processing completed in {processing_time:.2f}s for user {user_id}")
            
        except Exception as e:
            logger.error(f"❌ Real-time processing error: {e}")
            await websocket.send_json({
                "type": "processing_error", 
                "error": str(e),
                "timestamp": time.time()
            })
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected streams"""
        disconnected = []
        
        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {user_id}: {e}")
                disconnected.append(user_id)
        
        # Clean up disconnected clients
        for user_id in disconnected:
            self.disconnect_stream(user_id)

# Global real-time processor instance
realtime_processor = None