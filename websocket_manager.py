"""
WebSocket Manager for Project Noir Cloud
Handles real-time communication between devices and services
"""

import json
import logging
import asyncio
from typing import Dict, Set, Optional, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from models import WebSocketMessage, MessageType
from location_memory import LocationMemoryManager

logger = logging.getLogger(__name__)
# Local location manager for storing GPS updates
location_manager = LocationMemoryManager()

class ConnectionManager:
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.connection_metadata: Dict[str, Dict] = {}  # connection_id -> metadata
        self.room_connections: Dict[str, Set[str]] = {}  # room_id -> set of connection_ids
        
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     user_id: str = None, device_type: str = "unknown") -> bool:
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[connection_id] = websocket
            
            # Store metadata
            self.connection_metadata[connection_id] = {
                "user_id": user_id,
                "device_type": device_type,
                "connected_at": datetime.utcnow(),
                "last_ping": datetime.utcnow()
            }
            
            # Associate with user if provided
            if user_id:
                if user_id not in self.user_connections:
                    self.user_connections[user_id] = set()
                self.user_connections[user_id].add(connection_id)
            
            logger.info(f"🔗 Connection {connection_id} established for user {user_id} ({device_type})")
            
            # Send welcome message
            welcome_msg = WebSocketMessage(
                type=MessageType.SYSTEM,
                data={"message": "Connected to Project Noir Cloud", "connection_id": connection_id},
                timestamp=datetime.utcnow()
            )
            await self.send_to_connection(connection_id, welcome_msg.dict())
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket {connection_id}: {e}")
            return False
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        try:
            # Get metadata before removal
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            
            # Remove from active connections
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            # Remove from rooms
            for room_id, connections in self.room_connections.items():
                connections.discard(connection_id)
            
            # Remove metadata
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            logger.info(f"🔌 Connection {connection_id} disconnected (user: {user_id})")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def send_to_connection(self, connection_id: str, message: Dict) -> bool:
        """Send message to specific connection"""
        try:
            if connection_id in self.active_connections:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message, default=str))
                return True
            else:
                logger.warning(f"Connection {connection_id} not found")
                return False
                
        except WebSocketDisconnect:
            logger.warning(f"Connection {connection_id} disconnected during send")
            self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            return False
    
    async def send_to_user(self, user_id: str, message: Dict) -> int:
        """Send message to all connections for a user"""
        sent_count = 0
        
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])  # Copy to avoid modification during iteration
            
            for connection_id in connection_ids:
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
        
        if sent_count > 0:
            logger.info(f"📤 Sent message to {sent_count} connections for user {user_id}")
        
        return sent_count
    
    async def broadcast_to_room(self, room_id: str, message: Dict, exclude_connection: str = None) -> int:
        """Broadcast message to all connections in a room"""
        sent_count = 0
        
        if room_id in self.room_connections:
            connection_ids = list(self.room_connections[room_id])
            
            for connection_id in connection_ids:
                if connection_id != exclude_connection:
                    if await self.send_to_connection(connection_id, message):
                        sent_count += 1
        
        logger.info(f"📡 Broadcast to {sent_count} connections in room {room_id}")
        return sent_count
    
    async def join_room(self, connection_id: str, room_id: str) -> bool:
        """Add connection to a room"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            if room_id not in self.room_connections:
                self.room_connections[room_id] = set()
            
            self.room_connections[room_id].add(connection_id)
            
            logger.info(f"🏠 Connection {connection_id} joined room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining room: {e}")
            return False
    
    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Remove connection from a room"""
        try:
            if room_id in self.room_connections:
                self.room_connections[room_id].discard(connection_id)
                
                # Clean up empty rooms
                if not self.room_connections[room_id]:
                    del self.room_connections[room_id]
            
            logger.info(f"🚪 Connection {connection_id} left room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving room: {e}")
            return False
    
    async def send_audio_data(self, user_id: str, audio_data: bytes, audio_type: str = "tts") -> bool:
        """Send audio data to user's devices"""
        message = WebSocketMessage(
            type=MessageType.AUDIO_DATA,
            data={
                "audio_type": audio_type,
                "audio_data": audio_data.hex(),  # Convert to hex for JSON transmission
                "format": "mp3"
            },
            timestamp=datetime.utcnow()
        )
        
        sent_count = await self.send_to_user(user_id, message.dict())
        return sent_count > 0
    
    async def send_navigation_update(self, user_id: str, navigation_data: Dict) -> bool:
        """Send navigation guidance to user"""
        message = WebSocketMessage(
            type=MessageType.NAVIGATION,
            data=navigation_data,
            timestamp=datetime.utcnow()
        )
        
        sent_count = await self.send_to_user(user_id, message.dict())
        return sent_count > 0
    
    async def send_detection_results(self, user_id: str, detections: List[Dict]) -> bool:
        """Send object detection results to user"""
        message = WebSocketMessage(
            type=MessageType.DETECTION_RESULT,
            data={"detections": detections},
            timestamp=datetime.utcnow()
        )
        
        sent_count = await self.send_to_user(user_id, message.dict())
        return sent_count > 0
    
    async def send_scene_analysis(self, user_id: str, analysis: Dict) -> bool:
        """Send scene analysis to user"""
        message = WebSocketMessage(
            type=MessageType.SCENE_ANALYSIS,
            data=analysis,
            timestamp=datetime.utcnow()
        )
        
        sent_count = await self.send_to_user(user_id, message.dict())
        return sent_count > 0
    
    async def ping_connections(self):
        """Send ping to all active connections to check health"""
        current_time = datetime.utcnow()
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                ping_message = WebSocketMessage(
                    type=MessageType.PING,
                    data={"timestamp": current_time},
                    timestamp=current_time
                )
                
                await websocket.send_text(json.dumps(ping_message.dict(), default=str))
                
                # Update last ping time
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_ping"] = current_time
                
            except WebSocketDisconnect:
                disconnected_connections.append(connection_id)
            except Exception as e:
                logger.error(f"Error pinging connection {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
        
        if disconnected_connections:
            logger.info(f"🧹 Cleaned up {len(disconnected_connections)} disconnected connections")
    
    def get_connection_stats(self) -> Dict:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "users_connected": len(self.user_connections),
            "active_rooms": len(self.room_connections),
            "connections_by_device": self._get_device_stats()
        }
    
    def _get_device_stats(self) -> Dict[str, int]:
        """Get connection count by device type"""
        device_stats = {}
        
        for metadata in self.connection_metadata.values():
            device_type = metadata.get("device_type", "unknown")
            device_stats[device_type] = device_stats.get(device_type, 0) + 1
        
        return device_stats
    
    async def handle_message(self, connection_id: str, message_data: str):
        """Handle incoming WebSocket message (entry point from main.py)"""
        try:
            # Parse JSON message
            message = json.loads(message_data)
            logger.info(f"📨 Received message from {connection_id}: {message.get('type', 'unknown')}")
            
            # Extract message components
            message_type = message.get("type")
            data = message.get("data", {})
            
            # Handle different message types
            if message_type == "connect":
                # Update connection metadata with device info
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id].update({
                        "device_type": data.get("device_type", "unknown"),
                        "last_activity": datetime.utcnow()
                    })
                logger.info(f"✅ Device {data.get('device_type')} connected for {connection_id}")
                
            elif message_type == "ping":
                # Handle heartbeat ping
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_ping"] = datetime.utcnow()
                
                # Send pong response
                pong_response = {
                    "type": "pong", 
                    "data": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "connection_id": connection_id
                    }
                }
                await self.send_to_connection(connection_id, pong_response)
                logger.debug(f"🏓 Sent pong to {connection_id}")
                
            elif message_type == "gps_update":
                # Handle GPS location data
                gps_data = data
                logger.info(f"📍 GPS update from {connection_id}: lat={gps_data.get('latitude')}, lon={gps_data.get('longitude')}")
                
                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()
                    
                # Here you could process the GPS data, store it, or forward it to other services
                # Store current location in Firestore
                user = gps_data.get('user_id') or self.connection_metadata.get(connection_id, {}).get('user_id')
                location_manager.update_current_location(
                    user_id=user,
                    latitude=gps_data.get('latitude', 0),
                    longitude=gps_data.get('longitude', 0),
                    altitude=gps_data.get('altitude', 0)
                )
                # Acknowledge receipt
                ack_response = {
                    "type": "gps_ack",
                    "data": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "received"
                    }
                }
                await self.send_to_connection(connection_id, ack_response)
                
            else:
                # Delegate to existing handler for other message types
                await self.handle_connection_message(connection_id, message)
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON from {connection_id}: {e}")
        except Exception as e:
            logger.error(f"❌ Error handling message from {connection_id}: {e}")

    async def handle_connection_message(self, connection_id: str, message: Dict):
        """Handle incoming message from WebSocket connection"""
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            if message_type == MessageType.PONG:
                # Update last ping time
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]["last_ping"] = datetime.utcnow()
            
            elif message_type == MessageType.JOIN_ROOM:
                room_id = data.get("room_id")
                if room_id:
                    await self.join_room(connection_id, room_id)
            
            elif message_type == MessageType.LEAVE_ROOM:
                room_id = data.get("room_id")
                if room_id:
                    await self.leave_room(connection_id, room_id)
            
            elif message_type == MessageType.AUDIO_STREAM:
                # Handle incoming audio stream data
                await self._handle_audio_stream(connection_id, data)
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling connection message: {e}")
    
    async def _handle_audio_stream(self, connection_id: str, data: Dict):
        """Handle incoming audio stream data"""
        try:
            # Audio stream data would be processed here
            # This could trigger speech-to-text processing
            audio_data = data.get("audio_data")
            
            if audio_data:
                # Convert hex back to bytes
                audio_bytes = bytes.fromhex(audio_data)
                
                # TODO: Process audio stream with speech-to-text
                logger.info(f"🎤 Received audio stream from connection {connection_id}")
                
        except Exception as e:
            logger.error(f"Error handling audio stream: {e}")

# Global connection manager instance
connection_manager = ConnectionManager()

# Alias for backwards compatibility
WebSocketManager = ConnectionManager

async def cleanup_connections_task():
    """Background task to clean up stale connections"""
    while True:
        try:
            await connection_manager.ping_connections()
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(5)
