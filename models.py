"""
Data models for Project Noir Cloud API
"""

from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum

class CommandIntent(str, Enum):
    SCENE_DESCRIPTION = "scene_description"
    OBJECT_FINDING = "object_finding"
    LOCATION_SAVE = "location_save"
    LOCATION_NAVIGATE = "location_navigate"
    TEXT_READING = "text_reading"
    UNKNOWN = "unknown"

class MessageType(str, Enum):
    """WebSocket message types for real-time communication"""
    SYSTEM = "system"
    AUDIO_DATA = "audio_data"
    NAVIGATION = "navigation"
    DETECTION_RESULT = "detection_result"
    SCENE_ANALYSIS = "scene_analysis"
    PING = "ping"
    PONG = "pong"
    JOIN_ROOM = "join_room"
    LEAVE_ROOM = "leave_room"
    AUDIO_STREAM = "audio_stream"

class GPSData(BaseModel):
    isValid: bool
    latitude: float
    longitude: float
    altitude: Optional[float] = 0.0
    satellites: Optional[int] = 0
    accuracy: Optional[float] = None
    timestamp: Optional[datetime] = None

class FrameData(BaseModel):
    user_id: str = "default"
    session_id: str
    gps_data: GPSData
    distance: int  # From distance sensor
    timestamp: datetime
    image_url: Optional[str] = None
    processed: bool = False

class VoiceCommand(BaseModel):
    command: str
    intent: CommandIntent
    parameters: Dict[str, Any] = {}
    confidence: Optional[float] = None
    user_id: str = "default"
    timestamp: datetime

class Location(BaseModel):
    name: str
    latitude: float
    longitude: float
    altitude: Optional[float] = 0.0
    user_id: str
    created_at: datetime
    description: Optional[str] = None

class ObjectDetection(BaseModel):
    object_class: str
    confidence: float
    bounding_box: Dict[str, float]  # x, y, width, height
    center_point: Dict[str, float]  # x, y coordinates
    distance_estimate: Optional[float] = None
    direction: Optional[str] = None  # "left", "right", "center"

class SceneAnalysis(BaseModel):
    description: str
    objects: List[ObjectDetection]
    spatial_layout: Dict[str, Any]
    navigation_suggestions: List[str]
    confidence: float
    processing_time: float

class AudioResponse(BaseModel):
    text: str
    audio_url: str
    duration: Optional[float] = None
    voice_model: str = "en-US-Neural2-F"

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime
    processing_time: Optional[float] = None

class WebSocketMessage(BaseModel):
    type: MessageType
    user_id: Optional[str] = None
    data: Dict[str, Any]
    timestamp: datetime

# New models for spatial positioning and navigation features
class SpatialGuidanceRequest(BaseModel):
    user_id: str
    target_object: str  # e.g., "TV", "door", "chair"
    session_id: Optional[str] = None

class SaveLocationRequest(BaseModel):
    user_id: str
    location_name: str  # e.g., "home", "office", "gen house"
    latitude: float
    longitude: float
    description: Optional[str] = None

class NavigationRequest(BaseModel):
    user_id: str
    destination: str  # Name of saved location
    current_latitude: float
    current_longitude: float

class DepthAnalysisRequest(BaseModel):
    user_id: str
    return_colorized: bool = True
    session_id: Optional[str] = None

class UserSession(BaseModel):
    user_id: str
    session_id: str
    start_time: datetime
    last_activity: datetime
    current_location: Optional[GPSData] = None
    active_websockets: int = 0
    total_commands: int = 0
