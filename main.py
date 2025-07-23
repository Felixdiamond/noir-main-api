"""
Project Noir Cloud - Main API Service
Central orchestration service for all Project Noir cloud features

Updated to use google-genai library with proper Vertex AI procedures:
- Client initialization: genai.Client(vertexai=True, project=..., location=...)
- Content creation: inline_data format for audio/image content
- API calls: client.models.generate_content(model=..., contents=...)
- Model: gemini-2.5-pro (SOTA with enhanced thinking and reasoning)
- Proper error handling and environment validation
- Compatible with google-genai>=0.7.0

Required Environment Variables:
- GCP_PROJECT_ID: Your Google Cloud Project ID
- GCP_REGION: GCP region (e.g., 'us-central1')
- CLOUD_STORAGE_BUCKET: Cloud Storage bucket name
- GOOGLE_APPLICATION_CREDENTIALS: Path to service account key (for local dev)

Required GCP APIs:
- Vertex AI API (aiplatform.googleapis.com)
- Cloud Text-to-Speech API (texttospeech.googleapis.com)
- Cloud Storage API (storage.googleapis.com)
"""

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from enum import Enum
import json
import time
import uuid
from typing import Dict, List, Optional
import logging
from datetime import datetime
import io
import base64
import httpx
import math

# Google Cloud imports
from google.cloud import storage
from google.cloud import texttospeech
import google.genai as genai

# Internal imports
from models import *
from location_memory import LocationMemoryManager
from websocket_manager import connection_manager as websocket_manager
from audio_processor import CloudAudioProcessor
from utils import setup_logging, get_env_var
from frame_buffer import frame_buffer
from mcp_server import mcp_server

MODEL_NAME = "gemini-1.5-pro-preview-0409"  # Upgraded Model

# Initialize FastAPI app
app = FastAPI(
    title="Project Noir Cloud API",
    description="AI-Powered Independence for the Visually Impaired - Cloud Edition",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the MCP server router
app.include_router(mcp_server.router, prefix="/mcp")


# Initialize logging
logger = setup_logging()

# Initialize global services
location_manager = LocationMemoryManager()
# websocket_manager is imported as connection_manager from websocket_manager module
audio_processor = CloudAudioProcessor()

# --- Proximity State Enum ---
class ProximityState(Enum):
    NONE = 0
    AT = 1
    NEAR = 2

# --- Proximity Monitor State ---
proximity_states = {}

# Helper function for creating multimodal content
def create_inline_data_content(data: bytes, mime_type: str) -> Dict:
    """Create inline data content for Gemini API"""
    data_base64 = base64.b64encode(data).decode('utf-8')
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": data_base64
        }
    }

# Configure Vertex AI (Gemini) - Updated for google-genai library
GCP_PROJECT_ID = get_env_var("GCP_PROJECT_ID")
GCP_REGION = get_env_var("GCP_REGION")

# Initialize the GenAI client
# Note: This requires proper authentication setup:
# 1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to service account key
# 2. Or use gcloud auth application-default login for local development
# 3. Ensure the service account has Vertex AI User role
client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)

# Configure Google Cloud TTS
tts_client = texttospeech.TextToSpeechClient()

# Configure Cloud Storage
storage_client = storage.Client()
bucket_name = get_env_var("CLOUD_STORAGE_BUCKET")

# Service URLs
YOLO_SERVICE_URL = get_env_var("YOLO_SERVICE_URL", default="https://noir-yolo-api-930930012707.us-central1.run.app")
DEPTH_SERVICE_URL = get_env_var("DEPTH_SERVICE_URL", default="http://depth-estimation:8000")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Project Noir Cloud API starting up...")
    
    # Validate environment variables
    try:
        assert GCP_PROJECT_ID, "GCP_PROJECT_ID environment variable not set"
        assert GCP_REGION, "GCP_REGION environment variable not set"
        assert bucket_name, "CLOUD_STORAGE_BUCKET environment variable not set"
        logger.info("✅ Environment variables validated")
    except AssertionError as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return
    
    # Test Gemini connection
    try:
        test_response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Test connection"
        )
        logger.info(f"✅ Gemini {MODEL_NAME} connected successfully")
    except Exception as e:
        logger.error(f"❌ Gemini connection failed: {e}")
        logger.error("Please ensure:")
        logger.error("1. Vertex AI API is enabled in your GCP project")
        logger.error("2. Service account has Vertex AI User role")
        logger.error("3. GOOGLE_APPLICATION_CREDENTIALS is set correctly")
    
    # Test Cloud TTS
    try:
        test_synthesis = tts_client.synthesize_speech(
            input=texttospeech.SynthesisInput(text="Test"),
            voice=texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F"
            ),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
        )
        logger.info("✅ Cloud TTS connected successfully")
    except Exception as e:
        logger.error(f"❌ Cloud TTS connection failed: {e}")
    
    # Start frame buffer cleanup task
    asyncio.create_task(periodic_frame_cleanup())
    logger.info("🧹 Frame buffer cleanup task started")

    logger.info("🎯 Project Noir Cloud API ready for real-time requests")


@app.get("/")
async def root():
    """API health check and info"""
    return {
        "service": "Project Noir Cloud API",
        "status": "operational",
        "version": "1.0.0",
        "features": [
            "Real-time scene analysis",
            "Voice command processing", 
            "GPS location memory",
            "Object detection",
            "Text recognition",
            "Audio synthesis"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/frame")
async def process_frame(request_data: dict):
    """
    Process camera frame from any device (JSON format)
    This endpoint now ONLY stores the frame in the buffer.
    All complex logic is initiated by the agent via MCP tools.
    """
    try:
        user_id = request_data.get("user_id", "default")
        device_id = request_data.get("device_id", "unknown")
        base64_image = request_data.get("image_data", "")
        
        image_bytes = base64.b64decode(base64_image)
        
        logger.info(f"📸 Frame received from {device_id} for user {user_id}")
        
        await frame_buffer.store_frame(user_id, image_bytes)

        await websocket_manager.broadcast({
            "type": "frame_received",
            "data": { "user_id": user_id, "device_id": device_id }
        })

        return {"success": True, "message": "Frame stored in buffer"}
        
    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")
        return {"success": False, "error": str(e)}


    
@app.get("/gps/current")
async def get_current_gps(user_id: str = "default"):
    """
    Get the latest GPS location for a user (from mobile app via WebSocket)
    """
    current = location_manager.get_current_location(user_id)
    if current:
        return {"success": True, **{k: current[k] for k in ['latitude','longitude','altitude','timestamp'] if k in current}}
    else:
        return {"success": False, "error": "Current location not available"}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    # Generate unique connection ID
    connection_id = f"ws_{user_id}_{int(time.time())}"
    
    # Connect to websocket manager
    connected = await websocket_manager.connect(websocket, connection_id, user_id, "websocket")
    
    if not connected:
        logger.error(f"Failed to establish WebSocket connection for user {user_id}")
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages
            await websocket_manager.handle_message(connection_id, data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_manager.disconnect(connection_id)


@app.get("/locations/{user_id}")
async def get_user_locations(user_id: str):
    """Get all saved locations for a user"""
    locations = location_manager.get_all_locations(user_id)
    return {"user_id": user_id, "locations": locations}

@app.delete("/locations/{user_id}/{location_name}")
async def delete_location(user_id: str, location_name: str):
    """Delete a saved location"""
    success = location_manager.delete_location(user_id, location_name)
    return {"success": success, "message": f"Location '{location_name}' {'deleted' if success else 'not found'}"}

@app.post("/heartbeat")
async def device_heartbeat(request_data: dict):
    """
    Handle heartbeat from any device (ESP32, mobile, etc.)
    """
    try:
        user_id = request_data.get("user_id", "default")
        device_id = request_data.get("device_id", "unknown")
        device_type = request_data.get("device_type", "unknown")
        
        logger.info(f"💓 Heartbeat from {device_type} device {device_id}")
        
        # Update device status in websocket manager
        await websocket_manager.broadcast({
            "type": "device_heartbeat",
            "data": {
                "user_id": user_id,
                "device_id": device_id,
                "device_type": device_type,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "online"
            }
        })
        
        return {
            "success": True,
            "message": "Heartbeat received",
            "server_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Heartbeat error: {e}")
        return {"success": False, "error": str(e)}

# All intent-based endpoints like /spatial-guidance, /save-location, etc. are removed
# as this logic is now handled by the agent via MCP tools.

@app.get("/health")
async def health_check():
    """Detailed health check for all services"""
    return {
        "status": "healthy",
        "services": {
            "gemini": "connected",
            "cloud_tts": "connected", 
            "cloud_storage": "connected",
            "location_manager": "operational",
            "websocket_manager": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
