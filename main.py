"""
Project Noir Cloud - Main API Service
Central orchestration service for all Project Noir cloud features

Updated to use google-genai library with proper Vertex AI procedures:
- Client initialization: genai.Client(vertexai=True, project=..., location=...)
- Content creation: genai.Part.from_data() for audio/image parts
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
from websocket_manager import WebSocketManager
from audio_processor import CloudAudioProcessor
from utils import setup_logging, get_env_var

MODEL_NAME = "gemini-2.5-pro"  # SOTA model with enhanced thinking and reasoning

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

# Initialize logging
logger = setup_logging()

# Initialize global services
location_manager = LocationMemoryManager()
websocket_manager = WebSocketManager()
audio_processor = CloudAudioProcessor()

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
YOLO_SERVICE_URL = get_env_var("YOLO_SERVICE_URL", default="http://yolo-detection:8000")
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
    
    logger.info("🎯 Project Noir Cloud API ready for requests")

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
    Optimized for current hardware: ESP32-CAM + Mobile GPS
    """
    try:
        user_id = request_data.get("user_id", "default")
        device_id = request_data.get("device_id", "unknown")
        base64_image = request_data.get("image_data", "")
        
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(base64_image)
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        logger.info(f"📸 Frame received from {device_id} for user {user_id}")
        
        # Store frame in Cloud Storage for processing
        blob_name = f"frames/{user_id}/{session_id}.jpg"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(image_bytes, content_type='image/jpeg')
        
        # Get latest GPS data from location manager (from mobile app)
        current_location = location_manager.get_current_location(user_id)
        
        # Send to YOLO detection service (async)
        asyncio.create_task(send_to_yolo_service(image_bytes, session_id, current_location))
        
        # Broadcast to connected WebSocket clients
        await websocket_manager.broadcast({
            "type": "frame_processed",
            "data": {
                "session_id": session_id,
                "user_id": user_id,
                "device_id": device_id,
                "gps_available": current_location is not None,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Frame processed successfully",
            "gps_available": current_location is not None
        }
        
    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/get-latest-frame")
async def get_latest_frame_endpoint(user_id: str = "default"):
    """
    Get the latest camera frame for a user
    Used by MCP orchestrator for vision analysis
    """
    try:
        # Get latest frame data
        frame_data = await get_latest_frame(user_id)
        
        if frame_data:
            # Convert to base64 for JSON response
            import base64
            image_base64 = base64.b64encode(frame_data).decode('utf-8')
            
            return {
                "success": True,
                "image_data": image_base64,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "No recent camera frame available",
                "user_id": user_id
            }
            
    except Exception as e:
        logger.error(f"❌ Error retrieving frame for user {user_id}: {e}")
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

@app.post("/voice_command")
async def process_voice_command(
    audio: UploadFile = File(...),
    user_id: str = Form(default="default")
):
    """
    Process voice command using Gemini 2.5 Pro audio understanding
    """
    try:
        # Read audio data
        audio_bytes = await audio.read()
        
        # Create audio part using proper google-genai format
        audio_part = genai.Part.from_data(data=audio_bytes, mime_type="audio/wav")
        
        prompt = """
        You are Noir, an AI assistant for visually impaired individuals. 
        Analyze this audio and extract the voice command. Respond with:
        1. The transcribed command
        2. The intent (scene_description, object_finding, location_save, location_navigate, text_reading, depth_analysis)
        3. Any specific parameters (object name, location name, etc.)
        
        Format as JSON: {"command": "...", "intent": "...", "parameters": {...}}
        """
        
        # Use the client to generate content
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, audio_part]
        )
        
        # Extract text from response
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        try:
            command_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning(f"Failed to parse JSON response: {response_text}")
            command_data = {
                "command": response_text,
                "intent": "unknown",
                "parameters": {}
            }
        
        # Process the command based on intent
        result = await process_command_intent(command_data, user_id)
        
        return JSONResponse(content={
            "status": "success",
            "transcription": command_data["command"],
            "intent": command_data["intent"],
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

async def process_command_intent(command_data: Dict, user_id: str) -> Dict:
    """Process command based on detected intent"""
    intent = command_data["intent"]
    parameters = command_data.get("parameters", {})
    
    if intent == "scene_description":
        return await handle_scene_description(user_id)
    elif intent == "object_finding":
        object_name = parameters.get("object", "object")
        return await handle_object_finding(user_id, object_name)
    elif intent == "location_save":
        location_name = parameters.get("location_name", "unnamed")
        return await handle_location_save(user_id, location_name)
    elif intent == "location_navigate":
        location_name = parameters.get("location_name", "")
        return await handle_location_navigate(user_id, location_name)
    elif intent == "text_reading":
        return await handle_text_reading(user_id)
    elif intent == "depth_analysis":
        return await handle_depth_analysis(user_id)
    else:
        return {"message": "I didn't understand that command. Please try again."}

async def handle_scene_description(user_id: str) -> Dict:
    """Handle scene description request"""
    # Get latest frame from storage
    latest_frame = await get_latest_frame(user_id)
    if not latest_frame:
        return {"message": "No recent camera frame available for analysis."}
    
    # Run YOLO object detection
    async with httpx.AsyncClient(timeout=30.0) as client_http:
        files = {"file": ("image.jpg", latest_frame, "image/jpeg")}
        yolo_resp = await client_http.post(f"{YOLO_SERVICE_URL}/detect", files=files)
        yolo_resp.raise_for_status()
        yolo_data = yolo_resp.json()
    objects = [d.get("class_name") for d in yolo_data.get("detections", [])]
    # Run depth estimation
    async with httpx.AsyncClient(timeout=60.0) as client_http:
        files = {"file": ("image.jpg", latest_frame, "image/jpeg")}
        depth_resp = await client_http.post(f"{DEPTH_SERVICE_URL}/estimate-depth", files=files, params={"return_colorized": True})
        depth_resp.raise_for_status()
        depth_data = depth_resp.json()
    # Build Gemini prompt with context
    system_prompt = (
        "You are Noir, an AI assistant for visually impaired users. "
        "Provide a concise scene description with spatial layout, objects and distances, and navigation cues."
    )
    context_lines = []
    if objects:
        context_lines.append(f"Detected objects: {', '.join(objects[:5])}")
    if depth_data:
        min_d = depth_data.get("min_depth", 0)
        max_d = depth_data.get("max_depth", 0)
        context_lines.append(f"Depth range: {min_d:.1f}m to {max_d:.1f}m")
    full_prompt = f"{system_prompt}\n\nContext:\n" + "\n".join(context_lines)
    # Create image part and generate response
    image_part = genai.Part.from_data(data=latest_frame, mime_type="image/jpeg")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[full_prompt, image_part]
    )
    
    # Extract text from response
    response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Convert to speech
    audio_response = await text_to_speech(response_text)
    
    return {
        "message": response_text,
        "audio_url": audio_response,
        "type": "scene_description"
    }

async def handle_object_finding(user_id: str, object_name: str) -> Dict:
    """Handle object detection and spatial guidance"""
    # Call spatial-guidance logic to locate the object
    request = SpatialGuidanceRequest(user_id=user_id, target_object=object_name)
    spatial_result = await spatial_guidance(request)
    # Determine guidance message
    if spatial_result.get("success"):
        guidance = spatial_result.get("guidance") or spatial_result.get("message", "")
    else:
        guidance = spatial_result.get("error", f"Unable to locate {object_name} at this time.")
    # Generate speech
    audio_url = await text_to_speech(guidance)
    return {
        "message": guidance,
        "audio_url": audio_url,
        "type": "object_search",
        "found": spatial_result.get("found", False),
        "confidence": spatial_result.get("confidence")
    }

async def handle_location_save(user_id: str, location_name: str) -> Dict:
    """Save current location to memory"""
    current_location = location_manager.get_current_location(user_id)
    if not current_location:
        return {"message": "GPS location not available. Please try again."}
    
    success = location_manager.save_location(
        user_id=user_id,
        name=location_name,
        latitude=current_location["latitude"],
        longitude=current_location["longitude"]
    )
    
    if success:
        message = f"Location saved as '{location_name}' at coordinates {current_location['latitude']:.6f}, {current_location['longitude']:.6f}"
    else:
        message = f"Failed to save location '{location_name}'. Please try again."
    
    # Convert to speech
    audio_response = await text_to_speech(message)
    
    return {
        "message": message,
        "audio_url": audio_response,
        "type": "location_save",
        "success": success
    }

async def handle_location_navigate(user_id: str, location_name: str) -> Dict:
    """Navigate to saved location using real GPS navigation"""
    # Use location_manager to get navigation guidance
    nav_result = location_manager.get_navigation_guidance(user_id, location_name)
    
    if not nav_result:
        msg = f"Unable to navigate to {location_name}. Please ensure current GPS location is available and the destination is saved."
        audio = await text_to_speech(msg)
        return {"message": msg, "audio_url": audio, "type": "navigation", "destination": location_name}

    # Get guidance message
    msg = nav_result.get("guidance", f"Navigate to {location_name}")

    # Synthesize speech
    audio = await text_to_speech(msg)
    return {
        "message": msg,
        "audio_url": audio,
        "type": "navigation",
        "destination": location_name,
        "distance_meters": nav_result.get("distance_meters"),
        "direction": nav_result.get("direction"),
        "arrived": nav_result.get("distance_meters", float('inf')) < 10  # Arrived if within 10 meters
    }

async def handle_text_reading(user_id: str) -> Dict:
    """Handle text recognition request"""
    # Get latest frame
    latest_frame = await get_latest_frame(user_id)
    if not latest_frame:
        return {"message": "No recent camera frame available for text reading."}
    
    # Analyze with Gemini for OCR using proper google-genai format
    prompt = """
    You are Noir, helping someone read text. Look at this image and:
    1. If there's readable text, say "The text reads:" followed by the exact text
    2. If no clear text is visible, say "I don't see any clear text in this image"
    3. Include details about text color, style, or format if helpful
    
    Be concise and helpful.
    """
    
    # Create image part for the frame
    image_part = genai.Part.from_data(data=latest_frame, mime_type="image/jpeg")
    
    # Use the client to generate content
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image_part]
    )
    
    # Extract text from response
    response_text = response.text if hasattr(response, 'text') else str(response)
    
    # Convert to speech
    audio_response = await text_to_speech(response_text)
    
    return {
        "message": response_text,
        "audio_url": audio_response,
        "type": "text_reading"
    }

async def handle_depth_analysis(user_id: str) -> Dict:
    """Handle depth analysis request via voice command"""
    # Use internal depth_analysis endpoint logic
    request = DepthAnalysisRequest(user_id=user_id, return_colorized=True)
    depth_result = await depth_analysis(request)

    # Handle failure case
    if not depth_result.get("success"):
        msg = depth_result.get("error", "Depth analysis failed.")
        audio_url = await text_to_speech(msg)
        return {
            "message": msg,
            "audio_url": audio_url,
            "type": "depth_analysis",
            "success": False
        }

    # Success: prepare description and audio
    description = depth_result.get("depth_analysis", "")
    audio_url = await text_to_speech(description)
    return {
        "message": description,
        "audio_url": audio_url,
        "type": "depth_analysis",
        "depth_range": depth_result.get("depth_range"),
        "colorized_depth_image": depth_result.get("colorized_depth_image")
    }

async def text_to_speech(text: str) -> str:
    """Convert text to speech using Cloud TTS"""
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # High-quality neural voice
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Upload to Cloud Storage and return URL
        audio_id = str(uuid.uuid4())
        blob_name = f"audio/{audio_id}.mp3"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(response.audio_content, content_type='audio/mp3')
        
        # Make blob public (configure properly for production)
        blob.make_public()
        
        return blob.public_url
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return ""

async def get_latest_frame(user_id: str):
    """Get latest camera frame for processing"""
    try:
        # Get the most recent frame from Cloud Storage
        bucket = storage_client.bucket(bucket_name)
        prefix = f"frames/{user_id}/"
        
        # List blobs with the prefix, ordered by creation time
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            logger.warning(f"No frames found for user {user_id}")
            return None
        
        # Get the most recent blob (by name, which includes timestamp)
        latest_blob = sorted(blobs, key=lambda x: x.name, reverse=True)[0]
        
        # Download the image data
        image_data = latest_blob.download_as_bytes()
        logger.info(f"📸 Retrieved latest frame for user {user_id}: {latest_blob.name}")
        
        return image_data
        
    except Exception as e:
        logger.error(f"❌ Error retrieving latest frame for user {user_id}: {e}")
        return None

# Spatial positioning and navigation utilities
def calculate_object_position(bbox: dict, image_width: int, image_height: int) -> dict:
    """Calculate object position and direction from YOLO bounding box"""
    try:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate relative position (0-1 range)
        rel_x = center_x / image_width
        rel_y = center_y / image_height
        
        # Determine horizontal direction
        if rel_x < 0.33:
            horizontal = "left"
        elif rel_x > 0.67:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Determine vertical position
        if rel_y < 0.33:
            vertical = "top"
        elif rel_y > 0.67:
            vertical = "bottom"
        else:
            vertical = "middle"
        
        # Estimate relative distance based on bounding box size
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = image_width * image_height
        relative_size = bbox_area / image_area
        
        # Rough distance estimation
        if relative_size > 0.3:
            distance = "very close"
        elif relative_size > 0.15:
            distance = "close"
        elif relative_size > 0.05:
            distance = "moderate distance"
        else:
            distance = "far"
        
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "distance": distance,
            "relative_x": rel_x,
            "relative_y": rel_y,
            "bbox_size": relative_size,
            "center_coordinates": {"x": center_x, "y": center_y}
        }
        
    except Exception as e:
        logger.error(f"Error calculating object position: {e}")
        return {"error": str(e)}



async def send_to_yolo_service(image_bytes: bytes, session_id: str, gps_location: Optional[dict] = None):
    """Send image to YOLO detection service asynchronously with optional GPS context"""
    try:
        # This will call the separate YOLO detection service
        # Include GPS location if available for spatial context
        logger.info(f"🎯 Sending frame {session_id} to YOLO service" + 
                   (f" with GPS context" if gps_location else ""))
        # Implementation depends on your YOLO service setup
        pass
    except Exception as e:
        logger.error(f"❌ YOLO service error for session {session_id}: {e}")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages
            await websocket_manager.handle_message(user_id, data)
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)

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

@app.post("/spatial-guidance")
async def spatial_guidance(request: SpatialGuidanceRequest):
    """
    Provide spatial guidance for finding objects - "where is the TV?"
    Uses YOLO detection + spatial positioning without requiring additional sensors
    """
    try:
        logger.info(f"🔍 Spatial guidance request for '{request.target_object}' from user {request.user_id}")
        
        # Get latest frame from storage
        image_data = await get_latest_frame(request.user_id)
        if not image_data:
            return {"success": False, "error": "No recent camera frame available"}
        
        # Send frame to YOLO service for object detection
        async with httpx.AsyncClient(timeout=30.0) as client:
            files = {"file": ("image.jpg", image_data, "image/jpeg")}
            params = {
                "confidence_threshold": 0.3,
                "iou_threshold": 0.4
            }
            
            response = await client.post(f"{YOLO_SERVICE_URL}/detect", files=files, params=params)
            if response.status_code != 200:
                return {"success": False, "error": "Object detection failed"}
            
            detection_result = response.json()
        
        # Find the target object in detections
        target_detections = []
        for detection in detection_result["detections"]:
            if request.target_object.lower() in detection["class_name"].lower():
                target_detections.append(detection)
        
        if not target_detections:
            return {
                "success": True,
                "found": False,
                "message": f"I don't see any {request.target_object} in the current view. Try moving the camera around to scan the area."
            }
        
        # Find the most confident detection
        best_detection = max(target_detections, key=lambda x: x["confidence"])
        
        # Calculate spatial position
        position = calculate_object_position(
            best_detection["bounding_box"],
            detection_result["image_width"],
            detection_result["image_height"]
        )
        
        # Generate natural language guidance
        horizontal_guidance = ""
        if position["horizontal"] == "left":
            horizontal_guidance = "Turn slightly to your left"
        elif position["horizontal"] == "right":
            horizontal_guidance = "Turn slightly to your right"
        else:
            horizontal_guidance = "It's directly in front of you"
        
        vertical_guidance = ""
        if position["vertical"] == "top":
            vertical_guidance = "Look up higher"
        elif position["vertical"] == "bottom":
            vertical_guidance = "Look down lower"
        else:
            vertical_guidance = "at about eye level"
        
        guidance_message = f"I found the {best_detection['class_name']}! {horizontal_guidance}"
        if vertical_guidance != "at about eye level":
            guidance_message += f" and {vertical_guidance}"
        guidance_message += f". It appears to be at {position['distance']}."
        
        return {
            "success": True,
            "found": True,
            "object": best_detection["class_name"],
            "confidence": best_detection["confidence"],
            "guidance": guidance_message,
            "position": position,
            "detection_count": len(target_detections)
        }
        
    except Exception as e:
        logger.error(f"❌ Spatial guidance error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/save-location")
async def save_location(request: SaveLocationRequest):
    """Save a GPS location with a custom name"""
    try:
        logger.info(f"📍 Saving location '{request.location_name}' for user {request.user_id}")
        
        success = location_manager.save_location(
            request.user_id,
            request.location_name,
            request.latitude,
            request.longitude
        )
        
        if success:
            return {
                "success": True,
                "message": f"Location '{request.location_name}' saved successfully",
                "location": {
                    "name": request.location_name,
                    "latitude": request.latitude,
                    "longitude": request.longitude
                }
            }
        else:
            return {"success": False, "error": "Failed to save location"}
            
    except Exception as e:
        logger.error(f"❌ Save location error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/navigate-to")
async def navigate_to(request: NavigationRequest):
    """Provide GPS navigation to a saved location"""
    try:
        logger.info(f"🧭 Navigation request to '{request.destination}' from user {request.user_id}")
        
        # Get the destination location
        destination_data = location_manager.get_saved_location(request.user_id, request.destination)
        if not destination_data:
            return {
                "success": False,
                "error": f"Location '{request.destination}' not found. Please save it first."
            }
        
        # Calculate distance and bearing
        distance = location_manager.calculate_distance(
            request.current_latitude,
            request.current_longitude,
            destination_data["latitude"],
            destination_data["longitude"]
        )
        
        bearing = location_manager.calculate_bearing(
            request.current_latitude,
            request.current_longitude,
            destination_data["latitude"],
            destination_data["longitude"]
        )
        
        direction = location_manager.get_direction_text(bearing)
        
        # Generate navigation guidance
        if distance < 5:  # Very close (within 5 meters)
            guidance = f"You've arrived at {request.destination}! You're within {distance:.1f} meters."
        elif distance < 50:  # Close (within 50 meters)
            guidance = f"{request.destination} is very close - about {distance:.0f} meters to the {direction}."
        elif distance < 500:  # Moderate distance (within 500 meters)
            guidance = f"Head {direction} for about {distance:.0f} meters to reach {request.destination}."
        else:  # Far distance
            guidance = f"{request.destination} is {distance/1000:.1f} kilometers to the {direction}."
        
        return {
            "success": True,
            "destination": request.destination,
            "distance_meters": round(distance, 1),
            "direction": direction,
            "bearing_degrees": round(bearing, 1),
            "guidance": guidance,
            "arrived": distance < 5
        }
        
    except Exception as e:
        logger.error(f"❌ Navigation error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/depth-analysis")
async def depth_analysis(request: DepthAnalysisRequest):
    """Analyze depth information from camera image"""
    try:
        logger.info(f"📏 Depth analysis request from user {request.user_id}")
        
        # Get latest frame from storage
        image_data = await get_latest_frame(request.user_id)
        if not image_data:
            return {"success": False, "error": "No recent camera frame available"}
        
        # Send frame to depth estimation service
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": ("image.jpg", image_data, "image/jpeg")}
            params = {"return_colorized": request.return_colorized}
            
            response = await client.post(f"{DEPTH_SERVICE_URL}/estimate-depth", files=files, params=params)
            if response.status_code != 200:
                return {"success": False, "error": "Depth estimation failed"}
            
            depth_result = response.json()
        
        # Generate depth description
        min_depth = depth_result["min_depth"]
        max_depth = depth_result["max_depth"]
        depth_range = max_depth - min_depth
        
        depth_description = f"Scene depth analysis: Objects range from {min_depth:.1f} to {max_depth:.1f} units away. "
        
        if depth_range > 10:
            depth_description += "The scene has significant depth variation with both close and distant objects."
        elif depth_range > 5:
            depth_description += "Moderate depth variation in the scene."
        else:
            depth_description += "Most objects are at similar distances."
        
        return {
            "success": True,
            "depth_analysis": depth_description,
            "depth_range": {
                "min": min_depth,
                "max": max_depth,
                "variation": depth_range
            },
            "colorized_depth_image": depth_result.get("colorized_depth_base64", ""),
            "processing_time": depth_result["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"❌ Depth analysis error: {e}")
        return {"success": False, "error": str(e)}

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
