"""
Core business logic for Project Noir Cloud API.
This file contains the implementation of the granular MCP tools.
"""

import asyncio
import base64
import httpx
import logging
from typing import Dict, List

# Internal imports
from utils import get_env_var
from frame_buffer import frame_buffer
from location_memory import LocationMemoryManager
import google.genai as genai
import googlemaps

logger = logging.getLogger(__name__)

# --- Service URLs and Model Config ---
YOLO_SERVICE_URL = get_env_var("YOLO_SERVICE_URL", default="http://yolo-detection:8000")
DEPTH_SERVICE_URL = get_env_var("DEPTH_SERVICE_URL", default="http://depth-estimation:8000")
GCP_PROJECT_ID = get_env_var("GCP_PROJECT_ID")
GCP_REGION = get_env_var("GCP_REGION")
GOOGLE_MAPS_API_KEY = get_env_var("GOOGLE_MAPS_API_KEY")
MODEL_NAME = "gemini-2.5-pro" # Upgraded Model

# --- Initialize Clients ---
genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
location_manager = LocationMemoryManager(project_id=GCP_PROJECT_ID)
maps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


# --- New Granular MCP Tool Implementations ---

async def get_frames(user_id: str) -> Dict:
    """Fetches the last 3 camera frames from the buffer."""
    try:
        frames_data = await frame_buffer.get_latest_frames(user_id, count=3)
        if not frames_data:
            return {"error": "No frames available in the buffer."}
        
        encoded_frames = [base64.b64encode(frame).decode('utf-8') for frame in frames_data]
        return {"frames": encoded_frames}
    except Exception as e:
        logger.error(f"Error fetching frames for {user_id}: {e}")
        return {"error": str(e)}

async def analyze_vision(user_id: str, frames: List[str], prompt: str) -> Dict:
    """Sends frames and a prompt to the Gemini vision model for analysis."""
    try:
        image_contents = [{"inline_data": {"mime_type": "image/jpeg", "data": frame}} for frame in frames]
        
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt] + image_contents
        )
        return {"description": response.text}
    except Exception as e:
        logger.error(f"Error in vision analysis for {user_id}: {e}")
        return {"error": str(e)}

async def find_objects(user_id: str, frames: List[str]) -> Dict:
    """Sends frames to the YOLO service for object detection."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [client.post(f"{YOLO_SERVICE_URL}/detect", files={"file": ("image.jpg", base64.b64decode(frame), "image/jpeg")}) for frame in frames]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_detections = []
        for resp in responses:
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                all_detections.extend(resp.json().get("detections", []))
        
        # Deduplicate and aggregate results
        unique_objects = {}
        for det in all_detections:
            name = det['class_name']
            if name not in unique_objects or unique_objects[name]['confidence'] < det['confidence']:
                unique_objects[name] = det
                
        return {"objects": list(unique_objects.values())}

async def get_depth(user_id: str, frames: List[str]) -> Dict:
    """Sends frames to the depth estimation service."""
    # Use the most recent frame for depth analysis
    if not frames:
        return {"error": "No frames provided for depth analysis."}
    
    latest_frame = frames[-1]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"file": ("image.jpg", base64.b64decode(latest_frame), "image/jpeg")}
        try:
            response = await client.post(f"{DEPTH_SERVICE_URL}/estimate-depth", files=files)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Depth estimation failed with status {response.status_code}"}
        except Exception as e:
            logger.error(f"Error calling depth service for {user_id}: {e}")
            return {"error": str(e)}

async def get_current_location(user_id: str) -> Dict:
    """Gets the current GPS location."""
    location = location_manager.get_current_location(user_id)
    if location:
        return {"location": location}
    return {"error": "Current location not available."}

async def save_location(user_id: str, location_name: str, latitude: float, longitude: float) -> Dict:
    """Saves a location with given coordinates."""
    success = location_manager.save_location(user_id, location_name, latitude, longitude)
    if success:
        return {"status": "success", "message": f"Location '{location_name}' saved."}
    return {"status": "error", "message": "Failed to save location."}

async def get_saved_location(user_id: str, location_name: str) -> Dict:
    """Retrieves the coordinates of a saved location."""
    location = location_manager.get_saved_location(user_id, location_name)
    if location:
        return {"location": location}
    return {"error": f"Location '{location_name}' not found."}

async def navigate_to_location(user_id: str, destination_name: str) -> Dict:
    """
    Provides turn-by-turn navigation instructions from the user's current location
    to a named destination.
    """
    try:
        current_location = await get_current_location(user_id)
        if "error" in current_location:
            return current_location

        origin_lat = current_location["data"]["latitude"]
        origin_lon = current_location["data"]["longitude"]
        origin = f"{origin_lat},{origin_lon}"

        logger.info(f"[CORE LOGIC] Requesting navigation from {origin} to {destination_name}")

        directions_result = maps_client.directions(origin,
                                                   destination_name,
                                                   mode="walking")

        if not directions_result:
            return {"error": "Could not find a route to the destination."}

        # Extracting and formatting the steps
        steps = []
        for i, step in enumerate(directions_result[0]['legs'][0]['steps']):
            steps.append(f"{i+1}. {step['html_instructions']}")

        return {
            "status": "success",
            "data": {
                "summary": directions_result[0]['summary'],
                "distance": directions_result[0]['legs'][0]['distance']['text'],
                "duration": directions_result[0]['legs'][0]['duration']['text'],
                "steps": steps,
                "full_response": directions_result
            }
        }
    except Exception as e:
        logger.error(f"Error getting navigation directions: {e}")
        return {"error": "An unexpected error occurred while fetching navigation directions."} 