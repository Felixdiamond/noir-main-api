from fastmcp import FastMCP
from typing import Dict, List
import logging
from logic import (
    get_frames,
    analyze_vision,
    find_objects,
    get_depth,
    get_current_location,
    save_location,
    get_saved_location,
    navigate_to_location
)

logger = logging.getLogger(__name__)

mcp_server = FastMCP(
    name="ProjectNoir",
    instructions="A suite of tools to help visually impaired users perceive and navigate their environment."
)

@mcp_server.tool()
async def get_camera_frames(user_id: str) -> Dict:
    """
    Fetches the three most recent camera frames from the user's device.
    
    This is the primary way to get visual input. The output is a list of
    base64-encoded image strings, which can be passed to other vision-based tools.
    
    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        A dictionary containing a list of base64 encoded image strings.
        Example: {'status': 'success', 'data': {'frames': ['/9j/4AAQSkZJRg...', ...]}}
    """
    logger.info(f"[MCP] Tool called: get_camera_frames for user {user_id}")
    return await get_frames(user_id)

@mcp_server.tool()
async def analyze_scene_with_vision_model(user_id: str, frames: List[str], prompt: str) -> Dict:
    """
    Analyzes a list of image frames using the Gemini 2.5 Pro vision model.
    
    This is a powerful and general-purpose vision tool. Use it for:
    - General scene understanding and description.
    - Reading text from the images.
    - Answering any visual question that doesn't involve specific object detection or depth analysis.

    Args:
        user_id (str): The unique identifier for the user.
        frames (List[str]): A list of base64-encoded image strings, typically from `get_camera_frames`.
        prompt (str): The specific question or instruction for the vision model.

    Returns:
        A dictionary containing the model's textual analysis of the scene.
    """
    logger.info(f"[MCP] Tool called: analyze_scene_with_vision_model for user {user_id}")
    return await analyze_vision(user_id, frames, prompt)

@mcp_server.tool()
async def find_objects_in_scene(user_id: str, frames: List[str]) -> Dict:
    """
    Performs object detection on a list of image frames using a YOLO model.
    
    Use this tool when you need to identify and locate all objects in the user's view.
    It returns a list of detected objects, their positions (bounding boxes), and confidence scores.
    This is more specialized than `analyze_scene_with_vision_model`.

    Args:
        user_id (str): The unique identifier for the user.
        frames (List[str]): A list of base64-encoded image strings, typically from `get_camera_frames`.

    Returns:
        A dictionary containing a list of detected objects and their coordinates.
    """
    logger.info(f"[MCP] Tool called: find_objects_in_scene for user {user_id}")
    return await find_objects(user_id, frames)

@mcp_server.tool()
async def analyze_scene_depth(user_id: str, frames: List[str]) -> Dict:
    """
    Analyzes the depth of a scene from a list of image frames.
    
    Use this tool to understand the relative distances of objects in the scene.
    It returns a depth map and a general analysis of the scene's depth.
    
    Args:
        user_id (str): The unique identifier for the user.
        frames (List[str]): A list of base64-encoded image strings, typically from `get_camera_frames`.

    Returns:
        A dictionary containing the depth analysis.
    """
    logger.info(f"[MCP] Tool called: analyze_scene_depth for user {user_id}")
    return await get_depth(user_id, frames)

@mcp_server.tool()
async def get_current_user_location(user_id: str) -> Dict:
    """
    Retrieves the user's current GPS coordinates (latitude and longitude).
    
    This tool is essential for any location-based tasks.

    Args:
        user_id (str): The unique identifier for the user.

    Returns:
        A dictionary containing the user's latitude and longitude.
    """
    logger.info(f"[MCP] Tool called: get_current_user_location for user {user_id}")
    return await get_current_location(user_id)

@mcp_server.tool()
async def save_current_location(user_id: str, location_name: str) -> Dict:
    """
    Saves the user's current location with a descriptive name.
    
    This tool first gets the user's current GPS coordinates and then saves them
    to a persistent memory associated with the provided name.

    Args:
        user_id (str): The unique identifier for the user.
        location_name (str): The name to associate with the current location (e.g., "Home", "Office").

    Returns:
        A dictionary confirming the save operation was successful.
    """
    logger.info(f"[MCP] Tool called: save_current_location for user {user_id} with name {location_name}")
    return await save_location(user_id, location_name)

@mcp_server.tool()
async def get_coordinates_of_saved_location(user_id: str, location_name: str) -> Dict:
    """
    Retrieves the GPS coordinates for a previously saved location.
    
    Args:
        user_id (str): The unique identifier for the user.
        location_name (str): The name of the location to retrieve.

    Returns:
        A dictionary containing the saved location's latitude and longitude.
    """
    logger.info(f"[MCP] Tool called: get_coordinates_of_saved_location for user {user_id} with name {location_name}")
    return await get_saved_location(user_id, location_name)

@mcp_server.tool()
async def navigate_to_saved_location(user_id: str, destination_name: str) -> Dict:
    """
    Provides turn-by-turn walking directions from the user's current location to a named destination.
    
    This tool uses the Google Maps Directions API to generate a route. The destination can be
    a named location (e.g., "Eiffel Tower") or a previously saved location name.

    Args:
        user_id (str): The unique identifier for the user.
        destination_name (str): The name of the destination to navigate to.

    Returns:
        A dictionary containing a summary of the route and a list of step-by-step instructions.
    """
    logger.info(f"[MCP] Tool called: navigate_to_saved_location for user {user_id} to {destination_name}")
    return await navigate_to_location(user_id, destination_name) 