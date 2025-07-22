"""
Location Memory Manager for Project Noir Cloud
Handles GPS location saving, retrieval, and navigation
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from google.cloud import firestore
from models import Location, GPSData

logger = logging.getLogger(__name__)

class LocationMemoryManager:
    def __init__(self):
        """Initialize Firestore client for location storage"""
        try:
            self.db = firestore.Client()
            self.locations_collection = "user_locations"
            self.current_locations_collection = "current_locations"
            logger.info("✅ LocationMemoryManager initialized with Firestore")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firestore: {e}")
            self.db = None
    
    def save_location(self, user_id: str, name: str, latitude: float, longitude: float, 
                     altitude: float = 0.0, description: str = None) -> bool:
        """Save a named location for a user"""
        try:
            if not self.db:
                logger.error("Firestore not available")
                return False
            
            location_data = {
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "description": description
            }
            
            # Use user_id/location_name as document ID for easy retrieval
            doc_id = f"{user_id}_{name.lower().replace(' ', '_')}"
            
            self.db.collection(self.locations_collection).document(doc_id).set(location_data)
            
            logger.info(f"📍 Saved location '{name}' for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving location: {e}")
            return False
    
    def get_saved_location(self, user_id: str, name: str) -> Optional[Dict]:
        """Retrieve a saved location by name"""
        try:
            if not self.db:
                return None
            
            doc_id = f"{user_id}_{name.lower().replace(' ', '_')}"
            doc = self.db.collection(self.locations_collection).document(doc_id).get()
            
            if doc.exists:
                location_data = doc.to_dict()
                logger.info(f"📍 Retrieved location '{name}' for user {user_id}")
                return location_data
            else:
                logger.warning(f"Location '{name}' not found for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving location: {e}")
            return None
    
    def get_all_locations(self, user_id: str) -> List[Dict]:
        """Get all saved locations for a user"""
        try:
            if not self.db:
                return []
            
            query = self.db.collection(self.locations_collection).where("user_id", "==", user_id)
            docs = query.stream()
            
            locations = []
            for doc in docs:
                location_data = doc.to_dict()
                locations.append(location_data)
            
            logger.info(f"📍 Retrieved {len(locations)} locations for user {user_id}")
            return locations
            
        except Exception as e:
            logger.error(f"Error retrieving all locations: {e}")
            return []
    
    def delete_location(self, user_id: str, name: str) -> bool:
        """Delete a saved location"""
        try:
            if not self.db:
                return False
            
            doc_id = f"{user_id}_{name.lower().replace(' ', '_')}"
            self.db.collection(self.locations_collection).document(doc_id).delete()
            
            logger.info(f"🗑️ Deleted location '{name}' for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting location: {e}")
            return False
    
    def update_current_location(self, user_id: str, latitude: float, longitude: float, 
                              altitude: float = 0.0, timestamp: datetime = None) -> bool:
        """Update user's current GPS location"""
        try:
            if not self.db:
                return False
            
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            current_location_data = {
                "user_id": user_id,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "timestamp": timestamp,
                "updated_at": datetime.utcnow()
            }
            
            self.db.collection(self.current_locations_collection).document(user_id).set(current_location_data)
            
            # Don't log every update to avoid spam
            return True
            
        except Exception as e:
            logger.error(f"Error updating current location: {e}")
            return False
    
    def get_current_location(self, user_id: str) -> Optional[Dict]:
        """Get user's current location"""
        try:
            if not self.db:
                return None
            
            doc = self.db.collection(self.current_locations_collection).document(user_id).get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving current location: {e}")
            return None
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters using Haversine formula"""
        import math
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in meters
        r = 6371000
        
        return c * r
    
    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing (direction) from point 1 to point 2 in degrees"""
        import math
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Convert to 0-360 degrees
        
        return bearing
    
    def get_direction_text(self, bearing: float) -> str:
        """Convert bearing to human-readable direction"""
        if bearing < 22.5 or bearing >= 337.5:
            return "north"
        elif 22.5 <= bearing < 67.5:
            return "northeast"
        elif 67.5 <= bearing < 112.5:
            return "east"
        elif 112.5 <= bearing < 157.5:
            return "southeast"
        elif 157.5 <= bearing < 202.5:
            return "south"
        elif 202.5 <= bearing < 247.5:
            return "southwest"
        elif 247.5 <= bearing < 292.5:
            return "west"
        elif 292.5 <= bearing < 337.5:
            return "northwest"
    
    def get_navigation_guidance(self, user_id: str, destination_name: str) -> Optional[Dict]:
        """Get navigation guidance from current location to destination"""
        try:
            current = self.get_current_location(user_id)
            destination = self.get_saved_location(user_id, destination_name)
            
            if not current or not destination:
                return None
            
            distance = self.calculate_distance(
                current["latitude"], current["longitude"],
                destination["latitude"], destination["longitude"]
            )
            
            bearing = self.calculate_bearing(
                current["latitude"], current["longitude"],
                destination["latitude"], destination["longitude"]
            )
            
            direction = self.get_direction_text(bearing)
            
            return {
                "destination": destination_name,
                "distance_meters": round(distance, 1),
                "distance_text": self._format_distance(distance),
                "bearing": round(bearing, 1),
                "direction": direction,
                "guidance": f"Head {direction} for {self._format_distance(distance)} to reach {destination_name}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating navigation guidance: {e}")
            return None
    
    def _format_distance(self, distance_meters: float) -> str:
        """Format distance in human-readable format"""
        if distance_meters < 1:
            return f"{int(distance_meters * 100)} centimeters"
        elif distance_meters < 1000:
            return f"{int(distance_meters)} meters"
        else:
            km = distance_meters / 1000
            return f"{km:.1f} kilometers"
    
    def check_arrival(self, user_id: str, destination_name: str, threshold_meters: float = 10.0) -> bool:
        """Check if user has arrived at destination (within threshold)"""
        try:
            current = self.get_current_location(user_id)
            destination = self.get_saved_location(user_id, destination_name)
            
            if not current or not destination:
                return False
            
            distance = self.calculate_distance(
                current["latitude"], current["longitude"],
                destination["latitude"], destination["longitude"]
            )
            
            return distance <= threshold_meters
            
        except Exception as e:
            logger.error(f"Error checking arrival: {e}")
            return False
