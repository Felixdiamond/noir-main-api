"""
Utility functions for Project Noir Cloud API
"""

import os
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration for the application"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Create and return application logger
    logger = logging.getLogger("project_noir_cloud")
    logger.info(f"📝 Logging initialized at {level} level")
    
    return logger


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = True) -> str:
    """Get environment variable with optional default and validation"""
    
    value = os.getenv(var_name, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable '{var_name}' is not set")
    
    if value is None:
        return ""
    
    return value


def validate_gcp_config() -> dict:
    """Validate Google Cloud Platform configuration"""
    
    config = {}
    
    try:
        config['project_id'] = get_env_var('GCP_PROJECT_ID')
        config['region'] = get_env_var('GCP_REGION', 'us-central1', required=False)
        config['bucket_name'] = get_env_var('CLOUD_STORAGE_BUCKET')
        
        # Optional: Service account key path
        config['service_account'] = get_env_var('GOOGLE_APPLICATION_CREDENTIALS', required=False)
        
        return config
        
    except ValueError as e:
        raise ValueError(f"GCP configuration error: {e}")


def format_response(status: str, message: str, data: dict = None) -> dict:
    """Format standardized API response"""
    
    response = {
        "status": status,
        "message": message,
        "timestamp": "2025-07-21T00:00:00Z"  # Will be replaced with actual timestamp in production
    }
    
    if data:
        response["data"] = data
    
    return response


def sanitize_user_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input for safety"""
    
    if not text:
        return ""
    
    # Basic sanitization
    sanitized = text.strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove potentially dangerous characters (basic filtering)
    dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', '\n', '\r', '\t']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized
