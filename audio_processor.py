"""
Audio Processor for Project Noir Cloud
Bridge module for audio processing functionality
"""

from audio_processor_utils import AudioProcessor

# Alias for cloud-specific audio processing
CloudAudioProcessor = AudioProcessor

__all__ = ["CloudAudioProcessor", "AudioProcessor"]
