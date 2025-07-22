"""
Audio Processing Utilities for Project Noir Cloud
Handles speech-to-text, text-to-speech, and audio stream management
"""

import io
import json
import logging
import asyncio
from typing import Optional, Dict, List
import google.cloud.texttospeech as tts
import google.cloud.speech as speech
from google.oauth2 import service_account
import wave
import numpy as np

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        """Initialize Google Cloud TTS and Speech clients"""
        try:
            # Initialize Text-to-Speech client
            self.tts_client = tts.TextToSpeechClient()
            logger.info("✅ Google Cloud TTS client initialized")
            
            # Initialize Speech-to-Text client
            self.speech_client = speech.SpeechClient()
            logger.info("✅ Google Cloud Speech client initialized")
            
            # Configure TTS voice
            self.voice = tts.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",  # Female neural voice
                ssml_gender=tts.SsmlVoiceGender.FEMALE
            )
            
            # Configure audio format
            self.audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                speaking_rate=1.1,  # Slightly faster for efficiency
                pitch=0.0,
                volume_gain_db=0.0
            )
            
            # Configure speech recognition
            self.speech_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model="latest_long"  # Better for longer speech
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio clients: {e}")
            self.tts_client = None
            self.speech_client = None
    
    async def text_to_speech(self, text: str, voice_name: str = None) -> Optional[bytes]:
        """Convert text to speech audio"""
        try:
            if not self.tts_client:
                logger.error("TTS client not available")
                return None
            
            # Use custom voice if provided
            voice_config = self.voice
            if voice_name:
                voice_config = tts.VoiceSelectionParams(
                    language_code="en-US",
                    name=voice_name,
                    ssml_gender=tts.SsmlVoiceGender.FEMALE
                )
            
            # Create synthesis input
            synthesis_input = tts.SynthesisInput(text=text)
            
            # Generate speech
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_config,
                audio_config=self.audio_config
            )
            
            logger.info(f"🔊 Generated TTS audio for text: '{text[:50]}...'")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return None
    
    async def speech_to_text(self, audio_data: bytes) -> Optional[Dict]:
        """Convert speech audio to text"""
        try:
            if not self.speech_client:
                logger.error("Speech client not available")
                return None
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform speech recognition
            response = self.speech_client.recognize(
                config=self.speech_config,
                audio=audio
            )
            
            if not response.results:
                logger.warning("No speech recognized")
                return None
            
            # Get the best result
            result = response.results[0]
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence
            
            # Extract word timings
            word_info = []
            if hasattr(result.alternatives[0], 'words'):
                for word in result.alternatives[0].words:
                    word_info.append({
                        "word": word.word,
                        "start_time": word.start_time.total_seconds(),
                        "end_time": word.end_time.total_seconds()
                    })
            
            logger.info(f"🎤 Recognized speech: '{transcript}' (confidence: {confidence:.2f})")
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "words": word_info
            }
            
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return None
    
    async def streaming_speech_to_text(self, audio_stream):
        """Handle streaming speech recognition"""
        try:
            if not self.speech_client:
                logger.error("Speech client not available")
                return
            
            # Configure streaming recognition
            streaming_config = speech.StreamingRecognitionConfig(
                config=self.speech_config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create streaming recognition requests
            def request_generator():
                yield speech.StreamingRecognizeRequest(
                    streaming_config=streaming_config
                )
                
                for chunk in audio_stream:
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            
            # Start streaming recognition
            responses = self.speech_client.streaming_recognize(request_generator())
            
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    yield {
                        "transcript": transcript,
                        "is_final": is_final,
                        "confidence": result.alternatives[0].confidence if is_final else 0.0
                    }
                    
        except Exception as e:
            logger.error(f"Error in streaming speech recognition: {e}")
    
    def create_ssml_text(self, text: str, emphasis: str = None, 
                        pause_before: float = 0, pause_after: float = 0) -> str:
        """Create SSML formatted text for better TTS control"""
        ssml = f'<speak>'
        
        if pause_before > 0:
            ssml += f'<break time="{pause_before}s"/>'
        
        if emphasis:
            ssml += f'<emphasis level="{emphasis}">{text}</emphasis>'
        else:
            ssml += text
        
        if pause_after > 0:
            ssml += f'<break time="{pause_after}s"/>'
        
        ssml += '</speak>'
        return ssml
    
    async def generate_navigation_audio(self, guidance_text: str) -> Optional[bytes]:
        """Generate specially formatted audio for navigation guidance"""
        try:
            # Add SSML for better navigation audio
            ssml_text = self.create_ssml_text(
                guidance_text,
                emphasis="moderate",
                pause_before=0.2,
                pause_after=0.3
            )
            
            # Use SSML input instead of plain text
            synthesis_input = tts.SynthesisInput(ssml=ssml_text)
            
            # Generate with navigation-optimized settings
            nav_audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                speaking_rate=1.0,  # Normal speed for navigation
                pitch=2.0,  # Slightly higher pitch for clarity
                volume_gain_db=2.0  # Slightly louder
            )
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=nav_audio_config
            )
            
            logger.info(f"🧭 Generated navigation audio: '{guidance_text}'")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error generating navigation audio: {e}")
            return None
    
    async def generate_alert_audio(self, alert_text: str, urgency: str = "medium") -> Optional[bytes]:
        """Generate alert audio with appropriate urgency"""
        try:
            # Configure based on urgency
            emphasis_level = "moderate"
            speaking_rate = 1.0
            pitch = 0.0
            volume_gain = 0.0
            
            if urgency == "high":
                emphasis_level = "strong"
                speaking_rate = 1.2
                pitch = 4.0
                volume_gain = 4.0
            elif urgency == "low":
                emphasis_level = "reduced"
                speaking_rate = 0.9
                pitch = -2.0
                volume_gain = -2.0
            
            # Create SSML with urgency formatting
            ssml_text = self.create_ssml_text(
                alert_text,
                emphasis=emphasis_level,
                pause_before=0.1,
                pause_after=0.2
            )
            
            synthesis_input = tts.SynthesisInput(ssml=ssml_text)
            
            alert_audio_config = tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain
            )
            
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=alert_audio_config
            )
            
            logger.info(f"🚨 Generated {urgency} urgency alert: '{alert_text}'")
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error generating alert audio: {e}")
            return None
    
    def process_voice_command(self, transcript: str) -> Dict:
        """Process voice command and extract intent"""
        transcript_lower = transcript.lower().strip()
        
        # Navigation commands
        if any(word in transcript_lower for word in ["take me to", "navigate to", "go to", "direction to"]):
            # Extract location name
            for phrase in ["take me to", "navigate to", "go to", "direction to"]:
                if phrase in transcript_lower:
                    location = transcript_lower.split(phrase)[1].strip()
                    return {
                        "intent": "navigate",
                        "location": location,
                        "confidence": 0.9
                    }
        
        # Save location commands
        if any(word in transcript_lower for word in ["save this location", "remember this place", "save location"]):
            # Extract location name if provided
            location_name = "current_location"
            if "as" in transcript_lower:
                location_name = transcript_lower.split("as")[1].strip()
            elif "called" in transcript_lower:
                location_name = transcript_lower.split("called")[1].strip()
            
            return {
                "intent": "save_location",
                "location_name": location_name,
                "confidence": 0.8
            }
        
        # Object detection commands
        if any(word in transcript_lower for word in ["what do you see", "describe the scene", "what's in front", "scan area"]):
            return {
                "intent": "describe_scene",
                "confidence": 0.9
            }
        
        # Find object commands
        if any(word in transcript_lower for word in ["find", "where is", "locate"]):
            # Extract object name
            object_name = transcript_lower
            for phrase in ["find the", "find a", "where is the", "where is a", "locate the", "locate a"]:
                if phrase in transcript_lower:
                    object_name = transcript_lower.split(phrase)[1].strip()
                    break
            
            return {
                "intent": "find_object",
                "object": object_name,
                "confidence": 0.8
            }
        
        # Help commands
        if any(word in transcript_lower for word in ["help", "what can you do", "commands", "instructions"]):
            return {
                "intent": "help",
                "confidence": 0.9
            }
        
        # Default to general query
        return {
            "intent": "general_query",
            "query": transcript,
            "confidence": 0.5
        }
