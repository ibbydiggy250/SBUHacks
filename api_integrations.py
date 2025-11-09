"""
API Integrations - ElevenLabs TTS API client
"""

import os
import requests
from typing import Optional, Dict


class ElevenLabsAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ElevenLabs TTS API client
        Args:
            api_key: ElevenLabs API key (or set ELEVENLABS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        self.base_url = 'https://api.elevenlabs.io/v1'
        
        if not self.api_key:
            print("Warning: ELEVENLABS_API_KEY not set. Text-to-speech will be disabled.")
    
    def text_to_speech(
        self, 
        text: str, 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default voice (Rachel)
        model_id: str = "eleven_monolingual_v1"
    ) -> Optional[bytes]:
        """
        Convert text to speech using ElevenLabs API
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model_id: Model ID to use
        Returns:
            Audio data as bytes (MP3 format) or None if error
        """
        if not self.api_key:
            return None
        
        if not text:
            return None
        
        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            headers = {
                'Accept': 'audio/mpeg',
                'Content-Type': 'application/json',
                'xi-api-key': self.api_key
            }
            
            data = {
                'text': text,
                'model_id': model_id,
                'voice_settings': {
                    'stability': 0.5,
                    'similarity_boost': 0.75
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling ElevenLabs API: {e}")
            return None
    
    def get_voices(self) -> Optional[Dict]:
        """Get list of available voices"""
        if not self.api_key:
            return None
        
        try:
            url = f"{self.base_url}/voices"
            headers = {
                'xi-api-key': self.api_key
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching voices: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting voices: {e}")
            return None

