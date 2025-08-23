#!/usr/bin/env python3
"""
Transcription module for Recipe Extractor
Handles speech-to-text conversion using Whisper models.
"""

import logging
from transformers import pipeline

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Handles audio transcription using Whisper models."""
    
    def __init__(self, model_name: str, device: str):
        """
        Initialize transcription service.
        
        Args:
            model_name: Whisper model from HuggingFace
            device: Device to use ('cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype="float16" if self.device == "cuda" else "float32"
            )
            logger.info("Whisper model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Use Whisper pipeline for transcription without experimental chunking parameters
            # Whisper handles long-form transcription internally using its own chunking mechanism
            result = self.pipeline(
                audio_path,
                generate_kwargs={
                    "language": "malay",  # Bahasa Melayu
                    "task": "transcribe"
                },
                return_timestamps=True  # Enable timestamps for better accuracy
            )
            
            # Extract text from result (handle both timestamp and non-timestamp formats)
            if isinstance(result, dict):
                if "text" in result:
                    transcript = result["text"].strip()
                elif "chunks" in result:
                    # If result has chunks with timestamps, concatenate the text
                    transcript = " ".join([chunk["text"] for chunk in result["chunks"]]).strip()
                else:
                    transcript = str(result).strip()
            else:
                transcript = str(result).strip()
            
            logger.info(f"Transcription completed. Length: {len(transcript)} characters")
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
