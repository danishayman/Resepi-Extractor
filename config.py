#!/usr/bin/env python3
"""
Configuration management for Recipe Extractor
"""

import os
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables should be set manually
    pass


class Config:
    """Configuration class for Recipe Extractor pipeline."""
    
    def __init__(self,
                 whisper_model: Optional[str] = None,
                 gemini_api_key: Optional[str] = None,
                 gemini_model: Optional[str] = None,
                 device: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 audio_quality: Optional[str] = None,
                 supabase_url: Optional[str] = None,
                 supabase_key: Optional[str] = None,
                 skip_classification: Optional[bool] = None,
                 classification_confidence: Optional[float] = None):
        """
        Initialize configuration with optional overrides.
        
        Args:
            whisper_model: Whisper model from HuggingFace
            gemini_api_key: Google Gemini API key
            gemini_model: Gemini model name
            device: Device to use ('cpu', 'cuda', 'auto')
            output_dir: Output directory for recipes
            audio_quality: Audio quality for downloads
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service role key
            skip_classification: Whether to skip cooking video classification
            classification_confidence: Minimum confidence threshold for classification
        """
        # Load from environment variables with fallbacks
        self.whisper_model = whisper_model or os.getenv('WHISPER_MODEL', 'openai/whisper-large-v3-turbo')
        self.gemini_model = gemini_model or os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.device = self._get_device(device or os.getenv('DEVICE', 'auto'))
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', './recipes')
        self.audio_quality = audio_quality or os.getenv('AUDIO_QUALITY', '192')
        
        # Initialize API key
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                raise ValueError(
                    "Gemini API key is required. Either pass it as parameter or set GEMINI_API_KEY environment variable.\n"
                )
        self.gemini_api_key = gemini_api_key
        
        # Initialize Supabase settings
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        # Initialize classification settings
        self.skip_classification = skip_classification
        if self.skip_classification is None:
            self.skip_classification = os.getenv('SKIP_CLASSIFICATION', 'false').lower() == 'true'
        
        self.classification_confidence = classification_confidence
        if self.classification_confidence is None:
            try:
                self.classification_confidence = float(os.getenv('CLASSIFICATION_CONFIDENCE', '0.7'))
            except ValueError:
                self.classification_confidence = 0.7
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
