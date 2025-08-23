#!/usr/bin/env python3
"""
Audio processing module for Recipe Extractor
Handles audio download and preprocessing from video URLs.
"""

import os
import tempfile
import logging
from typing import Optional, Tuple
import yt_dlp

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio download and preprocessing operations."""
    
    def __init__(self, audio_quality: str = '192'):
        """
        Initialize audio processor.
        
        Args:
            audio_quality: Audio quality for downloads (default: '192')
        """
        self.audio_quality = audio_quality
    
    def download_audio(self, video_url: str, output_dir: Optional[str] = None) -> str:
        """
        Download audio from TikTok/Instagram video using yt-dlp.
        
        Args:
            video_url: URL of the video
            output_dir: Directory to save audio file
            
        Returns:
            Path to downloaded audio file
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_path = os.path.join(output_dir, "audio.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': self.audio_quality,
            }],
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            logger.info(f"Downloading audio from: {video_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded audio file
            audio_file = os.path.join(output_dir, "audio.wav")
            if os.path.exists(audio_file):
                logger.info(f"Audio downloaded successfully: {audio_file}")
                return audio_file
            else:
                raise FileNotFoundError("Downloaded audio file not found")
                
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            raise
    
    def get_video_info(self, video_url: str) -> dict:
        """
        Extract video metadata including thumbnail URL using yt-dlp.
        
        Args:
            video_url: URL of the video
            
        Returns:
            Dictionary containing video metadata including thumbnail URL
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            logger.info(f"Extracting video metadata from: {video_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Extract relevant metadata
                metadata = {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'thumbnail': info.get('thumbnail'),
                    'duration': info.get('duration'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                }
                
                logger.info(f"Video metadata extracted successfully. Thumbnail URL: {metadata.get('thumbnail')}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return {}
    
    def download_audio_with_metadata(self, video_url: str, output_dir: Optional[str] = None) -> Tuple[str, dict]:
        """
        Download audio and extract video metadata in one operation.
        
        Args:
            video_url: URL of the video
            output_dir: Directory to save audio file
            
        Returns:
            Tuple of (audio_file_path, video_metadata)
        """
        try:
            # First get metadata
            metadata = self.get_video_info(video_url)
            
            # Then download audio
            audio_path = self.download_audio(video_url, output_dir)
            
            return audio_path, metadata
            
        except Exception as e:
            logger.error(f"Error downloading audio with metadata: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for better transcription quality.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            # For now, return the original path
            # In the future, this could include:
            # - Noise reduction
            # - Volume normalization
            # - Format conversion
            # - Audio trimming/splitting for very long files
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Get file size to estimate duration (rough estimate)
            file_size = os.path.getsize(audio_path)
            logger.info(f"Audio file size: {file_size / (1024*1024):.2f} MB")
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
