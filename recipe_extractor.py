#!/usr/bin/env python3
"""
TikTok/Instagram Recipe Extraction Pipeline
Extracts structured recipes from cooking videos using Whisper and Gemini API.
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

from config import Config
from audio_processor import AudioProcessor
from transcription import TranscriptionService
from recipe_parser import RecipeParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeExtractor:
    def __init__(self, 
                    whisper_model: str = None, 
                    gemini_api_key: str = None,
                    gemini_model: str = None,
                    device: str = None,
                    output_dir: str = None,
                    audio_quality: str = None):
        """
        Initialize the recipe extraction pipeline.
        
        Args:
            whisper_model: Whisper model from HuggingFace (default: from WHISPER_MODEL env var or openai/whisper-large-v3-turbo)
            gemini_api_key: Google Gemini API key (default: from GEMINI_API_KEY env var)
            gemini_model: Gemini model name (default: from GEMINI_MODEL env var or gemini-1.5-flash)
            device: Device to use ('cpu', 'cuda', 'auto') (default: from DEVICE env var or auto)
            output_dir: Output directory for recipes (default: from OUTPUT_DIR env var or ./recipes)
            audio_quality: Audio quality for downloads (default: from AUDIO_QUALITY env var or 192)
        """
        # Initialize configuration
        self.config = Config(
            whisper_model=whisper_model,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            device=device,
            output_dir=output_dir,
            audio_quality=audio_quality
        )
        
        # Initialize service components
        self.audio_processor = AudioProcessor(self.config.audio_quality)
        self.transcription_service = TranscriptionService(self.config.whisper_model, self.config.device)
        self.recipe_parser = RecipeParser(self.config.gemini_api_key, self.config.gemini_model)

    
    def save_recipe(self, recipe_data: Dict, output_path: str = None) -> str:
        """
        Save recipe data to JSON file.
        
        Args:
            recipe_data: Recipe dictionary
            output_path: Path to save JSON file (default: uses output_dir/recipe.json)
            
        Returns:
            Path to saved file
        """
        try:
            # Use default output path if not provided
            if output_path is None:
                # Ensure output directory exists
                Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(self.config.output_dir, "recipe.json")
            else:
                # Ensure the directory for the output path exists
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Recipe saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving recipe: {e}")
            raise
    
    def process_video(self, video_url: str, output_json_path: str = None) -> Tuple[Dict, str]:
        """
        Complete pipeline to process a video and extract recipe.
        
        Args:
            video_url: TikTok/Instagram video URL
            output_json_path: Path to save the recipe JSON (default: uses output_dir/recipe.json)
            
        Returns:
            Tuple of (recipe_data, saved_file_path)
        """
        temp_dir = None
        audio_path = None
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Step 1: Download audio
            audio_path = self.audio_processor.download_audio(video_url, temp_dir)
            
            # Step 2: Preprocess audio
            processed_audio_path = self.audio_processor.preprocess_audio(audio_path)
            
            # Step 3: Transcribe audio
            transcript = self.transcription_service.transcribe(processed_audio_path)
            
            # Step 4: Extract recipe
            recipe_data = self.recipe_parser.extract_recipe(transcript)
            
            # Step 5: Save recipe
            saved_path = self.save_recipe(recipe_data, output_json_path)
            
            return recipe_data, saved_path
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        
        finally:
            # Cleanup temporary files
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            if temp_dir and os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass


def main():
    """Example usage of the recipe extraction pipeline."""
    
    # Example video URL (replace with actual TikTok/Instagram URL)
    video_url = "https://www.tiktok.com/@khairulaming/video/7092671985238478106?is_from_webapp=1&sender_device=pc&web_id=7534748416518538769"
    
    try:
        # Initialize the extractor (all parameters will be loaded from environment variables)
        extractor = RecipeExtractor()
        
        # Process the video (will use default output path from environment)
        recipe_data, saved_path = extractor.process_video(video_url)
        
        # Print results
        print("\n" + "="*50)
        print("RECIPE EXTRACTION COMPLETED!")
        print("="*50)
        print(f"Recipe saved to: {os.path.abspath(saved_path)}")
        print("\nExtracted Recipe:")
        print(json.dumps(recipe_data, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())