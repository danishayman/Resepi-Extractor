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
from typing import Dict, List, Optional, Tuple
import yt_dlp
from transformers import pipeline
import google.generativeai as genai
import re

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables should be set manually
    pass

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
        # Load from environment variables with fallbacks
        self.whisper_model_name = whisper_model or os.getenv('WHISPER_MODEL', 'openai/whisper-large-v3-turbo')
        self.gemini_model_name = gemini_model or os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        self.device = self._get_device(device or os.getenv('DEVICE', 'auto'))
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', './recipes')
        self.audio_quality = audio_quality or os.getenv('AUDIO_QUALITY', '192')
        
        # Initialize API key
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not gemini_api_key:
                raise ValueError(
                    "Gemini API key is required. Either pass it as parameter or set GEMINI_API_KEY environment variable.\n"
                    "Get your API key from: https://makersuite.google.com/app/apikey"
                )
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
        
        # Initialize models
        self.whisper_pipeline = None
        
        self._load_models()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_models(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype="float16" if self.device == "cuda" else "float32"
            )
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def download_audio(self, video_url: str, output_dir: str = None) -> str:
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
    
    def _preprocess_audio(self, audio_path: str) -> str:
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

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Preprocess audio for better quality
            processed_audio_path = self._preprocess_audio(audio_path)
            
            # Use Whisper pipeline for transcription with proper configuration for long audio
            result = self.whisper_pipeline(
                processed_audio_path,
                generate_kwargs={
                    "language": "malay",  # Bahasa Melayu
                    "task": "transcribe"
                },
                return_timestamps=True,  # Enable timestamps for long-form generation
                chunk_length_s=30,      # Process audio in 30-second chunks
                stride_length_s=5       # 5-second overlap between chunks
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
    
    def extract_recipe(self, transcript: str) -> Dict:
        """
        Extract structured recipe from transcript using Gemini API.
        
        Args:
            transcript: Transcribed text from video
            
        Returns:
            Dictionary containing recipe data
        """
        prompt = self._create_recipe_extraction_prompt(transcript)
        
        try:
            logger.info("Extracting recipe using Gemini...")
            
            # Generate response using Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                )
            )
            
            generated_text = response.text
            
            # Extract JSON from the generated text
            recipe_json = self._extract_json_from_response(generated_text)
            
            logger.info("Recipe extraction completed successfully")
            return recipe_json
            
        except Exception as e:
            logger.error(f"Error extracting recipe: {e}")
            raise
    
    def _create_recipe_extraction_prompt(self, transcript: str) -> str:
        """Create a prompt for recipe extraction."""
        prompt = f"""You are a helpful assistant that extracts structured recipe information from cooking video transcripts in Bahasa Melayu.

Given the following transcript from a cooking video, extract and return ONLY a valid JSON object with the recipe information. Do not include any other text, explanations, or markdown formatting.

The JSON should have this exact structure:
{{
    "title": "Recipe name in Bahasa Melayu",
    "ingredients": {{
        "main_ingredients": [
            {{
                "name": "Ingredient name",
                "quantity": "Amount with unit"
            }}
        ],
        "spices_and_seasonings": [
            {{
                "name": "Spice or seasoning name", 
                "quantity": "Amount with unit"
            }}
        ]
    }},
    "instructions": [
        "Step 1 description in detail",
        "Step 2 description in detail"
    ]
}}

Important guidelines:
- Group ingredients logically into sections like "main_ingredients", "spices_and_seasonings", "garnish", etc.
- Each ingredient should have separate "name" and "quantity" fields
- If no clear grouping is possible, use "main_ingredients" as the default section
- Make instructions detailed and clear
- Keep everything in Bahasa Melayu
- Return ONLY the JSON object, nothing else

Transcript:
{transcript}

JSON:"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from Gemini response."""
        try:
            # Clean up the response - remove markdown formatting if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Find JSON-like content in the response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                recipe_data = json.loads(json_str)
                
                # Validate structure
                required_keys = ["title", "ingredients", "instructions"]
                if all(key in recipe_data for key in required_keys):
                    # Validate ingredients structure
                    if isinstance(recipe_data["ingredients"], dict) and recipe_data["ingredients"]:
                        return recipe_data
            
            # Try parsing the entire cleaned response as JSON
            try:
                recipe_data = json.loads(cleaned_response)
                if all(key in recipe_data for key in ["title", "ingredients", "instructions"]):
                    # Validate ingredients structure
                    if isinstance(recipe_data["ingredients"], dict) and recipe_data["ingredients"]:
                        return recipe_data
            except:
                pass
            
            # Fallback: create structured response from text
            logger.warning("Could not extract valid JSON, creating fallback structure")
            return self._create_fallback_recipe(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, creating fallback structure")
            return self._create_fallback_recipe(response)
    
    def _create_fallback_recipe(self, response: str) -> Dict:
        """Create a fallback recipe structure when JSON extraction fails."""
        # Try to extract basic information from the response text
        lines = response.split('\n')
        
        title = "Extracted Recipe"
        ingredients = []
        instructions = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to identify sections
            if any(word in line.lower() for word in ['title', 'nama', 'resepi']):
                if ':' in line:
                    title = line.split(':', 1)[1].strip()
            elif any(word in line.lower() for word in ['bahan', 'ingredients']):
                current_section = 'ingredients'
            elif any(word in line.lower() for word in ['langkah', 'cara', 'steps', 'instructions']):
                current_section = 'instructions'
            elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                item = line[1:].strip()
                if current_section == 'ingredients':
                    # Try to parse ingredient into name and quantity
                    parts = item.split(' ', 1)
                    if len(parts) >= 2:
                        ingredients.append({
                            "name": parts[0],
                            "quantity": parts[1]
                        })
                    else:
                        ingredients.append({
                            "name": item,
                            "quantity": "secukup rasa"
                        })
                elif current_section == 'instructions':
                    instructions.append(item)
        
        # If we couldn't extract anything useful, use the raw response
        if not ingredients and not instructions:
            ingredients = [{"name": "Could not extract ingredients from transcript", "quantity": ""}]
            instructions = [f"Raw response: {response[:300]}..."]
        
        # Create structured ingredients with fallback section
        structured_ingredients = {
            "main_ingredients": ingredients if ingredients else [{"name": "Could not extract ingredients", "quantity": ""}]
        }
        
        return {
            "title": title,
            "ingredients": structured_ingredients,
            "instructions": instructions if instructions else ["Could not extract instructions"]
        }
    
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
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(self.output_dir, "recipe.json")
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
            audio_path = self.download_audio(video_url, temp_dir)
            
            # Step 2: Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            
            # Step 3: Extract recipe
            recipe_data = self.extract_recipe(transcript)
            
            # Step 4: Save recipe
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