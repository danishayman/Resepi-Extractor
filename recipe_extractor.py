#!/usr/bin/env python3
"""
TikTok/Instagram Recipe Extraction Pipeline
Extracts structured recipes from cooking videos using offline AI models.
"""

import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yt_dlp
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline
)
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeExtractor:
    def __init__(self, 
                 whisper_model: str = "openai/whisper-large-v3-turbo", 
                 llm_model: str = "microsoft/DialoGPT-medium",
                 device: str = "auto"):
        """
        Initialize the recipe extraction pipeline.
        
        Args:
            whisper_model: Whisper model name (e.g., 'openai/whisper-large-v3-turbo')
            llm_model: Hugging Face model for recipe extraction
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.whisper_model_name = whisper_model
        self.llm_model_name = llm_model
        self.device = self._get_device(device)
        
        # Initialize models
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipeline = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_pipeline = None
        
        self._load_models()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_models(self):
        """Load Whisper and LLM models."""
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            
            # Load Whisper model using Transformers
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.whisper_model_name, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            self.whisper_model.to(self.device)
            
            self.whisper_processor = AutoProcessor.from_pretrained(self.whisper_model_name)
            
            # Create Whisper pipeline
            self.whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=self.device,
            )
            
            logger.info(f"Loading LLM model: {self.llm_model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline for text generation
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
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
                'preferredquality': '192',
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
    
    def transcribe_audio(self, audio_path: str, save_transcript: bool = True, output_dir: str = "transcripts") -> str:
        """
        Transcribe audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            save_transcript: Whether to save transcript to a txt file
            output_dir: Directory to save transcript files
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Use the pipeline for transcription with long-form support
            result = self.whisper_pipeline(
                audio_path,
                return_timestamps=True,  # Required for long audio (>30 seconds)
                chunk_length_s=30,      # Process in 30-second chunks
                generate_kwargs={
                    "language": "ms",  # Bahasa Melayu
                    "task": "transcribe"
                }
            )
            
            transcript = result["text"].strip()
            logger.info(f"Transcription completed. Length: {len(transcript)} characters")
            
            # Save transcript to file if requested
            if save_transcript:
                # Include timestamp chunks if available
                transcript_data = {
                    "text": transcript,
                    "chunks": result.get("chunks", [])
                }
                transcript_path = self.save_transcript(transcript_data, audio_path, output_dir)
                logger.info(f"Transcript saved to: {transcript_path}")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def save_transcript(self, transcript_data, audio_path: str, output_dir: str = "transcripts") -> str:
        """
        Save transcript to a txt file.
        
        Args:
            transcript_data: The transcribed text data (dict with text and chunks)
            audio_path: Path to the original audio file (used for naming)
            output_dir: Directory to save transcript files
            
        Returns:
            Path to saved transcript file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename based on audio file name
            audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
            transcript_filename = f"{audio_filename}_transcript.txt"
            transcript_path = os.path.join(output_dir, transcript_filename)
            
            # Handle both string and dict input for backward compatibility
            if isinstance(transcript_data, str):
                transcript_text = transcript_data
                chunks = []
            else:
                transcript_text = transcript_data.get("text", "")
                chunks = transcript_data.get("chunks", [])
            
            # Save transcript with UTF-8 encoding to support Bahasa Melayu characters
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcript for: {os.path.basename(audio_path)}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write("FULL TRANSCRIPT:\n")
                f.write(transcript_text)
                
                # Add timestamp chunks if available
                if chunks:
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("TIMESTAMPED SEGMENTS:\n")
                    f.write("=" * 50 + "\n\n")
                    for chunk in chunks:
                        timestamp = chunk.get("timestamp", [0, 0])
                        text = chunk.get("text", "")
                        start_time = timestamp[0] if len(timestamp) > 0 else 0
                        end_time = timestamp[1] if len(timestamp) > 1 else start_time
                        f.write(f"[{start_time:.1f}s - {end_time:.1f}s]: {text}\n")
            
            logger.info(f"Transcript saved to: {transcript_path}")
            return transcript_path
            
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            raise
    
    def extract_recipe(self, transcript: str) -> Dict:
        """
        Extract structured recipe from transcript using local LLM.
        
        Args:
            transcript: Transcribed text from video
            
        Returns:
            Dictionary containing recipe data
        """
        prompt = self._create_recipe_extraction_prompt(transcript)
        
        try:
            logger.info("Extracting recipe using LLM...")
            
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=1000,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract JSON from the generated text
            recipe_json = self._extract_json_from_response(generated_text)
            
            logger.info("Recipe extraction completed successfully")
            return recipe_json
            
        except Exception as e:
            logger.error(f"Error extracting recipe: {e}")
            raise
    
    def _create_recipe_extraction_prompt(self, transcript: str) -> str:
        """Create a prompt for recipe extraction."""
        prompt = f"""You are an expert at extracting structured recipe information from Bahasa Melayu cooking video transcripts.

Extract the recipe information from this transcript and format it as a valid JSON object. Focus on identifying:
1. The dish name/title
2. All ingredients mentioned with their quantities
3. The cooking steps in chronological order

Transcript:
{transcript}

Return ONLY a valid JSON object with this exact structure:
{{
    "title": "Recipe name in Bahasa Melayu",
    "ingredients": [
        "ingredient 1 with quantity",
        "ingredient 2 with quantity"
    ],
    "steps": [
        "step 1 description",
        "step 2 description"
    ]
}}

JSON:"""
        return prompt
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from LLM response."""
        try:
            # Clean up the response
            response = response.strip()
            
            # Try to find JSON content after the prompt
            if "JSON:" in response:
                response = response.split("JSON:")[-1].strip()
            
            # Find JSON-like content in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                # Clean up common issues
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                
                recipe_data = json.loads(json_str)
                
                # Validate structure
                required_keys = ["title", "ingredients", "steps"]
                if all(key in recipe_data for key in required_keys):
                    # Ensure ingredients and steps are lists
                    if isinstance(recipe_data["ingredients"], list) and isinstance(recipe_data["steps"], list):
                        return recipe_data
            
            # Fallback: create structured response from text
            logger.warning("Could not extract valid JSON, creating fallback structure")
            return self._create_fallback_recipe(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, creating fallback structure")
            return self._create_fallback_recipe(response)
    
    def _create_fallback_recipe(self, response: str) -> Dict:
        """Create a fallback recipe structure when JSON extraction fails."""
        return {
            "title": "Extracted Recipe",
            "ingredients": ["Could not extract ingredients from transcript"],
            "steps": [f"Raw response: {response[:500]}..."]
        }
    
    def save_recipe(self, recipe_data: Dict, output_path: str = "recipe.json") -> str:
        """
        Save recipe data to JSON file.
        
        Args:
            recipe_data: Recipe dictionary
            output_path: Path to save JSON file
            
        Returns:
            Path to saved file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Recipe saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving recipe: {e}")
            raise
    
    def process_video(self, video_url: str, output_json_path: str = "recipe.json", save_transcript: bool = True, transcript_dir: str = "transcripts") -> Tuple[Dict, str]:
        """
        Complete pipeline to process a video and extract recipe.
        
        Args:
            video_url: TikTok/Instagram video URL
            output_json_path: Path to save the recipe JSON
            save_transcript: Whether to save transcript to txt file
            transcript_dir: Directory to save transcript files
            
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
            transcript = self.transcribe_audio(audio_path, save_transcript, transcript_dir)
            
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
    video_url = "https://www.tiktok.com/@example/video/1234567890"
    
    try:
        # Initialize the extractor
        extractor = RecipeExtractor(
            whisper_model="openai/whisper-large-v3-turbo",  # Fast and accurate
            llm_model="microsoft/Phi-3-mini-4k-instruct"  # Good for structured data extraction
        )
        
        # Process the video
        recipe_data, saved_path = extractor.process_video(video_url, "recipe.json")
        
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