#!/usr/bin/env python3
"""
Recipe parsing module for Recipe Extractor
Handles recipe extraction from transcripts using Gemini API.
"""

import json
import re
import logging
from typing import Dict
import google.generativeai as genai

logger = logging.getLogger(__name__)


class RecipeParser:
    """Handles recipe extraction from transcripts using Gemini API."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize recipe parser.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
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
            response = self.model.generate_content(
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
