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
        prompt = f"""You are a helpful assistant that extracts structured recipe information from cooking video transcripts in casual, everyday Bahasa Melayu.

Given the following transcript from a cooking video, extract and return ONLY a valid JSON object with the recipe information. Do not include any other text, explanations, or markdown formatting.

The JSON should have this exact structure:
{{
    "title": "Recipe name (can mix English and Bahasa Melayu - use whatever sounds natural, e.g. 'Chocolate Chip Cookies', 'Nasi Lemak Special', 'Korean BBQ Chicken')",
    "description": "Brief description in casual Bahasa Melayu with some English words (2-3 sentences, like how people normally talk)",
    "cuisine_type": "Type of cuisine (boleh guna English atau Malay, contoh: 'Western', 'Masakan Melayu', 'Korean', 'Dessert', 'Chinese', 'Fusion', etc)",
    "tags": ["tag1", "tag2", "tag3"],
    "ingredients": {{
        "main_ingredients": [
            {{
                "name": "Ingredient name (guna nama biasa, boleh English/Malay)",
                "quantity": "Amount with unit (contoh: '2 cawan', '500g', '1 tbsp')"
            }}
        ],
        "spices_and_seasonings": [
            {{
                "name": "Spice or seasoning name (nama biasa yang orang guna)", 
                "quantity": "Amount with unit"
            }}
        ]
    }},
    "instructions": {{
        "step1": "First specific action (guna bahasa biasa, tak perlu formal)",
        "step2": "Second specific action",
        "step3": "Third specific action",
        "step4": "Fourth specific action"
    }}
}}

Important guidelines:
- Guna bahasa biasa macam orang cakap sehari-hari - tak payah terlalu formal
- For titles: Boleh guna English words especially for popular dish names (contoh: "Creamy Carbonara", "Mango Sticky Rice", "Chicken Rendang")
- Group ingredients logically into sections like "main_ingredients", "spices_and_seasonings", "garnish", etc.
- Each ingredient should have separate "name" and "quantity" fields
- If no clear grouping is possible, use "main_ingredients" as the default section
- BREAK DOWN instructions into individual, specific steps - each step should contain ONLY ONE action
- Each step should be clear and concise (1-2 sentences maximum)
- Use as many steps as needed (step1, step2, step3, step4, step5, etc.) to properly break down the cooking process
- Contoh good individual steps: "Panaskan oven 180°C", "Mix cream cheese dengan gula sampai smooth", "Masukkan telur satu-satu", "Tuang mixture dalam pan"
- For description: Write a brief, appetizing description - guna bahasa santai dengan sikit English words kalau perlu
- For cuisine_type: Identify the type of cuisine - boleh guna English atau Malay, whatever sounds more natural
- For tags: Include 3-5 relevant tags - mix English/Malay yang sounds natural (contoh: "easy", "sedap", "dessert", "spicy", "comfort food")
- Guna mix English dan Bahasa Melayu yang sounds natural - macam orang Malaysia cakap biasa
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
                        # Validate instructions structure (should be dict, not list)
                        if isinstance(recipe_data["instructions"], dict) and recipe_data["instructions"]:
                            return recipe_data
            
            # Try parsing the entire cleaned response as JSON
            try:
                recipe_data = json.loads(cleaned_response)
                if all(key in recipe_data for key in ["title", "ingredients", "instructions"]):
                    # Validate ingredients structure
                    if isinstance(recipe_data["ingredients"], dict) and recipe_data["ingredients"]:
                        # Validate instructions structure (should be dict, not list)
                        if isinstance(recipe_data["instructions"], dict) and recipe_data["instructions"]:
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
            "description": "Recipe yang extract dari cooking video ni",
            "cuisine_type": "Mixed",
            "tags": ["recipe", "sedap", "homemade"],
            "ingredients": structured_ingredients,
            "instructions": {
                f"step{i+1}": instruction for i, instruction in enumerate(instructions)
            } if instructions else {"step1": "Tak dapat extract instructions properly"}
        }
