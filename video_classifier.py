#!/usr/bin/env python3
"""
Video Classifier for Recipe Extractor
Determines if a video is cooking-related before processing for recipe extraction.
"""

import logging
from typing import Dict, Tuple, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class VideoClassifier:
    """Classifies videos to determine if they are cooking-related."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize video classifier.
        
        Args:
            api_key: Google Gemini API key
            model_name: Gemini model name
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def classify_video(self, transcript: str, video_metadata: Dict = None) -> Tuple[bool, float, str]:
        """
        Classify if a video is cooking-related based on transcript and metadata.
        
        Args:
            transcript: Transcribed text from video
            video_metadata: Optional video metadata (title, description, etc.)
            
        Returns:
            Tuple of (is_cooking_video, confidence_score, reason)
        """
        prompt = self._create_classification_prompt(transcript, video_metadata)
        
        try:
            logger.info("Classifying video content...")
            
            # Generate response using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent classification
                    max_output_tokens=200,
                )
            )
            
            generated_text = response.text
            
            # Parse the classification result
            is_cooking, confidence, reason = self._parse_classification_response(generated_text)
            
            logger.info(f"Classification result: {'Cooking' if is_cooking else 'Non-cooking'} "
                       f"(confidence: {confidence:.2f}) - {reason}")
            
            return is_cooking, confidence, reason
            
        except Exception as e:
            logger.error(f"Error classifying video: {e}")
            # Default to False (not cooking) on error to be safe
            return False, 0.0, f"Classification error: {str(e)}"
    
    def _create_classification_prompt(self, transcript: str, video_metadata: Dict = None) -> str:
        """Create a prompt for video classification."""
        
        metadata_info = ""
        if video_metadata:
            title = video_metadata.get('title', '')
            description = video_metadata.get('description', '')
            if title:
                metadata_info += f"Video Title: {title}\n"
            if description:
                metadata_info += f"Video Description: {description}\n"
        
        prompt = f"""You are a video content classifier. Your task is to determine if a video is about COOKING based on the transcript and metadata provided.

COOKING VIDEOS include:
- Recipe tutorials and cooking demonstrations
- Food preparation and cooking techniques
- Baking and dessert making
- Beverage preparation (smoothies, cocktails, coffee, etc.)
- Kitchen tips and cooking hacks
- Restaurant/chef cooking content
- Food review WITH cooking/preparation shown
- Traditional/cultural cooking methods

NON-COOKING VIDEOS include:
- Pure food reviews without cooking/preparation
- Restaurant visits without cooking shown
- Food challenges or eating competitions
- General lifestyle, beauty, fashion content
- Travel vlogs (unless focused on cooking local food)
- Music, dance, comedy, or entertainment content
- Product unboxing or shopping content
- Educational content not related to cooking

{metadata_info}
Transcript:
{transcript}

Based on the content above, classify this video and respond in EXACTLY this format:
CLASSIFICATION: [COOKING/NOT_COOKING]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation in one sentence]

Examples of good responses:
CLASSIFICATION: COOKING
CONFIDENCE: 0.95
REASON: Video demonstrates step-by-step recipe preparation with ingredients and cooking instructions.

CLASSIFICATION: NOT_COOKING  
CONFIDENCE: 0.85
REASON: Content appears to be a dance/music video with no cooking or food preparation mentioned.

Now classify the video:"""
        
        return prompt
    
    def _parse_classification_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse the classification response from Gemini."""
        try:
            lines = response.strip().split('\n')
            
            classification = None
            confidence = 0.0
            reason = "Unable to parse classification response"
            
            for line in lines:
                line = line.strip()
                if line.startswith('CLASSIFICATION:'):
                    classification_text = line.split(':', 1)[1].strip().upper()
                    classification = classification_text == 'COOKING'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        # Ensure confidence is between 0 and 1
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        confidence = 0.5  # Default confidence
                elif line.startswith('REASON:'):
                    reason = line.split(':', 1)[1].strip()
            
            # If we couldn't parse classification, try to infer from response text
            if classification is None:
                response_lower = response.lower()
                if any(word in response_lower for word in ['cooking', 'recipe', 'food preparation', 'baking']):
                    classification = True
                    confidence = 0.6  # Lower confidence for inferred classification
                    reason = "Inferred as cooking video from response content"
                else:
                    classification = False
                    confidence = 0.6
                    reason = "Inferred as non-cooking video from response content"
            
            return classification, confidence, reason
            
        except Exception as e:
            logger.warning(f"Error parsing classification response: {e}")
            # Try simple keyword-based fallback
            response_lower = response.lower()
            cooking_keywords = ['recipe', 'cooking', 'bake', 'prepare', 'ingredient', 'cook', 'kitchen', 'food preparation']
            
            cooking_score = sum(1 for keyword in cooking_keywords if keyword in response_lower)
            is_cooking = cooking_score >= 2  # Require at least 2 cooking-related keywords
            
            return is_cooking, 0.5, f"Fallback classification based on keywords (score: {cooking_score})"
    
    def classify_from_keywords(self, text: str, title: str = "") -> Tuple[bool, float, str]:
        """
        Simple keyword-based classification as fallback.
        
        Args:
            text: Text to classify (transcript or description)
            title: Video title (optional)
            
        Returns:
            Tuple of (is_cooking_video, confidence_score, reason)
        """
        combined_text = f"{title} {text}".lower()
        
        # Strong cooking indicators
        strong_cooking_keywords = [
            'recipe', 'resepi', 'masak', 'cook', 'cooking', 'bake', 'baking', 
            'ingredients', 'bahan-bahan', 'cara masak', 'how to cook',
            'tutorial masak', 'cooking tutorial', 'step by step',
            'kitchen', 'dapur', 'chef', 'food preparation'
        ]
        
        # Moderate cooking indicators  
        moderate_cooking_keywords = [
            'food', 'makanan', 'eat', 'makan', 'taste', 'rasa', 'delicious', 'sedap',
            'restaurant', 'kedai makan', 'homemade', 'buatan sendiri'
        ]
        
        # Non-cooking indicators
        non_cooking_keywords = [
            'dance', 'music', 'song', 'lagu', 'menari', 'beauty', 'makeup',
            'fashion', 'travel', 'jalan-jalan', 'shopping', 'unboxing',
            'game', 'gaming', 'challenge', 'prank', 'comedy', 'funny'
        ]
        
        strong_score = sum(1 for keyword in strong_cooking_keywords if keyword in combined_text)
        moderate_score = sum(1 for keyword in moderate_cooking_keywords if keyword in combined_text)
        non_cooking_score = sum(1 for keyword in non_cooking_keywords if keyword in combined_text)
        
        # Calculate cooking score
        cooking_score = (strong_score * 2) + moderate_score
        
        # Decision logic
        if strong_score >= 2:
            return True, 0.9, f"Strong cooking indicators found ({strong_score} strong keywords)"
        elif strong_score >= 1 and moderate_score >= 2:
            return True, 0.8, f"Cooking indicators found ({strong_score} strong, {moderate_score} moderate keywords)"
        elif cooking_score >= 3 and non_cooking_score == 0:
            return True, 0.7, f"Multiple food-related keywords without non-cooking indicators"
        elif non_cooking_score >= 2:
            return False, 0.8, f"Non-cooking indicators found ({non_cooking_score} keywords)"
        elif cooking_score == 0:
            return False, 0.7, "No cooking-related keywords found"
        else:
            return False, 0.5, f"Ambiguous content (cooking score: {cooking_score}, non-cooking: {non_cooking_score})"


def main():
    """Test the video classifier."""
    import os
    
    # Test data
    cooking_transcript = """
    Hari ini saya nak tunjuk macam mana nak buat nasi lemak yang sedap. 
    Bahan-bahan yang kita perlukan adalah beras, santan, daun pandan, 
    garam dan gula. Pertama sekali, kita basuh beras sampai bersih...
    """
    
    non_cooking_transcript = """
    Hello everyone! Today I'm going to show you my daily makeup routine.
    First, I start with a good moisturizer, then I apply primer...
    """
    
    try:
        # Test with environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ GEMINI_API_KEY not found. Testing keyword-based classification only.")
            classifier = None
        else:
            classifier = VideoClassifier(api_key, 'gemini-1.5-flash')
        
        print("🧪 Testing Video Classifier")
        print("=" * 50)
        
        # Test keyword-based classification
        print("\n📝 Keyword-based Classification:")
        
        # Test cooking video
        is_cooking, confidence, reason = VideoClassifier.classify_from_keywords(
            None, cooking_transcript, "Cara Masak Nasi Lemak Sedap"
        )
        print(f"Cooking transcript: {'✅ COOKING' if is_cooking else '❌ NOT COOKING'} "
              f"(confidence: {confidence:.2f}) - {reason}")
        
        # Test non-cooking video
        is_cooking, confidence, reason = VideoClassifier.classify_from_keywords(
            None, non_cooking_transcript, "My Daily Makeup Routine"
        )
        print(f"Non-cooking transcript: {'✅ COOKING' if is_cooking else '❌ NOT COOKING'} "
              f"(confidence: {confidence:.2f}) - {reason}")
        
        # Test AI-based classification if API key is available
        if classifier:
            print("\n🤖 AI-based Classification:")
            
            # Test cooking video
            is_cooking, confidence, reason = classifier.classify_video(
                cooking_transcript, 
                {"title": "Cara Masak Nasi Lemak Sedap", "description": "Tutorial masak nasi lemak"}
            )
            print(f"Cooking transcript: {'✅ COOKING' if is_cooking else '❌ NOT COOKING'} "
                  f"(confidence: {confidence:.2f}) - {reason}")
            
            # Test non-cooking video
            is_cooking, confidence, reason = classifier.classify_video(
                non_cooking_transcript,
                {"title": "My Daily Makeup Routine", "description": "Beauty and skincare tips"}
            )
            print(f"Non-cooking transcript: {'✅ COOKING' if is_cooking else '❌ NOT COOKING'} "
                  f"(confidence: {confidence:.2f}) - {reason}")
        
        print("\n✅ Classification tests completed!")
        
    except Exception as e:
        print(f"❌ Error testing classifier: {e}")


if __name__ == "__main__":
    main()
