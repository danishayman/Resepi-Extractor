#!/usr/bin/env python3
"""
Example demonstrating thumbnail URL extraction and storage
"""

from audio_processor import AudioProcessor
from database_manager import DatabaseManager
from config import Config

def example_thumbnail_extraction():
    """Example of extracting and storing video thumbnail URLs."""
    
    # Initialize components
    config = Config()
    audio_processor = AudioProcessor()
    db_manager = DatabaseManager(config)
    
    # Example video URL
    video_url = "https://www.tiktok.com/@khairulaming/video/7092671985238478106"
    
    try:
        # Extract video metadata including thumbnail
        print("Extracting video metadata...")
        metadata = audio_processor.get_video_info(video_url)
        
        print(f"Video Title: {metadata.get('title')}")
        print(f"Thumbnail URL: {metadata.get('thumbnail')}")
        print(f"Duration: {metadata.get('duration')} seconds")
        print(f"Uploader: {metadata.get('uploader')}")
        
        # Example recipe data
        sample_recipe_data = {
            'title': 'Sample Recipe',
            'description': 'A sample recipe extracted from video',
            'cuisine_type': 'Masakan Melayu',
            'ingredients': {
                'main_ingredients': [
                    {'name': 'Beras', 'quantity': '2 cawan'}
                ]
            },
            'instructions': {
                'step1': 'Basuh beras hingga bersih'
            }
        }
        
        # Insert recipe with thumbnail URL
        print("\nInserting recipe with thumbnail URL...")
        recipe_id = db_manager.insert_recipe(
            recipe_data=sample_recipe_data,
            source_url=video_url,
            thumbnail_url=metadata.get('thumbnail')
        )
        
        if recipe_id:
            print(f"✅ Recipe inserted successfully with ID: {recipe_id}")
            print(f"   Thumbnail URL stored: {metadata.get('thumbnail')}")
        else:
            print("❌ Failed to insert recipe")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    example_thumbnail_extraction()
