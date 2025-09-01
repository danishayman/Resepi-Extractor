#!/usr/bin/env python3
"""
Script to fix expired thumbnail URLs by re-downloading thumbnails.
"""

import logging
from audio_processor import AudioProcessor
from database_manager import DatabaseManager
from config import Config
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_expired_thumbnails():
    """Re-download thumbnails for all recipes with expired URLs."""
    
    # Initialize components
    config = Config()
    db = DatabaseManager(config)
    audio_processor = AudioProcessor()
    
    # Get all recipes
    recipes = db.list_recipes(limit=1000)  # Adjust limit as needed
    
    logger.info(f"Found {len(recipes)} recipes to check")
    
    fixed_count = 0
    failed_count = 0
    
    for recipe in recipes:
        try:
            source_url = recipe.get('source_url')
            current_thumbnail = recipe.get('thumbnail_url')
            
            if not source_url:
                logger.warning(f"Recipe {recipe['id']} has no source URL")
                continue
            
            # Check if current thumbnail is a TikTok CDN URL (expired)
            if current_thumbnail and 'tiktokcdn.com' in current_thumbnail:
                logger.info(f"Fixing thumbnail for recipe: {recipe['title']}")
                
                # Get fresh video metadata
                video_metadata = audio_processor.get_video_info(source_url)
                
                if video_metadata.get('thumbnail'):
                    # Download new thumbnail
                    local_thumbnail = audio_processor.download_thumbnail(
                        video_metadata['thumbnail'], 
                        source_url
                    )
                    
                    if local_thumbnail:
                        # Update database record
                        updates = {
                            'thumbnail_url': local_thumbnail,
                            'extraction_metadata': {
                                **recipe.get('extraction_metadata', {}),
                                'thumbnail_fixed_at': datetime.now().isoformat(),
                                'original_expired_thumbnail': current_thumbnail
                            }
                        }
                        
                        if db.update_recipe(recipe['id'], updates):
                            logger.info(f"✅ Fixed thumbnail for: {recipe['title']}")
                            fixed_count += 1
                        else:
                            logger.error(f"❌ Failed to update database for: {recipe['title']}")
                            failed_count += 1
                    else:
                        logger.error(f"❌ Failed to download thumbnail for: {recipe['title']}")
                        failed_count += 1
                else:
                    logger.warning(f"⚠️ No thumbnail available for: {recipe['title']}")
            else:
                logger.info(f"✓ Thumbnail OK for: {recipe['title']}")
                
        except Exception as e:
            logger.error(f"❌ Error processing recipe {recipe.get('title', 'Unknown')}: {e}")
            failed_count += 1
    
    logger.info(f"Thumbnail fix complete: {fixed_count} fixed, {failed_count} failed")

if __name__ == "__main__":
    fix_expired_thumbnails()
