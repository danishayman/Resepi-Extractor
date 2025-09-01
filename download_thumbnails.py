#!/usr/bin/env python3
"""
Thumbnail Download Script for Recipe Extractor
Downloads thumbnails from source_url column and updates thumbnail_url with local paths.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from audio_processor import AudioProcessor
from database_manager import DatabaseManager
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thumbnail_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ThumbnailDownloader:
    """Downloads thumbnails for recipes and updates database with local paths."""
    
    def __init__(self, thumbnails_dir: str = "thumbnails"):
        """
        Initialize thumbnail downloader.
        
        Args:
            thumbnails_dir: Directory to save downloaded thumbnails
        """
        self.thumbnails_dir = thumbnails_dir
        self.config = Config()
        self.db = DatabaseManager(self.config)
        self.audio_processor = AudioProcessor()
        
        # Create thumbnails directory
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        logger.info(f"Thumbnails will be saved to: {os.path.abspath(self.thumbnails_dir)}")
    
    def get_recipes_needing_thumbnails(self, limit: int = 1000) -> List[Dict]:
        """
        Get recipes that need thumbnail downloads.
        
        Args:
            limit: Maximum number of recipes to process
            
        Returns:
            List of recipe dictionaries
        """
        try:
            # Get all recipes
            recipes = self.db.list_recipes(limit=limit)
            
            # Filter recipes that need thumbnail downloads
            recipes_needing_thumbnails = []
            
            for recipe in recipes:
                source_url = recipe.get('source_url')
                current_thumbnail = recipe.get('thumbnail_url')
                
                if not source_url:
                    logger.warning(f"Recipe {recipe.get('id')} has no source_url")
                    continue
                
                # Check if we need to download thumbnail
                needs_download = (
                    not current_thumbnail or  # No thumbnail at all
                    current_thumbnail.startswith('http') or  # Remote URL (not local)
                    'tiktokcdn.com' in current_thumbnail or  # Expired TikTok URL
                    not os.path.exists(current_thumbnail)  # Local file doesn't exist
                )
                
                if needs_download:
                    recipes_needing_thumbnails.append(recipe)
                else:
                    logger.debug(f"Recipe {recipe.get('title')} already has valid local thumbnail")
            
            logger.info(f"Found {len(recipes_needing_thumbnails)} recipes needing thumbnail downloads")
            return recipes_needing_thumbnails
            
        except Exception as e:
            logger.error(f"Error getting recipes: {e}")
            return []
    
    def download_recipe_thumbnail(self, recipe: Dict) -> Optional[str]:
        """
        Download thumbnail for a single recipe.
        
        Args:
            recipe: Recipe dictionary from database
            
        Returns:
            Local path to downloaded thumbnail, or None if failed
        """
        source_url = recipe.get('source_url')
        recipe_id = recipe.get('id')
        recipe_title = recipe.get('title', 'Unknown')
        
        try:
            logger.info(f"Processing thumbnail for: {recipe_title}")
            
            # Get fresh video metadata to get thumbnail URL
            video_metadata = self.audio_processor.get_video_info(source_url)
            
            if not video_metadata:
                logger.warning(f"Could not get video metadata for: {recipe_title}")
                return None
            
            thumbnail_url = video_metadata.get('thumbnail')
            if not thumbnail_url:
                logger.warning(f"No thumbnail URL found for: {recipe_title}")
                return None
            
            # Download thumbnail
            local_thumbnail_path = self.audio_processor.download_thumbnail(
                thumbnail_url, 
                source_url, 
                self.thumbnails_dir
            )
            
            if local_thumbnail_path:
                logger.info(f"✅ Downloaded thumbnail for: {recipe_title}")
                return local_thumbnail_path
            else:
                logger.error(f"❌ Failed to download thumbnail for: {recipe_title}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error downloading thumbnail for {recipe_title}: {e}")
            return None
    
    def update_recipe_thumbnail_url(self, recipe_id: str, local_thumbnail_path: str, 
                                   original_thumbnail_url: str = None) -> bool:
        """
        Update recipe's thumbnail_url in database.
        
        Args:
            recipe_id: Recipe ID
            local_thumbnail_path: Path to local thumbnail file
            original_thumbnail_url: Original remote thumbnail URL for tracking
            
        Returns:
            True if successful, False otherwise
        """
        try:
            updates = {
                'thumbnail_url': local_thumbnail_path,
                'extraction_metadata': {
                    'thumbnail_downloaded_at': datetime.now().isoformat(),
                    'local_thumbnail_path': local_thumbnail_path,
                }
            }
            
            # Add original URL to metadata if provided
            if original_thumbnail_url:
                updates['extraction_metadata']['original_thumbnail_url'] = original_thumbnail_url
            
            success = self.db.update_recipe(recipe_id, updates)
            
            if success:
                logger.info(f"✅ Updated database with local thumbnail path")
                return True
            else:
                logger.error(f"❌ Failed to update database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating recipe thumbnail URL: {e}")
            return False
    
    def download_all_thumbnails(self, limit: int = 1000, dry_run: bool = False) -> Tuple[int, int, int]:
        """
        Download thumbnails for all recipes that need them.
        
        Args:
            limit: Maximum number of recipes to process
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            Tuple of (success_count, failed_count, skipped_count)
        """
        logger.info("Starting thumbnail download process...")
        
        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
        
        # Get recipes needing thumbnails
        recipes = self.get_recipes_needing_thumbnails(limit)
        
        if not recipes:
            logger.info("No recipes need thumbnail downloads")
            return 0, 0, 0
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, recipe in enumerate(recipes, 1):
            recipe_id = recipe.get('id')
            recipe_title = recipe.get('title', 'Unknown')
            source_url = recipe.get('source_url')
            current_thumbnail = recipe.get('thumbnail_url')
            
            logger.info(f"\n[{i}/{len(recipes)}] Processing: {recipe_title}")
            logger.info(f"Source URL: {source_url}")
            logger.info(f"Current thumbnail: {current_thumbnail}")
            
            if dry_run:
                logger.info("DRY RUN: Would download thumbnail and update database")
                skipped_count += 1
                continue
            
            try:
                # Download thumbnail
                local_thumbnail_path = self.download_recipe_thumbnail(recipe)
                
                if local_thumbnail_path:
                    # Update database
                    if self.update_recipe_thumbnail_url(
                        recipe_id, 
                        local_thumbnail_path, 
                        current_thumbnail
                    ):
                        success_count += 1
                        logger.info(f"✅ Successfully processed: {recipe_title}")
                    else:
                        failed_count += 1
                        logger.error(f"❌ Failed to update database for: {recipe_title}")
                else:
                    failed_count += 1
                    logger.error(f"❌ Failed to download thumbnail for: {recipe_title}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error processing {recipe_title}: {e}")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"THUMBNAIL DOWNLOAD SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total recipes processed: {len(recipes)}")
        logger.info(f"Successfully downloaded: {success_count}")
        logger.info(f"Failed downloads: {failed_count}")
        logger.info(f"Skipped (dry run): {skipped_count}")
        logger.info(f"Success rate: {(success_count / len(recipes) * 100):.1f}%" if recipes else "N/A")
        logger.info(f"Thumbnails saved to: {os.path.abspath(self.thumbnails_dir)}")
        
        return success_count, failed_count, skipped_count
    
    def verify_thumbnails(self) -> Dict[str, int]:
        """
        Verify that downloaded thumbnails exist and are accessible.
        
        Returns:
            Dictionary with verification statistics
        """
        logger.info("Verifying downloaded thumbnails...")
        
        # Get all recipes
        recipes = self.db.list_recipes(limit=1000)
        
        stats = {
            'total_recipes': len(recipes),
            'has_thumbnail_url': 0,
            'local_thumbnails_exist': 0,
            'local_thumbnails_missing': 0,
            'remote_thumbnails': 0,
            'no_thumbnail': 0
        }
        
        for recipe in recipes:
            thumbnail_url = recipe.get('thumbnail_url')
            
            if not thumbnail_url:
                stats['no_thumbnail'] += 1
                continue
            
            stats['has_thumbnail_url'] += 1
            
            if thumbnail_url.startswith('http'):
                stats['remote_thumbnails'] += 1
            else:
                # Local path
                if os.path.exists(thumbnail_url):
                    stats['local_thumbnails_exist'] += 1
                else:
                    stats['local_thumbnails_missing'] += 1
                    logger.warning(f"Missing thumbnail file: {thumbnail_url} for recipe: {recipe.get('title')}")
        
        # Log verification results
        logger.info(f"\n{'='*50}")
        logger.info(f"THUMBNAIL VERIFICATION RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Total recipes: {stats['total_recipes']}")
        logger.info(f"Recipes with thumbnail_url: {stats['has_thumbnail_url']}")
        logger.info(f"Local thumbnails exist: {stats['local_thumbnails_exist']}")
        logger.info(f"Local thumbnails missing: {stats['local_thumbnails_missing']}")
        logger.info(f"Remote thumbnails: {stats['remote_thumbnails']}")
        logger.info(f"No thumbnail: {stats['no_thumbnail']}")
        
        return stats


def main():
    """Main function to run the thumbnail download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download thumbnails for recipes')
    parser.add_argument('--limit', type=int, default=1000, 
                       help='Maximum number of recipes to process (default: 1000)')
    parser.add_argument('--thumbnails-dir', default='thumbnails',
                       help='Directory to save thumbnails (default: thumbnails)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verify', action='store_true',
                       help='Verify existing thumbnails instead of downloading')
    
    args = parser.parse_args()
    
    try:
        # Initialize downloader
        downloader = ThumbnailDownloader(args.thumbnails_dir)
        
        # Test database connection
        if not downloader.db.test_connection():
            logger.error("❌ Database connection failed")
            return 1
        
        logger.info("✅ Database connection successful")
        
        if args.verify:
            # Verify existing thumbnails
            downloader.verify_thumbnails()
        else:
            # Download thumbnails
            success, failed, skipped = downloader.download_all_thumbnails(
                limit=args.limit,
                dry_run=args.dry_run
            )
            
            # Exit with error code if there were failures
            if failed > 0 and success == 0:
                return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
