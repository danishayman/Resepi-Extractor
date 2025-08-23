#!/usr/bin/env python3
"""
Recipe Extraction Automation Script
Automates the process of extracting recipes from multiple video URLs and storing them in Supabase.
"""

import os
import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from config import Config
from batch_processor import BatchProcessor
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recipe_automation.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Check and setup environment variables."""
    required_vars = ['GEMINI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        
        print("\n💡 Setup instructions:")
        print("1. Create a .env file in your project directory")
        print("2. Add the following variables:")
        print("   GEMINI_API_KEY=your_gemini_api_key")
        print("   SUPABASE_URL=your_supabase_project_url")
        print("   SUPABASE_KEY=your_supabase_anon_key")
        print("\n3. Optional variables:")
        print("   WHISPER_MODEL=openai/whisper-large-v3-turbo")
        print("   GEMINI_MODEL=gemini-1.5-flash")
        print("   DEVICE=auto")
        print("   OUTPUT_DIR=./recipes")
        print("   AUDIO_QUALITY=192")
        
        return False
    
    return True

def create_database_tables():
    """Create database tables if they don't exist."""
    try:
        db = DatabaseManager()
        print("📊 Database table setup:")
        db.create_tables()
        print("✅ Please run the SQL commands shown above in your Supabase dashboard")
        return True
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        return False

def process_urls_from_file(file_path: str, 
                          max_workers: int = 2,
                          delay: float = 10.0,
                          retry_failed: bool = False,
                          skip_classification: bool = False,
                          classification_confidence: float = 0.7) -> Dict[str, Any]:
    """
    Process URLs from a file.
    
    Args:
        file_path: Path to file containing URLs
        max_workers: Maximum concurrent workers
        delay: Delay between requests (seconds)
        retry_failed: Whether to retry failed URLs
        skip_classification: Whether to skip cooking video classification
        classification_confidence: Minimum confidence threshold for classification
        
    Returns:
        Processing summary
    """
    try:
        # Create config with classification settings
        config = Config(
            skip_classification=skip_classification,
            classification_confidence=classification_confidence
        )
        
        # Initialize batch processor
        processor = BatchProcessor(
            config=config,
            max_workers=max_workers,
            delay_between_requests=delay
        )
        
        # Test database connection
        if not processor.db_manager.test_connection():
            raise Exception("Database connection failed")
        
        # Load URLs based on file extension
        if file_path.endswith('.json'):
            urls = processor.load_urls_from_json(file_path)
        else:
            urls = processor.load_urls_from_file(file_path)
        
        if not urls:
            raise Exception(f"No valid URLs found in {file_path}")
        
        print(f"🎬 Processing {len(urls)} URLs from {file_path}")
        print(f"⚙️  Settings: {max_workers} workers, {delay}s delay")
        
        # Process URLs
        summary = processor.process_urls(urls, save_progress=True)
        
        # Retry failed URLs if requested
        if retry_failed and summary.get('failed', 0) > 0:
            print(f"\n🔄 Retrying {summary['failed']} failed URLs...")
            retry_summary = processor.retry_failed_urls(f"batch_progress_{int(time.time())}.json")
            
            # Merge summaries
            if 'successful' in retry_summary:
                summary['retry_successful'] = retry_summary['successful']
                summary['retry_failed'] = retry_summary.get('failed', 0)
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing URLs from file: {e}")
        return {'error': str(e)}

def process_single_url(url: str, skip_classification: bool = False, classification_confidence: float = 0.7) -> Dict[str, Any]:
    """
    Process a single URL.
    
    Args:
        url: Video URL to process
        skip_classification: Whether to skip cooking video classification
        classification_confidence: Minimum confidence threshold for classification
        
    Returns:
        Processing result
    """
    try:
        # Create config with classification settings
        config = Config(
            skip_classification=skip_classification,
            classification_confidence=classification_confidence
        )
        
        processor = BatchProcessor(config=config, max_workers=1, delay_between_requests=0)
        
        if not processor.db_manager.test_connection():
            raise Exception("Database connection failed")
        
        print(f"🎬 Processing single URL: {url}")
        
        result = processor.process_single_url(url)
        
        if result['success']:
            print(f"✅ Successfully processed recipe: {result['recipe_id']}")
        else:
            print(f"❌ Failed to process URL: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing single URL: {e}")
        return {'error': str(e)}

def show_database_stats():
    """Show database statistics."""
    try:
        db = DatabaseManager()
        
        if not db.test_connection():
            print("❌ Database connection failed")
            return
        
        stats = db.get_stats()
        
        print("📊 Database Statistics:")
        print("=" * 40)
        print(f"Total recipes: {stats.get('total_recipes', 0)}")
        print(f"Recent recipes (7 days): {stats.get('recent_recipes_7_days', 0)}")
        
        platforms = stats.get('platforms', [])
        if platforms:
            print("\nRecipes by platform:")
            for platform in platforms:
                print(f"  - {platform.get('video_platform', 'unknown')}: {platform.get('count', 0)}")
        
        # Show recent recipes
        recent_recipes = db.list_recipes(limit=5)
        if recent_recipes:
            print(f"\nRecent recipes:")
            for recipe in recent_recipes:
                print(f"  - {recipe.get('title', 'Untitled')} ({recipe.get('video_platform', 'unknown')})")
        
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

def main():
    """Main automation script."""
    parser = argparse.ArgumentParser(
        description="Recipe Extraction Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --file urls.txt                    # Process URLs from file
  %(prog)s --file urls.json --workers 3      # Use 3 concurrent workers
  %(prog)s --url "https://tiktok.com/..."    # Process single URL
  %(prog)s --stats                           # Show database statistics
  %(prog)s --setup                           # Setup database tables
        """
    )
    
    parser.add_argument('--file', '-f', help='File containing URLs to process (.txt or .json)')
    parser.add_argument('--url', '-u', help='Single URL to process')
    parser.add_argument('--workers', '-w', type=int, default=2, help='Maximum concurrent workers (default: 2)')
    parser.add_argument('--delay', '-d', type=float, default=10.0, help='Delay between requests in seconds (default: 10.0)')
    parser.add_argument('--retry', '-r', action='store_true', help='Retry failed URLs')
    parser.add_argument('--stats', '-s', action='store_true', help='Show database statistics')
    parser.add_argument('--setup', action='store_true', help='Setup database tables')
    parser.add_argument('--check-env', action='store_true', help='Check environment variables')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--skip-classification', action='store_true', help='Skip cooking video classification')
    parser.add_argument('--confidence', type=float, default=0.7, help='Minimum confidence threshold for classification (default: 0.7)')
    
    args = parser.parse_args()
    
    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment variables
    if args.check_env or not any([args.file, args.url, args.stats, args.setup]):
        if not setup_environment():
            return 1
        print("✅ Environment variables are properly configured")
    
    # Setup database tables
    if args.setup:
        if not create_database_tables():
            return 1
        return 0
    
    # Show database statistics
    if args.stats:
        show_database_stats()
        return 0
    
    # Process single URL
    if args.url:
        if not setup_environment():
            return 1
        
        result = process_single_url(
            args.url,
            skip_classification=args.skip_classification,
            classification_confidence=args.confidence
        )
        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            return 1
        return 0
    
    # Process URLs from file
    if args.file:
        if not setup_environment():
            return 1
        
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        
        summary = process_urls_from_file(
            args.file,
            max_workers=args.workers,
            delay=args.delay,
            retry_failed=args.retry,
            skip_classification=args.skip_classification,
            classification_confidence=args.confidence
        )
        
        if summary.get('error'):
            print(f"❌ Error: {summary['error']}")
            return 1
        
        print(f"\n🎉 Processing completed!")
        print(f"   Successful: {summary.get('successful', 0)}")
        print(f"   Failed: {summary.get('failed', 0)}")
        print(f"   Skipped (existing): {summary.get('skipped', 0)}")
        print(f"   Non-cooking videos: {summary.get('non_cooking', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        
        return 0
    
    # If no specific action, show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
