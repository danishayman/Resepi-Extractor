#!/usr/bin/env python3
"""
Simple script to run thumbnail downloads with common options.
"""

import sys
from download_thumbnails import ThumbnailDownloader

def main():
    """Run thumbnail download with default settings."""
    try:
        print("🖼️  Recipe Thumbnail Downloader")
        print("=" * 40)
        
        # Initialize downloader
        downloader = ThumbnailDownloader()
        
        # Test database connection
        if not downloader.db.test_connection():
            print("❌ Database connection failed")
            return 1
        
        print("✅ Database connection successful")
        
        # Ask user what they want to do
        print("\nOptions:")
        print("1. Download all missing thumbnails")
        print("2. Dry run (show what would be done)")
        print("3. Verify existing thumbnails")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting thumbnail download...")
            success, failed, skipped = downloader.download_all_thumbnails()
            
        elif choice == "2":
            print("\n🔍 Running dry run...")
            success, failed, skipped = downloader.download_all_thumbnails(dry_run=True)
            
        elif choice == "3":
            print("\n✅ Verifying thumbnails...")
            downloader.verify_thumbnails()
            
        else:
            print("❌ Invalid choice")
            return 1
        
        print("\n✅ Process completed!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
