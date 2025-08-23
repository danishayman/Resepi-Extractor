#!/usr/bin/env python3
"""
Database Manager for Recipe Extractor
Handles Supabase database operations for storing recipes.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from supabase import create_client, Client
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for recipe storage."""
    
    def __init__(self, config: Config = None):
        """
        Initialize database manager.
        
        Args:
            config: Configuration object with Supabase settings
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # Validate Supabase configuration
        if not self.config.supabase_url or not self.config.supabase_key:
            raise ValueError(
                "Supabase configuration is required. Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    def create_tables(self) -> bool:
        """
        Create necessary tables if they don't exist.
        This method provides the SQL for manual execution.
        
        Returns:
            True if successful
        """
        
        # SQL for creating the recipes table
        recipes_table_sql = """
        CREATE TABLE IF NOT EXISTS recipes (
            id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            source_url TEXT NOT NULL UNIQUE,
            thumbnail_url TEXT,
            video_platform TEXT,
            language TEXT DEFAULT 'ms',
            cuisine_type TEXT,
            difficulty_level TEXT,
            prep_time_minutes INTEGER,
            cook_time_minutes INTEGER,
            total_time_minutes INTEGER,
            servings INTEGER,
            ingredients JSONB NOT NULL,
            instructions JSONB NOT NULL,
            nutrition_info JSONB,
            tags TEXT[],
            transcript TEXT,
            extraction_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_recipes_source_url ON recipes(source_url);
        CREATE INDEX IF NOT EXISTS idx_recipes_title ON recipes(title);
        CREATE INDEX IF NOT EXISTS idx_recipes_cuisine_type ON recipes(cuisine_type);
        CREATE INDEX IF NOT EXISTS idx_recipes_created_at ON recipes(created_at);
        """
        
        logger.info("Table creation SQL:")
        logger.info(recipes_table_sql)
        logger.info("Please execute this SQL in your Supabase dashboard to create the tables.")
        
        return True
    
    def insert_recipe(self, recipe_data: Dict, source_url: str, transcript: str = None, thumbnail_url: str = None) -> Optional[str]:
        """
        Insert a recipe into the database.
        
        Args:
            recipe_data: Recipe data dictionary from extraction
            source_url: Original video URL
            transcript: Audio transcript (optional)
            thumbnail_url: Video thumbnail URL (optional)
            
        Returns:
            Recipe ID if successful, None otherwise
        """
        try:
            # Prepare recipe record
            recipe_record = {
                'title': recipe_data.get('title', 'Untitled Recipe'),
                'description': recipe_data.get('description'),
                'source_url': source_url,
                'thumbnail_url': thumbnail_url,
                'video_platform': self._detect_platform(source_url),
                'language': recipe_data.get('language', 'ms'),
                'cuisine_type': recipe_data.get('cuisine_type'),
                'difficulty_level': recipe_data.get('difficulty_level'),
                'prep_time_minutes': recipe_data.get('prep_time_minutes'),
                'cook_time_minutes': recipe_data.get('cook_time_minutes'),
                'total_time_minutes': recipe_data.get('total_time_minutes'),
                'servings': recipe_data.get('servings'),
                'ingredients': recipe_data.get('ingredients', []),
                'instructions': recipe_data.get('instructions', {}),
                'nutrition_info': recipe_data.get('nutrition_info'),
                'tags': recipe_data.get('tags', []),
                'transcript': transcript,
                'extraction_metadata': {
                    'extracted_at': datetime.now().isoformat(),
                    'whisper_model': self.config.whisper_model,
                    'gemini_model': self.config.gemini_model,
                    'original_data': recipe_data
                }
            }
            
            # Insert into database
            result = self.supabase.table('recipes').insert(recipe_record).execute()
            
            if result.data and len(result.data) > 0:
                recipe_id = result.data[0]['id']
                logger.info(f"Successfully inserted recipe: {recipe_id}")
                return recipe_id
            else:
                logger.error("No data returned from insert operation")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting recipe: {e}")
            return None
    
    def recipe_exists(self, source_url: str) -> bool:
        """
        Check if a recipe with the given source URL already exists.
        
        Args:
            source_url: Video URL to check
            
        Returns:
            True if recipe exists, False otherwise
        """
        try:
            result = self.supabase.table('recipes').select('id').eq('source_url', source_url).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Error checking recipe existence: {e}")
            return False
    
    def get_recipe(self, recipe_id: str = None, source_url: str = None) -> Optional[Dict]:
        """
        Get a recipe by ID or source URL.
        
        Args:
            recipe_id: Recipe UUID
            source_url: Source video URL
            
        Returns:
            Recipe data if found, None otherwise
        """
        try:
            if recipe_id:
                result = self.supabase.table('recipes').select('*').eq('id', recipe_id).execute()
            elif source_url:
                result = self.supabase.table('recipes').select('*').eq('source_url', source_url).execute()
            else:
                raise ValueError("Either recipe_id or source_url must be provided")
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error getting recipe: {e}")
            return None
    
    def list_recipes(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List recipes with pagination.
        
        Args:
            limit: Number of recipes to return
            offset: Number of recipes to skip
            
        Returns:
            List of recipe dictionaries
        """
        try:
            result = (self.supabase.table('recipes')
                     .select('*')
                     .order('created_at', desc=True)
                     .range(offset, offset + limit - 1)
                     .execute())
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error listing recipes: {e}")
            return []
    
    def update_recipe(self, recipe_id: str, updates: Dict) -> bool:
        """
        Update a recipe.
        
        Args:
            recipe_id: Recipe UUID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add updated timestamp
            updates['updated_at'] = datetime.now().isoformat()
            
            result = self.supabase.table('recipes').update(updates).eq('id', recipe_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully updated recipe: {recipe_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating recipe: {e}")
            return False
    
    def delete_recipe(self, recipe_id: str) -> bool:
        """
        Delete a recipe.
        
        Args:
            recipe_id: Recipe UUID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.supabase.table('recipes').delete().eq('id', recipe_id).execute()
            
            if result.data and len(result.data) > 0:
                logger.info(f"Successfully deleted recipe: {recipe_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting recipe: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database stats
        """
        try:
            # Total recipes
            total_result = self.supabase.table('recipes').select('id', count='exact').execute()
            total_recipes = total_result.count if hasattr(total_result, 'count') else 0
            
            # Recipes by platform
            platform_result = (self.supabase.table('recipes')
                             .select('video_platform', count='exact')
                             .execute())
            
            # Recent recipes (last 7 days)
            from datetime import timedelta
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent_result = (self.supabase.table('recipes')
                           .select('id', count='exact')
                           .gte('created_at', week_ago)
                           .execute())
            recent_recipes = recent_result.count if hasattr(recent_result, 'count') else 0
            
            return {
                'total_recipes': total_recipes,
                'recent_recipes_7_days': recent_recipes,
                'platforms': platform_result.data if platform_result.data else []
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def _detect_platform(self, url: str) -> str:
        """
        Detect video platform from URL.
        
        Args:
            url: Video URL
            
        Returns:
            Platform name
        """
        url_lower = url.lower()
        
        if 'tiktok.com' in url_lower:
            return 'tiktok'
        elif 'instagram.com' in url_lower:
            return 'instagram'
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'youtube'
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            return 'facebook'
        else:
            return 'other'
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful
        """
        try:
            # Try to get table info
            result = self.supabase.table('recipes').select('id').limit(1).execute()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


def main():
    """Test the database manager."""
    try:
        # Initialize database manager
        db = DatabaseManager()
        
        # Test connection
        if not db.test_connection():
            print("❌ Database connection failed")
            return
        
        print("✅ Database connection successful")
        
        # Show table creation SQL
        db.create_tables()
        
        # Show stats
        stats = db.get_stats()
        print(f"\n📊 Database Stats:")
        print(f"   Total recipes: {stats.get('total_recipes', 0)}")
        print(f"   Recent recipes (7 days): {stats.get('recent_recipes_7_days', 0)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
