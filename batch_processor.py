#!/usr/bin/env python3
"""
Batch Processor for Recipe Extractor
Handles processing multiple video URLs and storing results in database.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import concurrent.futures
from threading import Lock

from config import Config
from recipe_extractor import RecipeExtractor
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes multiple video URLs and stores recipes in database."""
    
    def __init__(self, 
                 config: Config = None,
                 max_workers: int = 2,
                 delay_between_requests: float = 5.0,
                 retry_attempts: int = 3):
        """
        Initialize batch processor.
        
        Args:
            config: Configuration object
            max_workers: Maximum number of concurrent workers
            delay_between_requests: Delay between processing requests (seconds)
            retry_attempts: Number of retry attempts for failed extractions
        """
        self.config = config or Config()
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests
        self.retry_attempts = retry_attempts
        
        # Initialize components
        self.recipe_extractor = RecipeExtractor(
            whisper_model=self.config.whisper_model,
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
            device=self.config.device,
            output_dir=self.config.output_dir,
            audio_quality=self.config.audio_quality
        )
        
        self.db_manager = DatabaseManager(self.config)
        
        # Processing statistics
        self.stats = {
            'total_urls': 0,
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'non_cooking': 0,
            'errors': [],
            'start_time': None,
            'end_time': None
        }
        
        # Thread lock for statistics
        self.stats_lock = Lock()
    
    def load_urls_from_file(self, file_path: str) -> List[str]:
        """
        Load URLs from a text file.
        
        Args:
            file_path: Path to file containing URLs (one per line)
            
        Returns:
            List of URLs
        """
        urls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        if self._is_valid_url(line):
                            urls.append(line)
                        else:
                            logger.warning(f"Invalid URL at line {line_num}: {line}")
            
            logger.info(f"Loaded {len(urls)} valid URLs from {file_path}")
            return urls
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading URLs from file: {e}")
            return []
    
    def load_urls_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load URLs with metadata from a JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of URL dictionaries with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            urls = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        urls.append({'url': item})
                    elif isinstance(item, dict) and 'url' in item:
                        urls.append(item)
                    else:
                        logger.warning(f"Invalid item in JSON: {item}")
            
            logger.info(f"Loaded {len(urls)} URLs from JSON file {file_path}")
            return urls
            
        except Exception as e:
            logger.error(f"Error loading URLs from JSON: {e}")
            return []
    
    def process_single_url(self, url_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single URL and store the result.
        
        Args:
            url_data: Dictionary containing URL and optional metadata
            
        Returns:
            Processing result dictionary
        """
        url = url_data.get('url') if isinstance(url_data, dict) else str(url_data)
        metadata = url_data if isinstance(url_data, dict) else {}
        
        result = {
            'url': url,
            'success': False,
            'recipe_id': None,
            'error': None,
            'processing_time': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        try:
            # Check if recipe already exists
            if self.db_manager.recipe_exists(url):
                logger.info(f"Recipe already exists for URL: {url}")
                result['success'] = True
                result['recipe_id'] = 'existing'
                result['skipped'] = True
                
                with self.stats_lock:
                    self.stats['skipped'] += 1
                
                return result
            
            logger.info(f"Processing URL: {url}")
            
            # Extract recipe from video (includes classification step)
            recipe_data, temp_file_path = self.recipe_extractor.process_video(url)
            
            # Store in database
            recipe_id = self.db_manager.insert_recipe(
                recipe_data=recipe_data,
                source_url=url,
                transcript=recipe_data.get('transcript'),  # If transcript is included in recipe_data
                thumbnail_url=recipe_data.get('video_metadata', {}).get('thumbnail')  # Extract thumbnail URL from video metadata
            )
            
            if recipe_id:
                result['success'] = True
                result['recipe_id'] = recipe_id
                logger.info(f"Successfully processed and stored recipe: {recipe_id}")
                
                with self.stats_lock:
                    self.stats['successful'] += 1
            else:
                result['error'] = "Failed to store recipe in database"
                logger.error(f"Failed to store recipe for URL: {url}")
                
                with self.stats_lock:
                    self.stats['failed'] += 1
            
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
        except ValueError as e:
            # Handle classification errors (non-cooking videos)
            error_msg = str(e)
            if "not cooking-related" in error_msg:
                logger.info(f"Skipping non-cooking video: {url} - {error_msg}")
                result['error'] = error_msg
                result['non_cooking'] = True
                
                with self.stats_lock:
                    self.stats['non_cooking'] += 1
            else:
                logger.error(f"ValueError processing {url}: {error_msg}")
                result['error'] = error_msg
                
                with self.stats_lock:
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'url': url,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    })
        
        except Exception as e:
            error_msg = f"Error processing {url}: {str(e)}"
            logger.error(error_msg)
            result['error'] = str(e)
            
            with self.stats_lock:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        finally:
            result['processing_time'] = time.time() - start_time
            
            with self.stats_lock:
                self.stats['processed'] += 1
        
        return result
    
    def process_urls(self, urls: List[Any], 
                    save_progress: bool = True,
                    progress_file: str = None) -> Dict[str, Any]:
        """
        Process multiple URLs.
        
        Args:
            urls: List of URLs or URL dictionaries
            save_progress: Whether to save progress to file
            progress_file: Path to progress file
            
        Returns:
            Processing summary
        """
        self.stats['total_urls'] = len(urls)
        self.stats['start_time'] = datetime.now().isoformat()
        
        if progress_file is None:
            progress_file = f"batch_progress_{int(time.time())}.json"
        
        results = []
        
        logger.info(f"Starting batch processing of {len(urls)} URLs")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Delay between requests: {self.delay_between_requests}s")
        
        try:
            # Process URLs with controlled concurrency
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_url = {}
                for i, url_data in enumerate(urls):
                    future = executor.submit(self.process_single_url, url_data)
                    future_to_url[future] = (i, url_data)
                    
                    # Add delay between submissions to avoid overwhelming services
                    if i < len(urls) - 1:  # Don't delay after the last submission
                        time.sleep(self.delay_between_requests)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    index, url_data = future_to_url[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Log progress
                        progress = (len(results) / len(urls)) * 100
                        logger.info(f"Progress: {len(results)}/{len(urls)} ({progress:.1f}%) - "
                                  f"URL: {result['url'][:50]}...")
                        
                        # Save progress periodically
                        if save_progress and len(results) % 10 == 0:
                            self._save_progress(results, progress_file)
                        
                    except Exception as e:
                        logger.error(f"Error getting result for URL {url_data}: {e}")
                        results.append({
                            'url': url_data.get('url') if isinstance(url_data, dict) else str(url_data),
                            'success': False,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
        
        except KeyboardInterrupt:
            logger.warning("Batch processing interrupted by user")
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        
        finally:
            self.stats['end_time'] = datetime.now().isoformat()
            
            # Save final results
            if save_progress:
                self._save_progress(results, progress_file)
            
            # Generate summary
            summary = self._generate_summary(results)
            
            logger.info("Batch processing completed")
            self._log_summary(summary)
            
            return summary
    
    def _save_progress(self, results: List[Dict], file_path: str):
        """Save progress to file."""
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'results': results
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Progress saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate processing summary."""
        successful = sum(1 for r in results if r.get('success', False) and not r.get('skipped', False))
        failed = sum(1 for r in results if not r.get('success', False) and not r.get('non_cooking', False))
        skipped = sum(1 for r in results if r.get('skipped', False))
        non_cooking = sum(1 for r in results if r.get('non_cooking', False))
        
        total_time = 0
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            total_time = (end - start).total_seconds()
        
        return {
            'total_urls': len(results),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'non_cooking': non_cooking,
            'success_rate': (successful / len(results)) * 100 if results else 0,
            'total_processing_time': total_time,
            'average_time_per_url': total_time / len(results) if results else 0,
            'errors': [r for r in results if not r.get('success', False) and not r.get('non_cooking', False)],
            'non_cooking_videos': [r for r in results if r.get('non_cooking', False)],
            'successful_recipes': [r['recipe_id'] for r in results if r.get('success', False) and r.get('recipe_id') not in ['existing']],
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_summary(self, summary: Dict[str, Any]):
        """Log processing summary."""
        logger.info("="*60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total URLs: {summary['total_urls']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Skipped (existing): {summary['skipped']}")
        logger.info(f"Non-cooking videos: {summary['non_cooking']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total Time: {summary['total_processing_time']:.1f}s")
        logger.info(f"Average Time per URL: {summary['average_time_per_url']:.1f}s")
        
        if summary['errors']:
            logger.info(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors'][:5]:  # Show first 5 errors
                logger.info(f"  - {error['url']}: {error.get('error', 'Unknown error')}")
            
            if len(summary['errors']) > 5:
                logger.info(f"  ... and {len(summary['errors']) - 5} more errors")
        
        logger.info("="*60)
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        return (url.startswith('http://') or url.startswith('https://')) and len(url) > 10
    
    def retry_failed_urls(self, progress_file: str) -> Dict[str, Any]:
        """
        Retry processing failed URLs from a progress file.
        
        Args:
            progress_file: Path to progress file
            
        Returns:
            Processing summary
        """
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # Extract failed URLs
            failed_urls = [
                r['url'] for r in progress_data.get('results', [])
                if not r.get('success', False)
            ]
            
            if not failed_urls:
                logger.info("No failed URLs to retry")
                return {'message': 'No failed URLs found'}
            
            logger.info(f"Retrying {len(failed_urls)} failed URLs")
            return self.process_urls(failed_urls, save_progress=True, 
                                   progress_file=f"retry_{progress_file}")
        
        except Exception as e:
            logger.error(f"Error retrying failed URLs: {e}")
            return {'error': str(e)}


def main():
    """Example usage of batch processor."""
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            max_workers=2,  # Conservative to avoid rate limiting
            delay_between_requests=10.0  # 10 second delay between requests
        )
        
        # Test database connection
        if not processor.db_manager.test_connection():
            print("❌ Database connection failed. Please check your Supabase configuration.")
            return 1
        
        print("✅ Database connection successful")
        
        # Example: Process URLs from file
        urls_file = "sample_urls.txt"
        if os.path.exists(urls_file):
            urls = processor.load_urls_from_file(urls_file)
            if urls:
                print(f"Processing {len(urls)} URLs from {urls_file}")
                summary = processor.process_urls(urls)
                print("Processing completed!")
            else:
                print(f"No valid URLs found in {urls_file}")
        else:
            print(f"Sample URLs file not found: {urls_file}")
            print("Create this file with video URLs (one per line) to test batch processing")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
