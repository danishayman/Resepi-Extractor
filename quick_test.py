#!/usr/bin/env python3
"""
Test script for the Recipe Extraction Pipeline
"""

import os
import json
from recipe_extractor import RecipeExtractor

def test_with_sample_video():
    """Test with a real TikTok/Instagram video URL"""
    
    # You can replace this with any cooking video URL from TikTok or Instagram
    test_urls = [
        # Add real URLs here for testing
        "https://www.tiktok.com/@khairulaming/video/7484150259581488392",  # Example
        "https://www.instagram.com/reel/DHK6mXayRPz/?utm_source=ig_web_copy_link",  # Example
    ]
    
    print("🧪 Testing Recipe Extraction Pipeline")
    print("=" * 50)
    
    # Initialize extractor
    extractor = RecipeExtractor(
        whisper_model="openai/whisper-large-v3-turbo",  # Fast and accurate turbo model
        llm_model="microsoft/DialoGPT-medium"  # Much better for structured data extraction
    )
    
    # Test with the first URL (replace with a real one)
    video_url = input("Enter a TikTok or Instagram cooking video URL: ").strip()
    
    if not video_url:
        print("❌ No URL provided. Exiting test.")
        return
    
    try:
        print(f"\n🎥 Processing video: {video_url}")
        print("This may take a few minutes...")
        
        # Process the video
        recipe_data, saved_path = extractor.process_video(
            video_url, 
            "test_recipe.json",
            save_transcript=True,  # Save transcript to txt file
            transcript_dir="transcripts"  # Save in transcripts folder
        )
        
        # Display results
        print("\n" + "✅ SUCCESS!" + " " * 40)
        print(f"📁 Recipe saved to: {os.path.abspath(saved_path)}")
        print("\n📋 Extracted Recipe:")
        print("-" * 30)
        print(json.dumps(recipe_data, ensure_ascii=False, indent=2))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        return False

def test_individual_components():
    """Test individual components of the pipeline"""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    extractor = RecipeExtractor(
        whisper_model="openai/whisper-large-v3-turbo",  # Fast turbo model for testing
        llm_model="microsoft/DialoGPT-medium"  # Better for structured extraction
    )
    
    # Test 1: Audio Download
    print("\n1️⃣ Testing Audio Download...")
    video_url = input("Enter a video URL to test audio download: ").strip()
    
    if video_url:
        try:
            audio_path = extractor.download_audio(video_url, "test_audio")
            print(f"✅ Audio downloaded: {audio_path}")
            
            # Test 2: Transcription
            print("\n2️⃣ Testing Transcription...")
            transcript = extractor.transcribe_audio(audio_path, save_transcript=True, output_dir="test_transcripts")
            print(f"✅ Transcript (first 200 chars): {transcript[:200]}...")
            print(f"📄 Transcript saved to: test_transcripts/ folder")
            
            # Test 3: Recipe Extraction
            print("\n3️⃣ Testing Recipe Extraction...")
            recipe = extractor.extract_recipe(transcript)
            print(f"✅ Recipe extracted:")
            print(json.dumps(recipe, ensure_ascii=False, indent=2))
            
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
        except Exception as e:
            print(f"❌ Component test failed: {e}")
    else:
        print("❌ No URL provided for component testing.")

def test_with_sample_transcript():
    """Test recipe extraction with a sample Bahasa Melayu transcript"""
    
    print("\n📝 Testing with Sample Transcript")
    print("=" * 50)
    
    # Sample cooking transcript in Bahasa Melayu
    sample_transcript = """
    Hari ini kita nak buat nasi lemak yang sedap. Bahan-bahan yang kita perlukan:
    Beras dua cawan, santan satu tin, daun pandan dua helai, garam secukup rasa.
    
    Langkah pertama, basuh beras sampai bersih. Kemudian masukkan santan, 
    daun pandan dan garam. Masak dalam rice cooker selama 20 minit.
    
    Untuk sambal, kita perlukan cili kering, bawang merah, belacan, dan gula.
    Kisar semua bahan sampai halus, kemudian tumis sampai wangi.
    """
    
    extractor = RecipeExtractor(
        whisper_model="openai/whisper-large-v3-turbo",
        llm_model="microsoft/Phi-3-mini-4k-instruct"
    )
    
    try:
        print("🤖 Processing sample transcript...")
        recipe = extractor.extract_recipe(sample_transcript)
        
        print("\n✅ Recipe extracted from transcript:")
        print(json.dumps(recipe, ensure_ascii=False, indent=2))
        
        # Save to file
        output_path = "sample_recipe.json"
        saved_path = extractor.save_recipe(recipe, output_path)
        print(f"\n💾 Sample recipe saved to: {os.path.abspath(saved_path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcript test failed: {e}")
        return False

def test_different_models():
    """Test with different model combinations"""
    
    print("\n🎯 Testing Different Model Combinations")
    print("=" * 50)
    
    model_configs = [
        {
            "name": "Turbo Fast (Recommended)",
            "whisper": "openai/whisper-large-v3-turbo",
            "llm": "microsoft/DialoGPT-medium"
        },
        {
            "name": "Alternative Turbo + Phi-3",
            "whisper": "openai/whisper-large-v3-turbo", 
            "llm": "microsoft/Phi-3-mini-4k-instruct"
        },
        {
            "name": "Standard Large (if turbo has issues)",
            "whisper": "openai/whisper-large-v3",
            "llm": "microsoft/Phi-3.5-mini-instruct"
        },
        {
            "name": "Alternative - Qwen2.5 (Good for multilingual)",
            "whisper": "openai/whisper-large-v3-turbo",
            "llm": "Qwen/Qwen2.5-3B-Instruct"
        },
        {
            "name": "Alternative - Gemma2 (Google model)",
            "whisper": "openai/whisper-large-v3-turbo",
            "llm": "google/gemma-2-2b-it"
        }
    ]
    
    sample_transcript = "Hari ini kita masak rendang. Bahan: daging, santan, rempah."
    
    for config in model_configs:
        print(f"\n🧪 Testing {config['name']} configuration...")
        try:
            extractor = RecipeExtractor(
                whisper_model=config['whisper'],
                llm_model=config['llm']
            )
            
            recipe = extractor.extract_recipe(sample_transcript)
            print(f"✅ {config['name']}: Recipe extracted successfully")
            print(f"   Title: {recipe.get('title', 'N/A')}")
            
        except Exception as e:
            print(f"❌ {config['name']}: Failed - {e}")

def interactive_test_menu():
    """Interactive test menu"""
    
    while True:
        print("\n🍳 Recipe Extraction Pipeline - Test Menu")
        print("=" * 50)
        print("1. Test with real video URL (full pipeline)")
        print("2. Test individual components")
        print("3. Test with sample Bahasa Melayu transcript")
        print("4. Test different model combinations")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ").strip()
        
        if choice == "1":
            test_with_sample_video()
        elif choice == "2":
            test_individual_components()
        elif choice == "3":
            test_with_sample_transcript()
        elif choice == "4":
            test_different_models()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    interactive_test_menu()