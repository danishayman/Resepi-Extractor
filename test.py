#!/usr/bin/env python3
"""
Test script for Recipe Extraction Pipeline with Whisper + Gemini
"""

import os
import json
from recipe_extractor import RecipeExtractor

def setup_gemini_api():
    """Help user set up Gemini API key"""
    
    print("🔑 Gemini API Setup")
    print("=" * 40)
    
    # Check if API key is already set
    existing_key = os.getenv('GEMINI_API_KEY')
    if existing_key:
        print(f"✅ GEMINI_API_KEY environment variable is already set")
        print(f"   Key: {existing_key[:10]}...{existing_key[-4:] if len(existing_key) > 14 else existing_key}")
        return existing_key
    
    print("❌ GEMINI_API_KEY not found in environment variables")
    print("\n📋 To get your Gemini API key:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Click 'Create API key'")
    print("3. Copy the generated key")
    
    print("\n💡 You can set it in two ways:")
    print("Option 1 - Environment variable (recommended):")
    print("   export GEMINI_API_KEY=your_api_key_here")
    print("\nOption 2 - Enter it now (temporary for this session):")
    
    api_key = input("\nEnter your Gemini API key (or press Enter to exit): ").strip()
    
    if not api_key:
        print("❌ No API key provided. Please set GEMINI_API_KEY and try again.")
        return None
    
    # Test the API key
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Test with a simple prompt
        response = model.generate_content("Say hello")
        print("✅ API key is valid!")
        return api_key
        
    except Exception as e:
        print(f"❌ API key validation failed: {e}")
        return None

def test_transcript_extraction():
    """Test recipe extraction with a sample transcript"""
    
    print("\n📝 Testing Recipe Extraction with Sample Transcript")
    print("=" * 60)
    
    # Get API key
    api_key = setup_gemini_api()
    if not api_key:
        return False
    
    # Sample Bahasa Melayu cooking transcript
    sample_transcript = """
    Assalamualaikum dan selamat datang ke channel saya. Hari ini kita akan masak rendang daging yang sedap dan mudah.
    
    Bahan-bahan yang diperlukan untuk rendang daging:
    - Daging lembu 1 kilogram, dipotong kiub
    - Santan pekat 2 tin 400ml
    - Kerisik kelapa 4 sudu besar
    - Serai 3 batang, diketuk
    - Daun kunyit 5 helai
    - Asam keping 6 biji
    - Garam dan gula secukup rasa
    
    Untuk rempah rendang:
    - Cili kering 20 biji
    - Bawang merah 8 ulas
    - Bawang putih 4 ulas
    - Halia 2 inci
    - Lengkuas 3 inci
    - Kunyit hidup 1 inci
    
    Sekarang kita mulakan proses memasak.
    
    Langkah pertama, kisar semua bahan rempah sampai halus dengan sedikit air.
    
    Kemudian panaskan minyak dalam kuali besar. Tumis rempah yang dikisar tadi selama 10 minit sehingga wangi dan tidak berair.
    
    Masukkan daging lembu, gaul rata dengan rempah. Masak selama 5 minit.
    
    Tuangkan santan, serai, dan daun kunyit. Biarkan mendidih dengan api sederhana.
    
    Tambahkan asam keping, garam dan gula. Kacau sesekali dan masak selama 2 jam sehingga daging empuk dan kuah pekat.
    
    Dalam 30 minit terakhir, masukkan kerisik kelapa untuk memberikan rasa yang lebih sedap.
    
    Rendang daging siap untuk dihidangkan. Selamat mencuba!
    """
    
    try:
        # Initialize extractor (using environment variables for all settings except API key)
        extractor = RecipeExtractor(
            gemini_api_key=api_key  # Only override API key, all other settings from env vars
        )
        
        print("🤖 Processing transcript with Gemini...")
        recipe = extractor.extract_recipe(sample_transcript)
        
        print("\n🎉 Recipe extraction completed!")
        print("=" * 50)
        print(json.dumps(recipe, ensure_ascii=False, indent=2))
        
        # Save to file
        output_file = "sample_rendang_recipe.json"
        saved_path = extractor.save_recipe(recipe, output_file)
        
        print(f"\n💾 Recipe saved to: {os.path.abspath(saved_path)}")
        
        # Validate the output
        print(f"\n✅ Validation:")
        print(f"   Title: {recipe.get('title', 'N/A')}")
        print(f"   Ingredients count: {len(recipe.get('ingredients', []))}")
        print(f"   Steps count: {len(recipe.get('steps', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whisper_loading():
    """Test if Whisper model loads correctly"""
    
    print("\n🎤 Testing Whisper Model Loading")
    print("=" * 50)
    
    try:
        print("📥 Loading Whisper model...")
        print("⚠️  Note: First time may take several minutes to download")
        
        from transformers import pipeline
        
        # Get model name from environment variable
        whisper_model = os.getenv('WHISPER_MODEL', 'openai/whisper-large-v3-turbo')
        
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            device=0 if os.system("nvidia-smi") == 0 else -1  # Use GPU if available
        )
        
        print("✅ Whisper model loaded successfully!")
        print(f"   Model: {whisper_model}")
        print(f"   Device: {'GPU (CUDA)' if whisper_pipeline.device.type == 'cuda' else 'CPU'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper loading failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline with a video URL"""
    
    print("\n🎬 Testing Full Pipeline with Video URL")
    print("=" * 50)
    
    # Get API key
    api_key = setup_gemini_api()
    if not api_key:
        return False
    
    print("\n📝 You can test with cooking videos from:")
    print("   - TikTok: @khairulaming, @syedkualalumpur")
    print("   - Instagram: Cooking reels with clear audio")
    print("   - YouTube Shorts: Malaysian cooking videos")
    
    video_url = input("\nEnter a cooking video URL (or press Enter to skip): ").strip()
    
    if not video_url:
        print("⏭️  Skipping full pipeline test")
        return True
    
    try:
        # Initialize extractor (using environment variables for all settings except API key)
        extractor = RecipeExtractor(
            gemini_api_key=api_key  # Only override API key, all other settings from env vars
        )
        
        print(f"\n🎥 Processing video: {video_url}")
        print("⏳ This may take 3-8 minutes depending on video length...")
        
        # Process the complete pipeline
        recipe_data, saved_path = extractor.process_video(video_url, "full_pipeline_recipe.json")
        
        print("\n🎉 FULL PIPELINE SUCCESS!")
        print("=" * 50)
        print(f"📁 Recipe saved to: {os.path.abspath(saved_path)}")
        print(f"\n📋 Extracted Recipe:")
        print(json.dumps(recipe_data, ensure_ascii=False, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        print("\n💡 This could be due to:")
        print("   - Invalid or private video URL")
        print("   - Network connectivity issues")
        print("   - Video platform restrictions")
        print("   - Audio quality too poor for transcription")
        return False

def main():
    """Main test function"""
    
    print("🧪 Recipe Extraction Pipeline Testing")
    print("   Whisper: openai/whisper-large-v3-turbo")
    print("   LLM: Google Gemini 1.5 Flash")
    print("=" * 60)
    
    tests = [
        ("🔑 API Key Setup", lambda: setup_gemini_api() is not None),
        ("🎤 Whisper Model Loading", test_whisper_loading),
        ("📝 Transcript Extraction", test_transcript_extraction),
        ("🎬 Full Pipeline (Optional)", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
                # Ask if user wants to continue
                if "Optional" not in test_name:
                    continue_tests = input(f"\nContinue with remaining tests? (y/n): ").strip().lower()
                    if continue_tests != 'y':
                        break
        
        except KeyboardInterrupt:
            print(f"\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed >= 3:  # API, Whisper, and Transcript tests
        print("🎉 Your pipeline is ready to use!")
        print("💡 You can now extract recipes from cooking videos!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()