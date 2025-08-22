#!/usr/bin/env python3
"""
Quick test script to verify the recipe extraction pipeline works
"""

from recipe_extractor import RecipeExtractor
import json
import os

def quick_test():
    """Quick test with sample data"""
    
    print("🚀 Quick Recipe Extraction Test")
    print("=" * 40)
    
    # Initialize with smaller models for faster testing
    print("🤖 Loading models...")
    extractor = RecipeExtractor(
        whisper_model="base",
        llm_model="mistralai/Mistral-7B-Instruct-v0.2"
    )
    
    # Test with a sample Bahasa Melayu cooking transcript
    sample_transcript = """
    Assalamualaikum semua! Hari ini chef nak ajar korang buat mee goreng mamak yang sedap gila!
    
    Bahan-bahan yang kita perlukan:
    - Mee kuning satu bungkus
    - Tahu dua keping, potong dadu
    - Telur dua biji
    - Bawang besar satu biji, hiris nipis
    - Cili padi lima batang
    - Kicap manis tiga sudu besar
    - Kicap cair satu sudu besar
    - Minyak untuk menumis
    - Garam dan gula secukup rasa
    
    Sekarang kita start masak!
    
    Langkah pertama, panaskan minyak dalam kuali. Masukkan bawang dan cili padi, tumis sampai wangi.
    
    Lepas tu masukkan telur, scramble kejap sampai separa masak.
    
    Kemudian masukkan tahu, goreng sebentar.
    
    Masukkan mee kuning, gaul rata dengan bahan lain.
    
    Tuang kicap manis dan kicap cair, gaul sebati.
    
    Tambah garam dan gula, rasa dan adjust mengikut citarasa masing-masing.
    
    Siap! Mee goreng mamak yang sedap dan mudah. Selamat mencuba!
    """
    
    try:
        print("📝 Processing sample cooking transcript...")
        recipe = extractor.extract_recipe(sample_transcript)
        
        print("\n✅ SUCCESS! Recipe extracted:")
        print("=" * 40)
        print(json.dumps(recipe, ensure_ascii=False, indent=2))
        
        # Save to file
        output_file = "test_recipe_output.json"
        saved_path = extractor.save_recipe(recipe, output_file)
        
        print(f"\n💾 Recipe saved to: {os.path.abspath(saved_path)}")
        print(f"📏 File size: {os.path.getsize(saved_path)} bytes")
        
        # Verify the saved file
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_recipe = json.load(f)
            print(f"\n✅ File verification passed!")
            print(f"   Title: {saved_recipe.get('title', 'N/A')}")
            print(f"   Ingredients count: {len(saved_recipe.get('ingredients', []))}")
            print(f"   Steps count: {len(saved_recipe.get('steps', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_url():
    """Test with a real video URL"""
    
    print("\n🎥 Testing with Real Video URL")
    print("=" * 40)
    
    # Some example URLs you can try (replace with actual working URLs):
    example_urls = [
        # TikTok cooking videos (you'll need to find real ones)
        "https://www.tiktok.com/@khairulaming/video/...",  # Popular Malaysian chef
        "https://www.tiktok.com/@syedkualalumpur/video/...",  # Another popular chef
        
        # Instagram Reels (replace with real URLs)
        "https://www.instagram.com/reel/...",
        
        # YouTube Shorts also work with yt-dlp
        "https://www.youtube.com/shorts/...",
    ]
    
    print("📝 Example URLs you can try:")
    for i, url in enumerate(example_urls, 1):
        print(f"   {i}. {url}")
    
    print("\n⚠️  Note: You need to provide real, working video URLs")
    print("🔍 Try searching for Malaysian cooking videos on TikTok or Instagram")
    
    video_url = input("\nEnter a cooking video URL (or press Enter to skip): ").strip()
    
    if not video_url:
        print("⏭️  Skipping real URL test")
        return True
    
    try:
        extractor = RecipeExtractor()
        
        print(f"🎬 Processing: {video_url}")
        print("⏳ This may take 2-5 minutes...")
        
        recipe_data, saved_path = extractor.process_video(video_url, "real_video_recipe.json")
        
        print(f"\n🎉 SUCCESS! Recipe extracted from video!")
        print(f"📁 Saved to: {os.path.abspath(saved_path)}")
        print(f"\n📋 Recipe Preview:")
        print(json.dumps(recipe_data, ensure_ascii=False, indent=2))
        
        return True
        
    except Exception as e:
        print(f"\n❌ Real URL test failed: {e}")
        print("💡 This might be due to:")
        print("   - Invalid or private video URL")
        print("   - Network connectivity issues") 
        print("   - Video platform restrictions")
        return False

if __name__ == "__main__":
    print("🧪 Starting Recipe Extraction Tests...\n")
    
    # Test 1: Quick test with sample transcript
    success1 = quick_test()
    
    if success1:
        print("\n" + "="*50)
        # Test 2: Optional real URL test
        test_with_real_url()
    
    print(f"\n🏁 Testing completed!")
    print("💡 If the quick test worked, your pipeline is ready to use!")