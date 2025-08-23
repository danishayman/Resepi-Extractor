#!/usr/bin/env python3
"""
Setup script for Recipe Extraction Automation
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from example."""
    env_file = Path(".env")
    example_file = Path(".env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create a basic .env file
    env_content = """# Recipe Extraction Automation - Environment Variables
# Fill in your actual values below

# Required - Gemini API for recipe extraction
GEMINI_API_KEY=your_gemini_api_key_here

# Required - Supabase database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here

# Optional - Model configurations (defaults shown)
WHISPER_MODEL=openai/whisper-large-v3-turbo
GEMINI_MODEL=gemini-1.5-flash
DEVICE=auto
OUTPUT_DIR=./recipes
AUDIO_QUALITY=192

# Optional - Video classification settings
SKIP_CLASSIFICATION=false
CLASSIFICATION_CONFIDENCE=0.7
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please edit it with your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    required_modules = [
        'transformers',
        'torch',
        'google.generativeai',
        'supabase',
        'yt_dlp'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful")
    return True

def main():
    """Main setup function."""
    print("🚀 Recipe Extraction Automation Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Test imports
    if not test_imports():
        return 1
    
    # Create .env file
    if not create_env_file():
        return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - Get Gemini API key from: https://makersuite.google.com/app/apikey")
    print("   - Get Supabase credentials from your project dashboard")
    print("\n2. Setup database tables:")
    print("   python automate_recipes.py --setup")
    print("\n3. Test your setup:")
    print("   python automate_recipes.py --check-env")
    print("\n4. Process URLs:")
    print("   python automate_recipes.py --file sample_urls.txt")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
