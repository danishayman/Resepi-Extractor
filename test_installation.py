# test_installation.py
from recipe_extractor import RecipeExtractor
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test with a small model first
try:
    extractor = RecipeExtractor(
        whisper_model="tiny",
        llm_model="microsoft/DialoGPT-small"
    )
    print("Installation successful!")
except Exception as e:
    print(f"Installation error: {e}")