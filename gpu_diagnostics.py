#!/usr/bin/env python3
"""
GPU Diagnostics Script
Checks GPU availability and provides recommendations for optimal device usage.
"""

import torch
import sys
import platform
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import time
import psutil
import subprocess

def check_system_info():
    """Display system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()

def check_cuda_availability():
    """Check CUDA availability and GPU information."""
    print("=" * 60)
    print("CUDA & GPU INFORMATION")
    print("=" * 60)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / (1024**3):.1f} GB")
            print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  Memory allocated: {memory_allocated:.2f} GB")
            print(f"  Memory cached: {memory_cached:.2f} GB")
    else:
        print("No CUDA-capable GPUs found.")
        
        # Check if NVIDIA GPU exists but CUDA is not available
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("\nNVIDIA GPU detected but CUDA not available in PyTorch!")
                print("You may need to reinstall PyTorch with CUDA support.")
                print("\nNVIDIA-SMI output:")
                print(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("No NVIDIA GPUs detected or nvidia-smi not available.")
    
    print()

def benchmark_devices():
    """Benchmark CPU vs GPU performance for model loading and inference."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test with a smaller Whisper model for benchmarking
    model_name = "openai/whisper-tiny"
    
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
    
    results = {}
    
    for device in devices_to_test:
        print(f"\nTesting on {device.upper()}...")
        
        try:
            # Measure model loading time
            start_time = time.time()
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(device)
            
            processor = AutoProcessor.from_pretrained(model_name)
            
            loading_time = time.time() - start_time
            
            # Create dummy audio data for inference test
            import numpy as np
            sample_rate = 16000
            duration = 5  # 5 seconds
            dummy_audio = np.random.randn(sample_rate * duration).astype(np.float32)
            
            # Measure inference time
            start_time = time.time()
            
            inputs = processor(dummy_audio, sampling_rate=sample_rate, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=50)
            
            inference_time = time.time() - start_time
            
            # Memory usage
            if device == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_used = psutil.virtual_memory().used / (1024**3)
            
            results[device] = {
                "loading_time": loading_time,
                "inference_time": inference_time,
                "memory_used": memory_used
            }
            
            print(f"  Model loading time: {loading_time:.2f}s")
            print(f"  Inference time: {inference_time:.2f}s")
            print(f"  Memory used: {memory_used:.2f} GB")
            
            # Cleanup
            del model, processor, inputs
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error testing {device}: {e}")
            results[device] = {"error": str(e)}
    
    return results

def provide_recommendations(results):
    """Provide recommendations based on benchmark results."""
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("🔴 NO GPU DETECTED")
        print("Your system is using CPU only. This will be significantly slower.")
        print("\nTo enable GPU acceleration:")
        print("1. Install NVIDIA drivers if you have an NVIDIA GPU")
        print("2. Install CUDA toolkit")
        print("3. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchaudio")
        print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return
    
    if "cuda" in results and "cpu" in results:
        cuda_result = results["cuda"]
        cpu_result = results["cpu"]
        
        if "error" not in cuda_result and "error" not in cpu_result:
            speedup = cpu_result["inference_time"] / cuda_result["inference_time"]
            
            print("🟢 GPU AVAILABLE AND WORKING")
            print(f"GPU is {speedup:.1f}x faster than CPU for inference")
            
            if speedup > 2:
                print("✅ RECOMMENDATION: Use GPU (significant speedup)")
            elif speedup > 1.2:
                print("⚠️  RECOMMENDATION: Use GPU (moderate speedup)")
            else:
                print("❓ RECOMMENDATION: CPU might be sufficient (minimal speedup)")
            
            print(f"\nPerformance comparison:")
            print(f"  CPU inference: {cpu_result['inference_time']:.2f}s")
            print(f"  GPU inference: {cuda_result['inference_time']:.2f}s")
            print(f"  GPU memory: {cuda_result['memory_used']:.2f}GB")
        else:
            print("⚠️  GPU detected but benchmark failed")
            if "error" in cuda_result:
                print(f"GPU error: {cuda_result['error']}")
    
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION IN YOUR CODE")
    print("=" * 60)
    print("Your RecipeExtractor is configured to:")
    print('- Auto-detect device (device="auto")')
    print('- Use "cuda" if available, otherwise "cpu"')
    
    if torch.cuda.is_available():
        print("\n✅ Your code SHOULD be using GPU automatically")
        print("If it's still using CPU, check the console output for device information")
    else:
        print("\n🔴 Your code will use CPU (no GPU available)")

def check_model_device_usage():
    """Check what device your current models would use."""
    print("=" * 60)
    print("MODEL DEVICE ASSIGNMENT")
    print("=" * 60)
    
    # Simulate your RecipeExtractor device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RecipeExtractor would use: {device}")
    
    # Check memory requirements for your actual models
    models_info = {
        "openai/whisper-large-v3-turbo": "~1.5GB VRAM",
        "microsoft/DialoGPT-medium": "~1GB VRAM",
        "microsoft/Phi-3-mini-4k-instruct": "~2.5GB VRAM"
    }
    
    print("\nModel memory requirements:")
    for model, memory in models_info.items():
        print(f"  {model}: {memory}")
    
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nYour GPU has {total_gpu_memory:.1f}GB total memory")
        
        if total_gpu_memory >= 6:
            print("✅ Sufficient GPU memory for all models")
        elif total_gpu_memory >= 4:
            print("⚠️  Adequate GPU memory (may need to use smaller models)")
        else:
            print("🔴 Limited GPU memory (CPU might be better)")

def main():
    """Run complete GPU diagnostics."""
    print("GPU DIAGNOSTICS FOR RECIPE EXTRACTOR")
    print("=" * 60)
    
    check_system_info()
    check_cuda_availability()
    
    print("Running benchmark (this may take a few minutes)...")
    results = benchmark_devices()
    
    provide_recommendations(results)
    check_model_device_usage()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run this script to see your current GPU status")
    print("2. If GPU is available, your RecipeExtractor should use it automatically")
    print("3. Monitor console output when running recipe_extractor.py to confirm device usage")
    print("4. Consider using smaller models if GPU memory is limited")

if __name__ == "__main__":
    main()
