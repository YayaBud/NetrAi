import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

def check_cudnn_status():
    print("="*60)
    print("🚀 CuDNN Diagnostic Tool")
    print("="*60)
    
    # Check basic CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Exiting.")
        sys.exit(1)
        
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not Set')}")
    
    print("\n--- Test 1: Basic Float32 Convolution (Default) ---")
    try:
        x = torch.randn(16, 3, 256, 256, device="cuda")
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        out = conv(x)
        print("✅ Basic Float32 Convolution successful.")
    except Exception as e:
        print(f"❌ Basic Float32 Convolution FAILED:\n{e}")
        
    print("\n--- Test 2: BFloat16 Convolution (AMP Simulation) ---")
    try:
        x = torch.randn(16, 3, 256, 256, device="cuda", dtype=torch.bfloat16)
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda().to(torch.bfloat16)
        out = conv(x)
        print("✅ BFloat16 Convolution successful.")
    except Exception as e:
        print(f"❌ BFloat16 Convolution FAILED:\n{e}")

    print("\n--- Test 3: Channels Last (NHWC) Convolution ---")
    try:
        x = torch.randn(16, 3, 256, 256, device="cuda").to(memory_format=torch.channels_last)
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda().to(memory_format=torch.channels_last)
        out = conv(x)
        print("✅ Channels Last Convolution successful.")
    except Exception as e:
        print(f"❌ Channels Last Convolution FAILED:\n{e}")
        
    print("\n--- Test 4: Heavy VRAM Allocation (MIG Workspace Simulation) ---")
    try:
        # Simulate taking up VRAM before a convolution to see if workspace allocation fails
        dummy_tensors = [torch.randn(1024, 1024, 10, device="cuda") for _ in range(20)]
        x = torch.randn(16, 3, 256, 256, device="cuda")
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        out = conv(x)
        print("✅ Heavy VRAM Allocation Convolution successful.")
        del dummy_tensors
    except Exception as e:
        print(f"❌ Heavy VRAM Allocation Convolution FAILED:\n{e}")
        
    print("\n--- Test 5: Vision Transformer Patch Embed Simulation ---")
    # This directly mimics the failing traceback
    try:
        x = torch.randn(6, 3, 512, 512, device="cuda") # Batch 6, 512x512
        patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16).cuda()
        out = patch_embed(x)
        print("✅ Patch Embed Convolution successful.")
    except Exception as e:
        print(f"❌ Patch Embed Convolution FAILED:\n{e}")

    print("\n="*60)
    print("Diagnostic Complete.")

if __name__ == '__main__':
    check_cudnn_status()
