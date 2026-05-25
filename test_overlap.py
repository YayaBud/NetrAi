import os
import sys
import glob
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image

# Ensure the project root is at the beginning of sys.path to prevent module shadowing.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == "diffusion" else current_dir
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusion.models import RETFoundConditioner, CachedConditioner
from diffusion.diffusion import multidiffusion_reconstruct_full
from diffusion.data import make_transform

def main():
    print("🔧 Testing Overlap & Masking Logic...")
    config_path = "diffusion/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = device
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("Loading models...")
    # 1. Init Scheduler
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device)

    # 2. Init UNet
    model = UNet2DConditionModel(
        sample_size=256, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D","CrossAttnDownBlock2D", "CrossAttnDownBlock2D","CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D","CrossAttnUpBlock2D", "CrossAttnUpBlock2D","UpBlock2D"),
        cross_attention_dim=768,
    ).to(device)

    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    # 3. Init RETFound
    conditioner = RETFoundConditioner(cross_attention_dim=768, retfound_weights=cfg['paths'].get('retfound_weights')).to(device)
    cached_cond = CachedConditioner(conditioner)

    # 4. Load weights
    ckpt_path = os.path.join(cfg['paths']['checkpoint_dir'], "last.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌ Could not find checkpoint: {ckpt_path}")
        print("Please ensure you have a valid checkpoint in your checkpoint_dir.")
        return
    
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Helper to strip prefix in case it was compiled
    def strip_prefix(sd):
        return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        
    model.load_state_dict(strip_prefix(ckpt['model']))
    conditioner.proj.load_state_dict(strip_prefix(ckpt['conditioner_proj']))
    model.eval()
    cached_cond.eval()

    # 5. Load a test image
    val_dir = cfg['paths']['data_val']
    test_images = glob.glob(os.path.join(val_dir, "*.*"))
    if not test_images:
        print(f"❌ No images found in {val_dir}")
        return
        
    img_path = test_images[0] # Grab the very first validation image
    print(f"Testing on image: {img_path}")
    
    transform = make_transform(cfg['training']['crop_size'], is_train=False)
    raw_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(raw_pil).unsqueeze(0).to(device) # (1, 3, 512, 512)

    # 6. Run Inference
    print("Running MultiDiffusion Reconstruction (this will test overlap & stitching)...")
    with torch.inference_mode():
        recon_512, ms_resid = multidiffusion_reconstruct_full(
            img_tensor, img_path, model, cached_cond, ddim_scheduler,
            alphas_cumprod, device, amp_dtype, device_type,
            simplex_freq=cfg['diffusion']['simplex_freq'],
            simplex_octaves=cfg['diffusion']['simplex_octaves'],
            T_start=cfg['diffusion']['ddim_t_start'],
            n_steps=cfg['diffusion']['ddim_steps'],
            max_lcw=cfg['diffusion']['max_train_lcw']
        )

    # 7. Plotting
    print("Plotting results to overlap_test.png...")
    img_np = ((img_tensor[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    recon_np = ((recon_512[0].cpu().numpy().transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    resid_np = ms_resid[0, 0].cpu().numpy()

    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("1. Original Image")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("2. Stitched Reconstruction\n(Testing overlap seams)")
    plt.imshow(recon_np)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("3. Residual Map\n(Testing fixed circular mask bounds)")
    plt.imshow(resid_np, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("4. Clean Overlay")
    plt.imshow(img_np)
    import matplotlib.cm as cm
    import numpy as np
    norm_resid = np.clip(resid_np / (resid_np.max() + 1e-8), 0, 1)
    overlay = cm.inferno(norm_resid)
    overlay[..., 3] = norm_resid * 0.8  # Scale alpha by residual intensity
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("overlap_test.png", dpi=150)
    print("✅ Done! Check /workspace/retina_Ai/overlap_test.png")

if __name__ == "__main__":
    main()