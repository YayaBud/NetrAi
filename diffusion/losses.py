import torch 
import torch.nn.functional as F 


#-----------------------------------------------------------
#loss functions
#-----------------------------------------------------------


# Implements Min-SNR weighting to balance training across different noise levels.
# 1. Calculates Signal-to-Noise Ratio (SNR) for the given timesteps.
# 2. Clamps SNR at 'gamma' to prevent easy (low-noise) steps from dominating the loss.
# 3. Scales MSE loss so that hard (high-noise) steps have higher relative importance, 
#    improving the model's ability to generate structure from pure noise.

def snr_weighted_loss(pred_noise, noise, alphas_cumprod, timesteps, gamma=2.0):
    """
    SNR-Weighted Loss (Min-SNR Strategy) - Tuned for LCW & MultiDiffusion
    
    This function computes the MSE loss scaled by the Signal-to-Noise Ratio (SNR),
    but with an aggressive clamping threshold (gamma=2.0) that acts as the mathematical 
    counterpart to our Dynamic Global-Local Attention Fusion (LCW) engine.
    
    Architectural Synergy:
    1. Global Structure Priority (High Noise, t ~ 1000): 
       At high noise, LCW is low, forcing reliance on the global RETFound ViT embedding. 
       This function gives maximum weight (1.0) here, heavily penalizing mistakes in 
       coarse anatomical placement (e.g., optic disc location).
       
    2. Anti-Overfitting for Seamless Stitching (Low Noise, t ~ 0): 
       As the image cleans up, LCW scales up to focus on local 256px tiles for capillaries. 
       The aggressive gamma=2.0 forces the loss weight to drop off much faster than standard.
       This intentionally stops the model from obsessing over and memorizing the rigid edges 
       of the 256px tiles, preventing boundary artifacts during 512px MultiDiffusion inference.
       
    Args:
        pred_noise: The model's noise prediction.
        noise: The ground truth Gaussian noise.
        alphas_cumprod: Pre-computed alpha schedule to determine current SNR.
        timesteps: The current noise timestep(s) for the batch.
        gamma: Sensitivity cap (default 2.0). Clamps max weight to prevent local overfitting.
    """
    ac = alphas_cumprod[timesteps].float().view(-1, 1, 1, 1)
    snr = ac / (1 - ac) # SNR is deterministic for each timestep
    weight = torch.clamp(snr, max=gamma) / (snr + 1e-8)
    mse = F.mse_loss(pred_noise.float(), noise.float(), reduction="none").mean(dim=[1,2,3])

    return (weight.squeeze() * mse).mean()

def l1_focal_frequency_loss(pred_x0, x0, retinal_mask=None, alpha=0.05):
    """
    Hybrid Spatial L1 + Focal Frequency Loss (The Sniper & The Anchor)
    Engineered for DDR Dataset Micro-lesion detection.
    """
    # ---------------------------------------------------------
    # 1. THE ANCHOR: Spatial L1 Loss
    # ---------------------------------------------------------
    spatial_diff = torch.abs(pred_x0.float() - x0.float())
    
    if retinal_mask is not None:
        spatial_diff = spatial_diff * retinal_mask
        l_l1 = spatial_diff.sum() / (retinal_mask.sum() * spatial_diff.shape[1] + 1e-8)
    else:
        l_l1 = spatial_diff.mean()

    # ---------------------------------------------------------
    # 2. THE SNIPER: Focal Frequency Loss (FFL)
    # ---------------------------------------------------------
    # CRITICAL EDIT: Mask the inputs BEFORE the FFT. 
    # Otherwise, it wastes gradients learning the black camera borders.
    if retinal_mask is not None:
        pred_fft_in = pred_x0.float() * retinal_mask
        real_fft_in = x0.float() * retinal_mask
    else:
        pred_fft_in = pred_x0.float()
        real_fft_in = x0.float()

    # Get the 2D Fourier Transform
    fft_pred = torch.fft.fft2(pred_fft_in, norm='ortho')
    fft_real = torch.fft.fft2(real_fft_in, norm='ortho')
    
    # Extract the Amplitude
    amp_pred = torch.abs(fft_pred)
    amp_real = torch.abs(fft_real)
    
    # Calculate the raw error in frequency space
    freq_diff = torch.abs(amp_pred - amp_real)
    
    # Focal Weighting
    focal_weight = freq_diff.detach()
    l_freq = (focal_weight * freq_diff).mean()

    # ---------------------------------------------------------
    # 3. HYBRID FUSION
    # ---------------------------------------------------------
    return l_l1 + (alpha * l_freq)
    

def diffusion_loss(pred_noise, noise, pred_x0, x0, retinal_mask,
                   alphas_cumprod, timesteps, snr_gamma=2.0):
                   
    l_snr = snr_weighted_loss(pred_noise, noise, alphas_cumprod, timesteps, snr_gamma)
    
    # Route through the new FFL combo instead of the old pooling MSE
    l_hybrid  = l1_focal_frequency_loss(pred_x0, x0, retinal_mask, alpha=0.05)
    
    # Kept the dict key as 'ms' so train_diffusion_512_v4.py logging doesn't crash
    return 0.6 * l_snr + 0.4 * l_hybrid, {'snr': l_snr.detach(), 'ms': l_hybrid.detach()}