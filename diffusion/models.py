import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# -----------------------------------------------------------------------------
# RETFOUND CONDITIONER
# -----------------------------------------------------------------------------
class RETFoundConditioner(nn.Module):
    def __init__(self, cross_attention_dim=768, retfound_weights=None):
        super().__init__()
        self.vit = None
        self._load_retfound(retfound_weights)
        self.proj = nn.Sequential(
            nn.Linear(1024, 768), nn.GELU(),
            nn.Linear(768, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )
        # O6: register ImageNet normalization as buffers (no per-call allocation)
        self.register_buffer('_norm_mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('_norm_std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_retfound(self, retfound_weights=None):
        candidates = [
            retfound_weights,
            os.path.expanduser("~/RETFound_cfp_weights.pth"),
            os.path.expanduser("~/Downloads/RETFound_cfp_weights.pth"),
        ]
        candidates = [p for p in candidates if p is not None]
        try:
            import timm
            self.vit = timm.create_model('vit_large_patch16_224', pretrained=False,
                                          num_classes=0, global_pool='token')
            ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
            if ckpt_path:
                state       = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model_state = state.get('model', state)
                vit_state   = self.vit.state_dict()
                filtered    = {k: v for k, v in model_state.items()
                               if k in vit_state and v.shape == vit_state[k].shape}
                self.vit.load_state_dict(filtered, strict=False)
                print(f"RETFound loaded ({len(filtered)}/{len(vit_state)} keys)")
            else:
                print("WARNING: RETFound weights not found — random ViT-Large init.")
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.eval()
            print("RETFound frozen.")
        except ImportError:
            from torchvision.models import vit_l_16, ViT_L_16_Weights
            vit_full = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            vit_full.heads = nn.Identity()  # strip classifier, keep 1024-d class token
            self.vit = vit_full
            for p in self.vit.parameters():
                p.requires_grad_(False)
            self.vit.eval()
            print("WARNING: timm not found — using torchvision ViT-L fallback (no RETFound weights)")

    @torch.no_grad()
    def extract_features(self, x):
        # 1. Interpolate to RETFound's expected 224x224
        x_r = F.interpolate(x.float(), size=(224,224), mode='bicubic', align_corners=False)
        
        # 2. Revert from [-1, 1] (Diffusion space) back to [0, 1] (Standard RGB space)
        x_01 = (x_r + 1.0) / 2.0 
        
        x_n  = (x_01 - self._norm_mean) / self._norm_std 
        
        feats = self.vit(x_n)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return feats

    def forward(self, x):
        with torch.no_grad():
            feats = self.extract_features(x)
        return self.proj(feats.float()).unsqueeze(1)

    def train(self, mode=True):
        # Keeps ViT in eval (no dropout drift) — only proj trains
        super().train(mode)
        if self.vit is not None:
            self.vit.eval()
        return self


# -----------------------------------------------------------------------------
# CACHED CONDITIONER (VRAM Saver)
# -----------------------------------------------------------------------------
class CachedConditioner:
    def __init__(self, conditioner):
        self.conditioner = conditioner
        self._vit_cache  = OrderedDict()

    def __call__(self, batch_tensor, batch_keys=None):
        device = batch_tensor.device
        if batch_keys is None:
            with torch.no_grad():
                vit_feats = self.conditioner.extract_features(batch_tensor)
            return self.conditioner.proj(vit_feats.float()).unsqueeze(1)

        results, uncached_idx, uncached_tensors = [None]*len(batch_keys), [], []
        for i, key in enumerate(batch_keys):
            if key in self._vit_cache:
                results[i] = self._vit_cache[key].to(device, non_blocking=True)
                self._vit_cache.move_to_end(key)  # O10: true LRU
            else:
                uncached_idx.append(i); uncached_tensors.append(batch_tensor[i])

        if uncached_tensors:
            ub = torch.stack(uncached_tensors).to(device)
            with torch.no_grad():
                fresh = self.conditioner.extract_features(ub)
            for j, idx in enumerate(uncached_idx):
                self._vit_cache[batch_keys[idx]] = fresh[j].cpu()
                if len(self._vit_cache) > 500:
                    self._vit_cache.popitem(last=False)  # O10: evict oldest
                results[idx] = fresh[j]

        vit_tensor = torch.stack(results)
        return self.conditioner.proj(vit_tensor.float()).unsqueeze(1)

    def train(self):      self.conditioner.train()
    def eval(self):       self.conditioner.eval()
    def parameters(self): return self.conditioner.parameters()

    @property
    def proj(self): return self.conditioner.proj

    def get_full_image_cond(self, full_img_512, paths=None):
        """Get RETFound conditioning from the FULL 512px image.
        paths: None (no cache), str (single image), or list[str] (batch).
        """
        if paths is None:
            return self(full_img_512, None)
        if isinstance(paths, str):
            paths = [paths]
        keys = [(p, "full") for p in paths]
        return self(full_img_512, keys)

    def get_tile_conds_batched(self, img_512, tile_views):
        """Batch all 9 tile ViT passes into a single forward pass."""
        assert img_512.shape[0] == 1, "tile-cond batching assumes B=1"
        tiles = torch.stack([img_512[0, :, h0:h1, w0:w1]
                             for (h0, h1, w0, w1) in tile_views])
        with torch.no_grad():
            feats = self.conditioner.extract_features(tiles)
        local_conds = self.conditioner.proj(feats.float()).unsqueeze(1)
        return [local_conds[i:i+1] for i in range(len(tile_views))]