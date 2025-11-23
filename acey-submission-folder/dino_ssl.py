import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import math


class DINOHead(nn.Module):
    """
    Projection head for DINO with bottleneck architecture from DINOv2
    """
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, norm_last_layer=True):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        # Bottleneck layer (from DINOv2)
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Last layer with weight normalization
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        if norm_last_layer:
            self.last_layer.weight.data = F.normalize(self.last_layer.weight.data, dim=1)
            self.last_layer.weight.requires_grad = False  # Will be updated manually
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Wrapper to handle multiple crops (global + local views) with untied heads.
    Supports both tied (single head) and untied (separate global/local heads) modes.
    
    For evaluation: extracts normalized backbone features (frozen encoder).
    For training: processes multiple crops through heads.
    """
    def __init__(self, backbone, head, local_head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head  # Global head (or shared head if local_head is None)
        self.local_head = local_head  # Local head (None means tied heads)
        self.use_untied = local_head is not None
    
    def forward(self, x):
        # For evaluation: just extract backbone features (frozen)
        # Input is a single tensor (not a list of crops)
        if isinstance(x, list):
            # If list, concatenate and process (shouldn't happen in eval, but handle it)
            x = torch.cat(x, dim=0)
        
        # Extract backbone features and normalize
        features = self.backbone(x)
        features = F.normalize(features, dim=-1, p=2)
        return features


def get_backbone(arch='vit_small', patch_size=16, img_size=96, drop_path_rate=0.0):
    """
    Create a backbone network. ViT recommended for DINO.
    For 96x96 images, we use smaller models to stay under 100M params.
    
    Args:
        arch: Architecture name ('vit_tiny', 'vit_small', 'vit_base', 'resnet50')
        patch_size: Patch size (not used for timm models, kept for compatibility)
        img_size: Input image size
        drop_path_rate: Stochastic depth rate (drop path rate). Recommended:
            - 0.0-0.1 for ViT-tiny
            - 0.1-0.15 for ViT-small
            - 0.15-0.2 for ViT-base
    """
    if 'vit' in arch:
        # Using timm for ViT models
        if arch == 'vit_tiny':
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False, 
                                     img_size=img_size, num_classes=0,
                                     drop_path_rate=drop_path_rate)
        elif arch == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=False,
                                     img_size=img_size, num_classes=0,
                                     drop_path_rate=drop_path_rate)
        elif arch == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', pretrained=False,
                                     img_size=img_size, num_classes=0,
                                     drop_path_rate=drop_path_rate)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        embed_dim = model.embed_dim
    else:
        # ResNet backbone alternative (stochastic depth not applicable)
        model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        embed_dim = 2048
    
    return model, embed_dim


# Model parameter counting utility
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

