import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm
import math
import numpy as np


class RandomMaskingGenerator:
    """
    Random masking generator for iBOT.
    Masks patches in a block-wise manner (similar to BEiT).
    """
    def __init__(self, input_size, mask_ratio=0.4, min_num_patches=4, patch_size=16):
        """
        Args:
            input_size: Image size (e.g., 96)
            mask_ratio: Ratio of patches to mask (default: 0.4)
            min_num_patches: Minimum number of patches to keep unmasked
            patch_size: Patch size of the ViT (default: 16)
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.patch_size = patch_size
        self.h_patches = self.height // self.patch_size
        self.w_patches = self.width // self.patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.num_mask = int(mask_ratio * self.num_patches)
        self.min_num_patches = min_num_patches
        
        # Ensure we don't mask everything
        self.num_mask = min(self.num_mask, self.num_patches - self.min_num_patches)
    
    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str
    
    def __call__(self):
        """
        Returns:
            mask: Binary mask (1 for visible, 0 for masked)
        """
        mask = np.ones(self.num_patches, dtype=np.int32)
        mask_indices = np.random.choice(
            self.num_patches, self.num_mask, replace=False
        )
        mask[mask_indices] = 0
        return mask


class BlockwiseMaskingGenerator:
    """
    Block-wise masking generator (more structured masking).
    """
    def __init__(self, input_size, mask_ratio=0.4, min_num_patches=4, 
                 block_size=2, num_blocks=None, patch_size=16):
        """
        Args:
            input_size: Image size (e.g., 96)
            mask_ratio: Ratio of patches to mask
            min_num_patches: Minimum number of patches to keep unmasked
            block_size: Size of each block (in patches)
            num_blocks: Number of blocks to mask (if None, computed from mask_ratio)
            patch_size: Patch size of the ViT (default: 16)
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.patch_size = patch_size
        self.h_patches = self.height // self.patch_size
        self.w_patches = self.width // self.patch_size
        self.num_patches = self.h_patches * self.w_patches
        
        self.block_size = block_size
        self.h_blocks = self.h_patches // self.block_size
        self.w_blocks = self.w_patches // self.block_size
        self.num_blocks = self.h_blocks * self.w_blocks
        
        if num_blocks is None:
            self.num_mask_blocks = int(mask_ratio * self.num_blocks)
        else:
            self.num_mask_blocks = num_blocks
        
        # Ensure we don't mask everything
        self.num_mask_blocks = min(
            self.num_mask_blocks, 
            self.num_blocks - max(1, min_num_patches // (block_size * block_size))
        )
    
    def __call__(self):
        """
        Returns:
            mask: Binary mask (1 for visible, 0 for masked)
        """
        mask = np.ones(self.num_patches, dtype=np.int32)
        
        # Select blocks to mask
        block_indices = np.random.choice(
            self.num_blocks, self.num_mask_blocks, replace=False
        )
        
        for block_idx in block_indices:
            h_block = block_idx // self.w_blocks
            w_block = block_idx % self.w_blocks
            
            # Mask all patches in this block
            for h_offset in range(self.block_size):
                for w_offset in range(self.block_size):
                    h_patch = h_block * self.block_size + h_offset
                    w_patch = w_block * self.block_size + w_offset
                    
                    if h_patch < self.h_patches and w_patch < self.w_patches:
                        patch_idx = h_patch * self.w_patches + w_patch
                        mask[patch_idx] = 0
        
        return mask


class ProgressiveMaskingGenerator:
    """
    Progressive masking generator that increases mask ratio over time.
    Starts with easier masking (lower ratio) and gradually increases difficulty.
    """
    def __init__(self, input_size, mask_ratio_start=0.2, mask_ratio_end=0.4,
                 min_num_patches=4, patch_size=16, total_epochs=100):
        """
        Args:
            input_size: Image size (e.g., 96)
            mask_ratio_start: Initial mask ratio (default: 0.2)
            mask_ratio_end: Final mask ratio (default: 0.4)
            min_num_patches: Minimum number of patches to keep unmasked
            patch_size: Patch size of the ViT (default: 16)
            total_epochs: Total number of training epochs
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.patch_size = patch_size
        self.h_patches = self.height // self.patch_size
        self.w_patches = self.width // self.patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.min_num_patches = min_num_patches
        
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for progressive masking."""
        self.current_epoch = epoch
    
    def __call__(self):
        """
        Returns:
            mask: Binary mask (1 for visible, 0 for masked)
        """
        # Linear interpolation of mask ratio
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        current_mask_ratio = self.mask_ratio_start + \
            (self.mask_ratio_end - self.mask_ratio_start) * progress
        
        num_mask = int(current_mask_ratio * self.num_patches)
        num_mask = min(num_mask, self.num_patches - self.min_num_patches)
        
        mask = np.ones(self.num_patches, dtype=np.int32)
        mask_indices = np.random.choice(
            self.num_patches, num_mask, replace=False
        )
        mask[mask_indices] = 0
        return mask
    
    def __repr__(self):
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        current_mask_ratio = self.mask_ratio_start + \
            (self.mask_ratio_end - self.mask_ratio_start) * progress
        repr_str = "ProgressiveMask: total patches {}, current mask ratio {:.2f}, epoch {}/{}".format(
            self.num_patches, current_mask_ratio, self.current_epoch, self.total_epochs
        )
        return repr_str


class iBOTTokenizer(nn.Module):
    """
    Enhanced online tokenizer for iBOT with LayerNorm and deeper architecture.
    Predicts discrete tokens for masked patches.
    
    Improvements:
    - LayerNorm for training stability
    - Deeper architecture (3 layers instead of 2)
    - Residual connection option
    - SiLU activation (often better than GELU)
    """
    def __init__(self, embed_dim, num_tokens=8192, hidden_dim=512, 
                 use_layernorm=True, use_residual=False, activation='silu'):
        """
        Args:
            embed_dim: Dimension of patch embeddings
            num_tokens: Number of discrete tokens (vocabulary size)
            hidden_dim: Hidden dimension for tokenizer MLP
            use_layernorm: Whether to use LayerNorm (default: True)
            use_residual: Whether to use residual connection (default: False)
            activation: Activation function ('silu', 'gelu', or 'relu')
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.use_residual = use_residual and (embed_dim == hidden_dim)
        
        # Choose activation
        if activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn)
        
        # Middle layer (deeper architecture)
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn)
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, num_tokens))
        
        self.tokenizer = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Patch embeddings [batch_size, num_patches, embed_dim]
        Returns:
            token_logits: Token predictions [batch_size, num_patches, num_tokens]
        """
        if self.use_residual:
            # Residual connection (only if dimensions match)
            residual = x
            out = self.tokenizer(x)
            # Project residual if needed (shouldn't be needed with matching dims)
            return out
        return self.tokenizer(x)


class iBOTHead(nn.Module):
    """
    Enhanced projection head for iBOT with LayerNorm and better activations.
    Used for self-distillation on visible patches.
    
    Improvements:
    - LayerNorm for training stability
    - SiLU activation (often better than GELU)
    - Optional dropout for regularization
    """
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, 
                 nlayers=3, norm_last_layer=True, use_layernorm=True, 
                 activation='silu', dropout=0.0):
        super().__init__()
        
        # Choose activation
        if activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_fn)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        if norm_last_layer:
            self.last_layer.weight.data = F.normalize(self.last_layer.weight.data, dim=1)
            self.last_layer.weight.requires_grad = False
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropiBOTWrapper(nn.Module):
    """
    Wrapper for iBOT that handles multiple crops and masking.
    Teacher sees full images, student sees masked images.
    
    For evaluation: extracts normalized backbone features (frozen encoder).
    For training: processes multiple crops through heads and tokenizers.
    """
    def __init__(self, backbone, head, tokenizer, local_head=None, local_tokenizer=None):
        super().__init__()
        self.backbone = backbone
        self.head = head  # Global head for CLS token (self-distillation)
        self.tokenizer = tokenizer  # Global tokenizer for patch tokens
        self.local_head = local_head  # Local head (optional)
        self.local_tokenizer = local_tokenizer  # Local tokenizer (optional)
        self.use_untied = local_head is not None
    
    def forward(self, x, mask=None, return_patch_tokens=False):
        """
        Args:
            x: Input images [batch_size, 3, H, W] or list of crops
            mask: Not used directly here (masking handled in loss)
            return_patch_tokens: If True, return patch-level token predictions
        
        Returns:
            If return_patch_tokens=False:
                head_output: Projection head output (CLS token)
            If return_patch_tokens=True:
                (head_output, token_logits): Both head output and token predictions
                token_logits: [batch_size, num_patches+1, num_tokens] (includes CLS token)
        """
        # Handle list of crops
        if isinstance(x, list):
            # First 2 are global, rest are local
            n_global = 2
            global_crops = x[:n_global]
            local_crops = x[n_global:] if len(x) > n_global else []
            
            outputs = []
            token_outputs = []
            
            # Process global crops
            if len(global_crops) > 0:
                global_batch = torch.cat(global_crops, dim=0)
                # Backbone returns [batch, num_patches+1, embed_dim] (includes CLS token)
                global_features = self.backbone(global_batch)
                
                # CLS token (first token) for self-distillation
                cls_token = global_features[:, 0]  # [batch, embed_dim]
                head_out = self.head(cls_token)  # [batch, out_dim]
                outputs.append(head_out)
                
                # All tokens (including CLS) for tokenizer
                if return_patch_tokens:
                    # Tokenizer processes all tokens (patches + CLS)
                    token_logits = self.tokenizer(global_features)  # [batch, num_patches+1, num_tokens]
                    token_outputs.append(token_logits)
            
            # Process local crops
            if len(local_crops) > 0:
                local_batch = torch.cat(local_crops, dim=0)
                local_features = self.backbone(local_batch)
                
                cls_token = local_features[:, 0]
                
                if self.use_untied:
                    head_out = self.local_head(cls_token)
                    outputs.append(head_out)
                    if return_patch_tokens and self.local_tokenizer is not None:
                        token_logits = self.local_tokenizer(local_features)
                        token_outputs.append(token_logits)
                else:
                    head_out = self.head(cls_token)
                    outputs.append(head_out)
                    if return_patch_tokens:
                        token_logits = self.tokenizer(local_features)
                        token_outputs.append(token_logits)
            
            if return_patch_tokens:
                return torch.cat(outputs) if outputs else torch.empty(0), \
                       torch.cat(token_outputs) if token_outputs else torch.empty(0)
            return torch.cat(outputs) if outputs else torch.empty(0)
        
        else:
            # Single image (for evaluation)
            features = self.backbone(x)
            # For evaluation, return normalized CLS token
            cls_token = features[:, 0]
            cls_token = F.normalize(cls_token, dim=-1, p=2)
            return cls_token


class iBOTLoss(nn.Module):
    """
    Enhanced iBOT loss combining:
    1. MIM loss: Token prediction for masked patches (with optional focal loss)
    2. Self-distillation loss: Consistency on visible patches (optional)
    3. KoLeo regularization: Feature diversity (from DINOv2)
    
    Improvements:
    - Optional focal loss for MIM (focuses on hard examples)
    - Temperature schedule for tokenizer (adaptive difficulty)
    """
    def __init__(self, out_dim, num_tokens, ncrops=8, 
                 warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=30, nepochs=100,
                 student_temp=0.1, center_momentum=0.9,
                 mim_loss_weight=1.0, cls_loss_weight=1.0,
                 koleo_weight=0.001, koleo_eps=1e-6,
                 mim_temp=0.07, use_focal_loss=False, focal_gamma=2.0):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.ncrops = ncrops
        self.mim_loss_weight = mim_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.koleo_weight = koleo_weight
        self.koleo_eps = koleo_eps
        self.mim_temp = mim_temp
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        
        # For self-distillation (DINO-style)
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature schedule for teacher (CLS distillation)
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
        # Temperature schedule for tokenizer (MIM) - can warmup from higher to lower
        # Lower temperature = sharper distribution = harder task
        mim_temp_start = mim_temp * 1.5  # Start easier
        mim_temp_end = mim_temp  # End at target
        self.mim_temp_schedule = torch.cat((
            torch.linspace(mim_temp_start, mim_temp_end, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * mim_temp_end
        ))
    
    def forward(self, student_output, teacher_output, 
                student_token_logits, teacher_token_logits,
                student_mask, epoch):
        """
        Args:
            student_output: Student head output [batch_size * ncrops, out_dim]
            teacher_output: Teacher head output [batch_size * 2, out_dim] (2 global views)
            student_token_logits: Student token predictions [batch_size * ncrops, num_patches+1, num_tokens]
            teacher_token_logits: Teacher token predictions [batch_size * 2, num_patches+1, num_tokens]
            student_mask: Student mask [batch_size, num_patches] (1=visible, 0=masked)
                          Note: mask is for patches only (excludes CLS token at index 0)
            epoch: Current epoch
        """
        total_loss = 0.0
        batch_size = student_mask.shape[0]
        num_patches = student_mask.shape[1]  # Excluding CLS token
        
        # Track component losses for monitoring
        mim_loss = torch.tensor(0.0, device=student_output.device)
        cls_loss = torch.tensor(0.0, device=student_output.device)
        koleo_loss = torch.tensor(0.0, device=student_output.device)
        
        # 1. MIM Loss: Token prediction for masked patches
        if self.mim_loss_weight > 0:
            # student_token_logits: [batch * ncrops, num_patches+1, num_tokens]
            # teacher_token_logits: [batch * 2, num_patches+1, num_tokens]
            # We only compute loss on patches (exclude CLS token at index 0)
            
            # Process each crop separately
            mim_losses = []
            for crop_idx in range(self.ncrops):
                start_idx = crop_idx * batch_size
                end_idx = (crop_idx + 1) * batch_size
                
                # Get student token predictions for this crop (patches only, exclude CLS)
                student_tokens_crop = student_token_logits[start_idx:end_idx, 1:, :]  # [batch, num_patches, num_tokens]
                
                # Get teacher token predictions (use first global view)
                teacher_tokens_crop = teacher_token_logits[:batch_size, 1:, :]  # [batch, num_patches, num_tokens]
                
                # Apply mask: only compute loss on masked patches
                # student_mask: [batch, num_patches] where 0=masked, 1=visible
                mask_expanded = student_mask.unsqueeze(-1)  # [batch, num_patches, 1]
                masked_patches = (student_mask == 0)  # [batch, num_patches]
                
                if masked_patches.sum() > 0:
                    # Get teacher soft targets for masked patches
                    teacher_tokens_masked = teacher_tokens_crop[masked_patches]  # [num_masked, num_tokens]
                    
                    # Use temperature schedule for tokenizer
                    temp = self.mim_temp_schedule[epoch] if epoch < len(self.mim_temp_schedule) else self.mim_temp_schedule[-1]
                    teacher_tokens_masked = F.softmax(teacher_tokens_masked / temp, dim=-1).detach()
                    
                    # Get student predictions for masked patches
                    student_tokens_masked = student_tokens_crop[masked_patches]  # [num_masked, num_tokens]
                    student_probs = F.softmax(student_tokens_masked, dim=-1)
                    student_log_probs = F.log_softmax(student_tokens_masked, dim=-1)
                    
                    if self.use_focal_loss:
                        # Focal loss: focuses on hard examples
                        # Weight = (1 - p_t)^gamma where p_t is probability of correct class
                        # Higher gamma = more focus on hard examples
                        p_t = torch.sum(teacher_tokens_masked * student_probs, dim=-1)  # [num_masked]
                        focal_weight = (1 - p_t) ** self.focal_gamma
                        ce_loss = torch.sum(-teacher_tokens_masked * student_log_probs, dim=-1)  # [num_masked]
                        mim_loss_crop = (focal_weight * ce_loss).mean()
                    else:
                        # Standard cross-entropy loss
                        mim_loss_crop = torch.sum(
                            -teacher_tokens_masked * student_log_probs, dim=-1
                        ).mean()
                    
                    mim_losses.append(mim_loss_crop)
            
            if len(mim_losses) > 0:
                mim_loss = sum(mim_losses) / len(mim_losses)
                total_loss += self.mim_loss_weight * mim_loss
        
        # 2. Self-distillation loss (DINO-style) on CLS tokens
        if self.cls_loss_weight > 0:
            student_out = student_output / self.student_temp
            student_out = student_out.chunk(self.ncrops)
            
            # Teacher centering and sharpening
            temp = self.teacher_temp_schedule[epoch] if epoch < len(self.teacher_temp_schedule) else self.teacher_temp_schedule[-1]
            teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
            teacher_out = teacher_out.detach().chunk(2)  # 2 global views for teacher
            
            cls_loss_terms = []
            
            for iq, q in enumerate(teacher_out):
                for v in range(len(student_out)):
                    if v == iq:
                        continue
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                    cls_loss_terms.append(loss.mean())
            
            if len(cls_loss_terms) > 0:
                cls_loss = torch.stack(cls_loss_terms).mean()
                total_loss += self.cls_loss_weight * cls_loss
            
            # Update center
            self.update_center(teacher_output)
        
        # 3. KoLeo Regularization: Feature diversity (from DINOv2)
        if self.koleo_weight > 0:
            # Compute KoLeo loss on teacher output to encourage feature diversity
            # KoLeo: -log(det(pairwise_distances)) encourages features to be spread out
            koleo_loss = self.compute_koleo_loss(teacher_output)
            total_loss += self.koleo_weight * koleo_loss
        
        # Return total loss and component losses for monitoring
        return total_loss, {
            'mim_loss': mim_loss.item() if isinstance(mim_loss, torch.Tensor) else mim_loss,
            'cls_loss': cls_loss.item() if isinstance(cls_loss, torch.Tensor) else cls_loss,
            'koleo_loss': koleo_loss.item() if isinstance(koleo_loss, torch.Tensor) else koleo_loss
        }
    
    def compute_koleo_loss(self, features):
        """
        Compute KoLeo regularization loss to encourage feature diversity.
        
        KoLeo loss = -sum(log(pairwise_distances + eps))
        This encourages features to be spread out in the embedding space.
        
        Args:
            features: Feature tensor [batch_size, feature_dim]
        
        Returns:
            koleo_loss: Scalar loss value
        """
        # Normalize features
        features = F.normalize(features, dim=-1, p=2)
        
        # Compute pairwise distances
        # features: [batch, dim]
        # pairwise_dist: [batch, batch] where dist[i,j] = ||features[i] - features[j]||^2
        pairwise_dist = torch.cdist(features, features, p=2) ** 2  # [batch, batch]
        
        # Add small epsilon to avoid log(0)
        pairwise_dist = pairwise_dist + self.koleo_eps
        
        # Mask out diagonal (distance to self is 0, not useful)
        batch_size = features.shape[0]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        
        # Compute negative log of distances (encourages larger distances)
        # Only consider off-diagonal elements
        valid_distances = pairwise_dist[mask]
        
        # KoLeo loss: negative sum of log distances
        # This encourages features to be far apart
        koleo_loss = -torch.sum(torch.log(valid_distances))
        
        # Normalize by number of pairs
        num_pairs = mask.sum().float()
        if num_pairs > 0:
            koleo_loss = koleo_loss / num_pairs
        
        return koleo_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center for teacher output"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentation:
    """
    Multi-crop augmentation for iBOT (similar to DINO).
    """
    def __init__(self, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4),
                 local_crops_number=6, size=96):
        # Global crops (2) - stronger augmentation
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale, 
                                       interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.075, scale=(0.02, 0.33)),
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale,
                                       interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.075, scale=(0.02, 0.33)),
        ])
        
        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.075, scale=(0.02, 0.33)),
        ])
    
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def get_backbone(arch='vit_small', patch_size=16, img_size=96, drop_path_rate=0.0):
    """
    Create a backbone network (ViT).
    """
    if 'vit' in arch:
        vit_kwargs = dict(
            pretrained=False,
            img_size=img_size,
            num_classes=0,   # no classifier head
            drop_path_rate=drop_path_rate,
            global_pool=''   # keep patch + CLS tokens (needed for iBOT)
        )
        if arch == 'vit_tiny':
            model = timm.create_model('vit_tiny_patch16_224', **vit_kwargs)
        elif arch == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', **vit_kwargs)
        elif arch == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', **vit_kwargs)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        embed_dim = model.embed_dim
    else:
        raise ValueError(f"iBOT requires ViT architecture, got: {arch}")
    
    return model, embed_dim


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=10,
                    start_warmup_value=0):
    """Cosine learning rate schedule with warmup"""
    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule


def cosine_scheduler_with_peak(peak_value, final_value, epochs, niter_per_ep, 
                               warmup_epochs=10, start_warmup_value=0):
    """
    Cosine learning rate schedule with warmup to a peak, then cosine decay.
    
    Args:
        peak_value: Peak LR reached after warmup
        final_value: Final LR (minimum)
        epochs: Total number of epochs
        niter_per_ep: Number of iterations per epoch
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Starting value for warmup (usually 0)
    
    Returns:
        schedule: LR schedule tensor
    """
    # Warmup: start_warmup_value -> peak_value
    warmup_schedule = torch.linspace(start_warmup_value, peak_value, warmup_epochs * niter_per_ep)
    
    # Cosine decay: peak_value -> final_value
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (peak_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule


@torch.no_grad()
def update_momentum(student, teacher, m):
    """Momentum update of the teacher network"""
    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """Cancel gradients for the last layer during initial training"""
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LARS(torch.optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    Useful for large batch training and self-supervised learning.
    
    Paper: "Large Batch Training of Convolutional Networks" (2017)
    """
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0, 
                 eta=0.001, eps=1e-9, trust_coefficient=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                       eta=eta, eps=eps, trust_coefficient=trust_coefficient)
        super(LARS, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coefficient = group['trust_coefficient']
            lr = group['lr']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Compute local learning rate using LARS
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)
                
                if param_norm != 0 and grad_norm != 0:
                    # Trust coefficient * param_norm / (grad_norm + weight_decay * param_norm)
                    local_lr = trust_coefficient * param_norm / (grad_norm + weight_decay * param_norm + eps)
                    local_lr = torch.clamp(local_lr, min=0.0, max=10.0)
                else:
                    local_lr = 1.0
                
                # Scale learning rate
                scaled_lr = lr * local_lr
                
                # Add weight decay to gradient
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum).add_(grad, alpha=scaled_lr)
                p.data.add_(momentum_buffer, alpha=-1.0)
        
        return loss


if __name__ == "__main__":
    # Quick test
    print("Testing iBOT components...")
    
    # Test backbone
    backbone, embed_dim = get_backbone('vit_small', img_size=96)
    print(f"Backbone embed_dim: {embed_dim}")
    print(f"Backbone parameters: {count_parameters(backbone):,}")
    
    # Test tokenizer
    tokenizer = iBOTTokenizer(embed_dim, num_tokens=8192)
    print(f"Tokenizer parameters: {count_parameters(tokenizer):,}")
    
    # Test head
    head = iBOTHead(embed_dim, out_dim=8192, bottleneck_dim=256)
    print(f"Head parameters: {count_parameters(head):,}")
    
    # Test masking
    mask_gen = RandomMaskingGenerator(96, mask_ratio=0.4)
    mask = mask_gen()
    print(f"Mask shape: {mask.shape}, masked patches: {(mask == 0).sum()}")
    
    print("\nAll tests passed!")

