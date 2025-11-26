import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import csv
from contextlib import nullcontext
from ibot_ssl import (
    get_backbone, iBOTHead, iBOTTokenizer, MultiCropiBOTWrapper, iBOTLoss,
    DataAugmentation, cosine_scheduler, cosine_scheduler_with_peak, update_momentum,
    cancel_gradients_last_layer, count_parameters,
    RandomMaskingGenerator, BlockwiseMaskingGenerator
)


def create_grad_scaler():
    """Handle newer and older AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    # Fallback for older torch versions
    return torch.cuda.amp.GradScaler()


def autocast_context(fp16_enabled):
    """Return the correct autocast context manager."""
    if not fp16_enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def get_vit_patch_size(backbone):
    """Extract patch size from timm ViT backbone."""
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "patch_size"):
        patch_size = backbone.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            return patch_size[0]
        return patch_size
    return 16


class UnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled pretraining images.
    Supports both local directory and Hugging Face datasets.
    """
    def __init__(self, root_dir, transform=None, use_hf_dataset=False, 
                 hf_dataset=None, hf_dataset_name=None):
        self.transform = transform
        self.use_hf_dataset = use_hf_dataset
        self._hf_dataset = None
        self.hf_dataset_name = hf_dataset_name
        
        if use_hf_dataset:
            if hf_dataset is not None:
                self._hf_dataset = hf_dataset
                self.length = len(hf_dataset)
            elif hf_dataset_name is not None:
                self.length = None
            else:
                raise ValueError("Either hf_dataset or hf_dataset_name must be provided")
            print(f"Using Hugging Face dataset: {hf_dataset_name or 'loaded'}")
        else:
            self.root_dir = Path(root_dir)
            self.image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                self.image_paths.extend(self.root_dir.rglob(ext))
            self.length = len(self.image_paths)
            print(f"Found {self.length} images in {root_dir}")
    
    @property
    def hf_dataset(self):
        """Lazy load Hugging Face dataset for multiprocessing compatibility."""
        if self.use_hf_dataset and self._hf_dataset is None:
            from datasets import load_dataset
            self._hf_dataset = load_dataset(self.hf_dataset_name, split="train")
            if self.length is None:
                self.length = len(self._hf_dataset)
        return self._hf_dataset
    
    def __len__(self):
        if self.length is None:
            _ = self.hf_dataset
        return self.length
    
    def __getitem__(self, idx):
        try:
            if self.use_hf_dataset:
                example = self.hf_dataset[idx]
                image = example["image"]
                
                if isinstance(image, Image.Image):
                    pass
                elif hasattr(image, '__array__'):
                    image = Image.fromarray(np.asarray(image))
                else:
                    try:
                        image = Image.fromarray(image)
                    except Exception as e:
                        if hasattr(image, 'tobytes'):
                            image = Image.frombytes(image.mode, image.size, image.tobytes())
                        else:
                            raise ValueError(f"Unable to convert image to PIL Image: {type(image)}, error: {e}")
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0  # Return dummy label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            raise


def apply_mask_to_patches(patches, mask, patch_size=16):
    """
    Apply mask to patches. For ViT, we need to mask the patch embeddings.
    In practice, we'll mask the input patches before they go through the backbone.
    
    Args:
        patches: Not used directly (we'll mask in the forward pass)
        mask: Binary mask [num_patches] (1=visible, 0=masked)
        patch_size: Patch size
    
    Returns:
        mask_tensor: Mask tensor for the model
    """
    return torch.from_numpy(mask).float()


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
                   data_loader, optimizer, lr_schedule, wd_schedule,
                   momentum_schedule, epoch, fp16_scaler, args, mask_generator):
    """
    Train for one epoch with iBOT.
    """
    loss_sum = 0.0
    loss_count = 0
    all_losses = []
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for it, (images, _) in enumerate(pbar):
        # Update learning rate and weight decay
        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            # Weight decay scheduling works for both LARS and AdamW
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it_global]
        
        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        
        # Generate mask for student views (same mask for all crops in a batch)
        batch_size = images[0].shape[0]
        num_patches = (args.image_size // 16) ** 2  # Assuming patch size 16
        
        # Generate a single mask per image in the batch (shared across crops)
        mask = mask_generator()
        student_mask = torch.from_numpy(mask).float().unsqueeze(0).repeat(batch_size, 1).cuda(non_blocking=True)
        
        # Forward pass
        with autocast_context(fp16_scaler is not None):
            # Teacher: full images (no masking)
            teacher_output, teacher_token_logits = teacher(
                images[:2], mask=None, return_patch_tokens=True
            )
            
            # Student: same images but we'll apply masking in loss computation
            # Note: The backbone processes all patches, but loss only computed on masked ones
            student_output, student_token_logits = student(
                images, mask=None, return_patch_tokens=True
            )
            
            # Compute loss
            loss, loss_components = ibot_loss(
                student_output=student_output,
                teacher_output=teacher_output,
                student_token_logits=student_token_logits,
                teacher_token_logits=teacher_token_logits,
                student_mask=student_mask,
                epoch=epoch
            )
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)
        
        # Loss computation validation checks
        loss_value = loss.item()
        
        # Check 1: Verify loss is in expected range for iBOT
        # Properly computed iBOT loss should start around 4-6 and decrease to 1-3
        if epoch == 0 and it == 0:
            if loss_value > 15.0:
                print(f"WARNING: Initial loss ({loss_value:.4f}) is very high (>15). "
                      f"Expected range: 4-6. Check for loss multiplication issues.")
            elif loss_value > 10.0:
                print(f"WARNING: Initial loss ({loss_value:.4f}) is high (>10). "
                      f"Expected range: 4-6. Monitor closely.")
        elif epoch == 0 and it < 10:
            # Early in first epoch, loss should be decreasing
            if loss_value > 12.0:
                print(f"WARNING: Loss ({loss_value:.4f}) is very high early in training. "
                      f"Check loss computation.")
        
        # Check 2: Verify loss components sum approximately to total loss
        # Total loss = mim_loss * mim_weight + cls_loss * cls_weight + koleo_loss * koleo_weight
        expected_total = (
            loss_components['mim_loss'] * args.mim_loss_weight +
            loss_components['cls_loss'] * args.cls_loss_weight +
            loss_components['koleo_loss'] * args.koleo_weight
        )
        diff = abs(loss_value - expected_total)
        if diff > 0.1:  # Allow small numerical differences
            if it % 100 == 0:  # Only warn occasionally to avoid spam
                print(f"WARNING: Loss component mismatch! "
                      f"Total={loss_value:.4f}, Sum of components={expected_total:.4f}, "
                      f"Diff={diff:.4f}")
        
        # Check 3: Verify loss is not being multiplied by batch size or num_crops
        # iBOT loss should already be averaged, so it shouldn't scale with batch size
        if epoch == 0 and it == 0:
            # Rough sanity check: loss shouldn't be proportional to batch size
            # If loss is way too high, might be multiplied by something
            if loss_value > batch_size * 0.5:
                print(f"WARNING: Loss ({loss_value:.4f}) might be multiplied by batch size "
                      f"({batch_size}) or number of crops. Check loss computation.")
        
        # Backward pass
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        # Loss component monitoring (every 5 epochs or every 100 iterations)
        if (epoch % 5 == 0 and it == 0) or (it % 100 == 0):
            print(f"Loss components: mim={loss_components['mim_loss']:.4f}, "
                  f"cls={loss_components['cls_loss']:.4f}, "
                  f"koleo={loss_components['koleo_loss']:.4f}")
        
        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]
            update_momentum(student, teacher_without_ddp, m)
        
        # Logging
        torch.cuda.synchronize()
        loss_value = loss.item()
        loss_sum += loss_value
        loss_count += 1
        all_losses.append(loss_value)
        
        avg_loss = loss_sum / loss_count
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    # Check 3: Detect unexpected loss increases at end of epoch
    if epoch > 2 and len(all_losses) >= len(data_loader):
        # Compare current epoch average to previous epoch average
        prev_epoch_losses = all_losses[:-len(data_loader)] if len(all_losses) > len(data_loader) else []
        if len(prev_epoch_losses) >= len(data_loader):
            prev_avg = sum(prev_epoch_losses[-len(data_loader):]) / len(data_loader)
            curr_avg = avg_loss
            if curr_avg > prev_avg * 1.5 and prev_avg < 5.0:
                print(f"WARNING: Loss increased significantly at end of epoch {epoch}: "
                      f"current avg={curr_avg:.4f} vs previous avg={prev_avg:.4f} "
                      f"(increase of {((curr_avg/prev_avg - 1) * 100):.1f}%)")
    
    metric_logger = {
        'loss': avg_loss,
        'lr': optimizer.param_groups[0]["lr"],
        'wd': optimizer.param_groups[0]["weight_decay"],
        'all_losses': all_losses
    }
    
    return metric_logger


def main(args):
    print("=" * 80)
    print("iBOT Self-Supervised Learning")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation
    transform = DataAugmentation(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.05, 0.32),
        local_crops_number=args.local_crops_number,
        size=args.image_size
    )
    
    # Dataset and DataLoader
    use_hf_dataset = args.data_path.startswith("hf://")
    hf_dataset = None
    
    if use_hf_dataset:
        from datasets import load_dataset
        hf_dataset_name = args.data_path.replace("hf://", "")
        print(f"Loading Hugging Face dataset: {hf_dataset_name}")
        hf_dataset = load_dataset(hf_dataset_name, split="train")
        dataset = UnlabeledImageDataset(
            args.data_path,
            transform=transform,
            use_hf_dataset=True,
            hf_dataset=hf_dataset,
            hf_dataset_name=hf_dataset_name
        )
    else:
        dataset = UnlabeledImageDataset(args.data_path, transform=transform)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of iterations per epoch: {len(data_loader)}")
    
    # Build student and teacher networks
    print("\nBuilding models...")
    if args.drop_path_rate is None:
        if 'tiny' in args.arch:
            drop_path_rate = 0.15
        elif 'small' in args.arch:
            drop_path_rate = 0.2
        elif 'base' in args.arch:
            drop_path_rate = 0.25
        else:
            drop_path_rate = 0.0
    else:
        drop_path_rate = args.drop_path_rate
    
    print(f"Using drop_path_rate: {drop_path_rate}")
    
    student_backbone, embed_dim = get_backbone(
        args.arch, img_size=args.image_size, drop_path_rate=drop_path_rate
    )
    teacher_backbone, _ = get_backbone(
        args.arch, img_size=args.image_size, drop_path_rate=drop_path_rate
    )
    
    # Create heads and tokenizers
    student_global_head = iBOTHead(
        embed_dim, args.out_dim, bottleneck_dim=args.bottleneck_dim,
        norm_last_layer=args.norm_last_layer
    )
    student_global_tokenizer = iBOTTokenizer(
        embed_dim, num_tokens=args.num_tokens, hidden_dim=args.tokenizer_hidden_dim
    )
    
    student_local_head = iBOTHead(
        embed_dim, args.out_dim, bottleneck_dim=args.bottleneck_dim,
        norm_last_layer=False
    )
    student_local_tokenizer = iBOTTokenizer(
        embed_dim, num_tokens=args.num_tokens, hidden_dim=args.tokenizer_hidden_dim
    )
    
    # Teacher only uses global head/tokenizer
    teacher_global_head = iBOTHead(
        embed_dim, args.out_dim, bottleneck_dim=args.bottleneck_dim
    )
    teacher_global_tokenizer = iBOTTokenizer(
        embed_dim, num_tokens=args.num_tokens, hidden_dim=args.tokenizer_hidden_dim
    )
    
    # Create models
    student = MultiCropiBOTWrapper(
        student_backbone, student_global_head, student_global_tokenizer,
        local_head=student_local_head, local_tokenizer=student_local_tokenizer
    )
    teacher = MultiCropiBOTWrapper(
        teacher_backbone, teacher_global_head, teacher_global_tokenizer,
        local_head=None, local_tokenizer=None
    )
    
    # Move to GPU
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Initialize teacher with student weights
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    filtered_state = {k: v for k, v in student_state.items() 
                     if k in teacher_state and teacher_state[k].shape == v.shape}
    teacher.load_state_dict(filtered_state, strict=False)
    
    # No gradients for teacher
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Check backbone parameters (assignment requirement: backbone < 100M)
    backbone_params = count_parameters(student_backbone)
    total_params = count_parameters(student)
    
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Total model parameters (backbone + heads + tokenizers): {total_params:,}")
    
    # Assignment requirement: Backbone must be strictly < 100M
    if backbone_params >= 100_000_000:
        print(f"\n❌ ERROR: Backbone has {backbone_params:,} parameters (limit is < 100M)")
        print("This violates assignment requirement #3: 'Backbone parameters must be strictly < 100M'")
        sys.exit(1)
    else:
        print(f"✅ Backbone is under 100M limit ({backbone_params:,} < 100,000,000)")
    
    # Also warn if total model is very large (for reference)
    if total_params >= 100_000_000:
        print(f"⚠️  WARNING: Total model has {total_params:,} parameters")
        print("   (Note: Assignment only restricts backbone, not total model)")
    
    patch_size = get_vit_patch_size(student_backbone)

    # Mask generator
    if args.mask_type == 'random':
        mask_generator = RandomMaskingGenerator(
            args.image_size,
            mask_ratio=args.mask_ratio,
            patch_size=patch_size
        )
    elif args.mask_type == 'blockwise':
        mask_generator = BlockwiseMaskingGenerator(
            args.image_size,
            mask_ratio=args.mask_ratio,
            block_size=args.block_size,
            patch_size=patch_size
        )
    else:
        raise ValueError(f"Unknown mask type: {args.mask_type}")
    
    print(f"Using {args.mask_type} masking with ratio {args.mask_ratio}")
    
    # Loss
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.num_tokens,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
        mim_loss_weight=args.mim_loss_weight,
        cls_loss_weight=args.cls_loss_weight,
        koleo_weight=args.koleo_weight,
        koleo_eps=args.koleo_eps
    ).to(device)
    
    # Optimizer - LARS for large batch training
    from ibot_ssl import LARS
    
    params_groups = [
        {'params': [p for n, p in student.named_parameters() if 'last_layer' not in n]},
        {'params': [p for n, p in student.named_parameters() if 'last_layer' in n],
         'weight_decay': 0.0, 'lr': args.lr * args.lr_last_layer_scale}
    ]
    
    if args.optimizer == 'lars':
        optimizer = LARS(
            params_groups,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            trust_coefficient=args.lars_trust_coefficient,
            eta=args.lars_eta
        )
        print(f"Using LARS optimizer with trust_coefficient={args.lars_trust_coefficient}")
    else:
        optimizer = torch.optim.AdamW(
            params_groups,
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        print("Using AdamW optimizer")
    
    # Learning rate schedule
    # For AdamW, don't scale by batch size (unlike LARS)
    if args.optimizer == 'lars':
        base_lr = args.lr * args.batch_size / 256.
    else:
        base_lr = args.lr
    
    # Use peak LR scheduler if max_lr is specified, otherwise use standard scheduler
    if hasattr(args, 'max_lr') and args.max_lr is not None and args.max_lr > base_lr:
        lr_schedule = cosine_scheduler_with_peak(
            args.max_lr,
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
        print(f"Using peak LR schedule: warmup to {args.max_lr}, then decay to {args.min_lr}")
    else:
        lr_schedule = cosine_scheduler(
            base_lr,
            args.min_lr,
            args.epochs, len(data_loader),
            warmup_epochs=args.warmup_epochs,
        )
    
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    # Momentum schedule
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs, len(data_loader)
    )
    
    print(f"\nLoss, optimizer and schedulers ready.")
    
    # Mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = create_grad_scaler()
        print("Using mixed precision training")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    # Setup loss logging
    loss_log_path = os.path.join(args.output_dir, 'loss_log.csv')
    loss_log_file = open(loss_log_path, 'w', newline='')
    loss_log_writer = csv.writer(loss_log_file)
    loss_log_writer.writerow(['epoch', 'iteration', 'loss', 'learning_rate'])
    print(f"Loss logging to: {loss_log_path}")
    
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        if 'fp16_scaler' in checkpoint and fp16_scaler is not None:
            fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
        print(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            student, teacher, teacher,
            ibot_loss, data_loader,
            optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch,
            fp16_scaler, args, mask_generator
        )
        
        # Log losses
        if 'all_losses' in train_stats:
            for it, loss_val in enumerate(train_stats['all_losses']):
                it_global = len(data_loader) * epoch + it
                lr_val = lr_schedule[it_global] if it_global < len(lr_schedule) else lr_schedule[-1]
                loss_log_writer.writerow([epoch, it, f'{loss_val:.6f}', f'{lr_val:.8e}'])
            loss_log_file.flush()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': args,
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth')
            torch.save(save_dict, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        stats_to_print = {k: v for k, v in train_stats.items() if k != 'all_losses'}
        print(f"Epoch {epoch} stats: {stats_to_print}")
    
    loss_log_file.close()
    print(f"Loss log saved to: {loss_log_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_checkpoint.pth')
    torch.save({
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'args': args,
    }, final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', add_help=False)
    
    # Model parameters
    parser.add_argument('--arch', default='vit_tiny', type=str,
                       choices=['vit_tiny', 'vit_small', 'vit_base'],
                       help='Architecture')
    parser.add_argument('--image_size', default=96, type=int, help='Image size')
    parser.add_argument('--out_dim', default=8192, type=int,
                       help='Dimensionality of the iBOT head output')
    parser.add_argument('--bottleneck_dim', default=256, type=int,
                       help='Dimensionality of bottleneck in projection head')
    parser.add_argument('--norm_last_layer', default=True, type=bool,
                       help='Whether to weight normalize the last layer')
    parser.add_argument('--drop_path_rate', default=None, type=float,
                       help='Stochastic depth rate')
    
    # Tokenizer parameters
    parser.add_argument('--num_tokens', default=8192, type=int,
                       help='Number of discrete tokens for tokenizer')
    parser.add_argument('--tokenizer_hidden_dim', default=512, type=int,
                       help='Hidden dimension for tokenizer MLP')
    
    # Masking parameters
    parser.add_argument('--mask_ratio', default=0.3, type=float,
                       help='Ratio of patches to mask')
    parser.add_argument('--mask_type', default='random', type=str,
                       choices=['random', 'blockwise'],
                       help='Type of masking strategy')
    parser.add_argument('--block_size', default=2, type=int,
                       help='Block size for blockwise masking')
    
    # Loss weights
    parser.add_argument('--mim_loss_weight', default=1.0, type=float,
                       help='Weight for MIM loss')
    parser.add_argument('--cls_loss_weight', default=1.0, type=float,
                       help='Weight for self-distillation (CLS) loss')
    parser.add_argument('--koleo_weight', default=0.0, type=float,
                       help='Weight for KoLeo regularization (feature diversity). Default: 0.0 (disabled - original iBOT doesn\'t use it)')
    parser.add_argument('--koleo_eps', default=1e-6, type=float,
                       help='Epsilon for KoLeo loss (prevents log(0))')
    
    # Temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                       help='Initial teacher temperature')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                       help='Final teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                       help='Number of epochs for teacher temperature warmup')
    parser.add_argument('--student_temp', default=0.1, type=float,
                       help='Student temperature')
    
    # Training parameters
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                       help='Base EMA parameter for teacher update')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Whether to use mixed precision training')
    parser.add_argument('--weight_decay', default=0.04, type=float,
                       help='Initial weight decay')
    parser.add_argument('--weight_decay_end', default=0.1, type=float,
                       help='Final weight decay')
    parser.add_argument('--clip_grad', default=1.0, type=float,
                       help='Maximal parameter gradient norm')
    parser.add_argument('--batch_size', default=128, type=int,
                       help='Per-GPU batch size')
    parser.add_argument('--epochs', default=100, type=int,
                       help='Number of epochs')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                       help='Number of epochs to freeze last layer')
    
    # Augmentation parameters
    parser.add_argument('--local_crops_number', default=6, type=int,
                       help='Number of small local views')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', default='adamw', type=str,
                       choices=['lars', 'adamw'],
                       help='Optimizer type (LARS or AdamW)')
    parser.add_argument('--lr', default=0.0003, type=float,
                       help='Learning rate (base LR for AdamW, typically 0.0003-0.0005)')
    parser.add_argument('--max_lr', default=None, type=float,
                       help='Peak learning rate after warmup (if None, uses base_lr as peak)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                       help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                       help='Number of epochs for learning rate warmup')
    parser.add_argument('--lr_last_layer_scale', default=1.0, type=float,
                       help='Learning rate scale for last layer')
    parser.add_argument('--momentum', default=0.9, type=float,
                       help='Momentum for LARS optimizer')
    parser.add_argument('--lars_trust_coefficient', default=0.001, type=float,
                       help='Trust coefficient for LARS (layer-wise LR scaling)')
    parser.add_argument('--lars_eta', default=0.001, type=float,
                       help='Eta parameter for LARS')
    
    # Misc
    parser.add_argument('--data_path', default='/mnt/user-data/uploads/pretrain/',
                       type=str, help='Path to pretraining data')
    parser.add_argument('--output_dir', default='./checkpoints', type=str,
                       help='Path to save checkpoints')
    parser.add_argument('--save_freq', default=10, type=int,
                       help='Save checkpoint every n epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='Number of data loading workers')
    
    # Resume
    parser.add_argument('--resume', default='', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    main(args)

