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
from dino_ssl import (
    get_backbone, DINOHead, MultiCropWrapper, DINOLoss, 
    DataAugmentation, cosine_scheduler, update_momentum,
    cancel_gradients_last_layer, count_parameters
)


class UnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled pretraining images.
    Supports both local directory and Hugging Face datasets.
    """
    def __init__(self, root_dir, transform=None, use_hf_dataset=False, hf_dataset=None, hf_dataset_name=None):
        self.transform = transform
        self.use_hf_dataset = use_hf_dataset
        self._hf_dataset = None  # Lazy loading for multiprocessing compatibility
        self.hf_dataset_name = hf_dataset_name
        
        if use_hf_dataset:
            if hf_dataset is not None:
                # Store dataset directly (for single process)
                self._hf_dataset = hf_dataset
                self.length = len(hf_dataset)
            elif hf_dataset_name is not None:
                # Store dataset name for lazy loading (multiprocessing compatible)
                self.length = None  # Will be set on first access
            else:
                raise ValueError("Either hf_dataset or hf_dataset_name must be provided")
            print(f"Using Hugging Face dataset: {hf_dataset_name or 'loaded'}")
        else:
            # Use local directory (original behavior)
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
            # Lazy load to get length
            _ = self.hf_dataset
        return self.length
    
    def __getitem__(self, idx):
        try:
            if self.use_hf_dataset:
                # Get image directly from Hugging Face dataset
                example = self.hf_dataset[idx]
                image = example["image"]
                
                # Handle different image formats from Hugging Face
                if isinstance(image, Image.Image):
                    # Already a PIL Image - most common case
                    pass
                elif hasattr(image, '__array__'):  # NumPy array or similar
                    image = Image.fromarray(np.asarray(image))
                else:
                    # Try to convert to PIL Image
                    try:
                        image = Image.fromarray(image)
                    except Exception as e:
                        # If conversion fails, try opening as bytes
                        if hasattr(image, 'tobytes'):
                            image = Image.frombytes(image.mode, image.size, image.tobytes())
                        else:
                            raise ValueError(f"Unable to convert image to PIL Image: {type(image)}, error: {e}")
                
                # Ensure RGB mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                # Load from local file system
                img_path = self.image_paths[idx]
                image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0  # Return dummy label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            print(f"Image type: {type(image) if 'image' in locals() else 'N/A'}")
            raise


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, 
                   data_loader, optimizer, lr_schedule, wd_schedule, 
                   momentum_schedule, epoch, fp16_scaler, args):
    """
    Train for one epoch
    """
    # Accumulate metrics for averaging
    loss_sum = 0.0
    loss_count = 0
    all_losses = []  # Store per-iteration losses
    
    # Create progress bar with initial description
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for it, (images, _) in enumerate(pbar):
        # Update learning rate and weight decay
        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it_global]
        
        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]
        
        # Teacher and student forward passes
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # Only global views for teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)
        
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)
        
        # Backward pass
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            # Gradient clipping
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
        
        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]
            update_momentum(student, teacher_without_ddp, m)
        
        # Logging per iteration
        torch.cuda.synchronize()
        loss_value = loss.item()
        loss_sum += loss_value
        loss_count += 1
        all_losses.append(loss_value)
        
        # Update progress bar with current and average loss
        avg_loss = loss_sum / loss_count
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    # Return metrics
    metric_logger = {
        'loss': avg_loss,
        'lr': optimizer.param_groups[0]["lr"],
        'wd': optimizer.param_groups[0]["weight_decay"],
        'all_losses': all_losses  # Include per-iteration losses
    }
    
    return metric_logger


def main(args):
    print("=" * 80)
    print("DINO Self-Supervised Learning")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation - use stronger augmentation to reduce overfitting
    transform = DataAugmentation(
        global_crops_scale=(0.32, 1.0),  # Wider scale range for more diversity
        local_crops_scale=(0.05, 0.32),  # Adjusted to match
        local_crops_number=args.local_crops_number,
        size=args.image_size
    )
    
    # Dataset and DataLoader
    # Support Hugging Face datasets if data_path starts with "hf://"
    use_hf_dataset = args.data_path.startswith("hf://")
    hf_dataset = None
    
    if use_hf_dataset:
        from datasets import load_dataset
        hf_dataset_name = args.data_path.replace("hf://", "")  # Remove "hf://" prefix
        print(f"Loading Hugging Face dataset: {hf_dataset_name}")
        # Load dataset to get length, but store name for multiprocessing compatibility
        hf_dataset = load_dataset(hf_dataset_name, split="train")
        dataset = UnlabeledImageDataset(
            args.data_path, 
            transform=transform, 
            use_hf_dataset=True, 
            hf_dataset=hf_dataset,  # Pre-load for main process
            hf_dataset_name=hf_dataset_name  # Store name for worker processes
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
    # Set default drop_path_rate based on architecture if not specified
    # Increased rates to reduce overfitting
    if args.drop_path_rate is None:
        if 'tiny' in args.arch:
            drop_path_rate = 0.15  # Increased from 0.1
        elif 'small' in args.arch:
            drop_path_rate = 0.2  # Increased from 0.1
        elif 'base' in args.arch:
            drop_path_rate = 0.25  # Increased from 0.15
        else:
            drop_path_rate = 0.0  # ResNet doesn't use drop_path
    else:
        drop_path_rate = args.drop_path_rate
    
    print(f"Using drop_path_rate: {drop_path_rate}")
    
    student_backbone, embed_dim = get_backbone(args.arch, img_size=args.image_size, 
                                                drop_path_rate=drop_path_rate)
    teacher_backbone, _ = get_backbone(args.arch, img_size=args.image_size, 
                                       drop_path_rate=drop_path_rate)
    
    # Create heads: global head and local head (untied)
    student_global_head = DINOHead(
        embed_dim,
        args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
        norm_last_layer=args.norm_last_layer,
    )
    student_local_head = DINOHead(
        embed_dim,
        args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
        norm_last_layer=False,  # Local head doesn't need norm_last_layer
    )
    
    # Teacher only uses global head (only sees global crops)
    teacher_global_head = DINOHead(
        embed_dim,
        args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
    )
    
    # Student uses untied heads (global + local)
    student = MultiCropWrapper(student_backbone, student_global_head, 
                               local_head=student_local_head)
    # Teacher only needs global head (only processes global crops)
    teacher = MultiCropWrapper(teacher_backbone, teacher_global_head, 
                               local_head=None)
    
    # Move to GPU
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Teacher and student start with the same weights
    # Filter out local_head weights since teacher doesn't have one
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    # Only load weights that exist in both
    filtered_state = {k: v for k, v in student_state.items() 
                     if k in teacher_state and teacher_state[k].shape == v.shape}
    teacher.load_state_dict(filtered_state, strict=False)
    # No gradients for teacher
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"Student parameters: {count_parameters(student):,}")
    print(f"Teacher parameters: {count_parameters(teacher):,}")
    
    if count_parameters(student) >= 100_000_000:
        print(f"WARNING: Model has {count_parameters(student):,} parameters (limit is 100M)")
    
    # Loss
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        student_temp=args.student_temp,
    ).to(device)
    
    # Optimizer
    params_groups = [
        {'params': [p for n, p in student.named_parameters() if 'last_layer' not in n]},
        {'params': [p for n, p in student.named_parameters() if 'last_layer' in n], 
         'weight_decay': 0.0, 'lr': args.lr * args.lr_last_layer_scale}
    ]
    optimizer = torch.optim.AdamW(params_groups)
    
    # Learning rate schedule
    lr_schedule = cosine_scheduler(
        args.lr * args.batch_size / 256.,  # Linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    # Momentum parameter schedule for teacher (DINOv2 uses cosine)
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1.0,
        args.epochs, len(data_loader)
    )
    
    print(f"\nLoss, optimizer and schedulers ready.")
    
    # Mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    # Setup loss logging CSV file
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
    
    for epoch in range(start_epoch, args.epochs):  # Change this line
        # Train one epoch
        train_stats = train_one_epoch(
            student, teacher, teacher,
            dino_loss, data_loader,
            optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch,
            fp16_scaler, args
        )
        
        # Log per-iteration losses to CSV
        if 'all_losses' in train_stats:
            for it, loss_val in enumerate(train_stats['all_losses']):
                it_global = len(data_loader) * epoch + it
                lr_val = lr_schedule[it_global] if it_global < len(lr_schedule) else lr_schedule[-1]
                loss_log_writer.writerow([epoch, it, f'{loss_val:.6f}', f'{lr_val:.8e}'])
            loss_log_file.flush()  # Ensure data is written
        
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
        
        # Print statistics (exclude all_losses from print to avoid clutter)
        stats_to_print = {k: v for k, v in train_stats.items() if k != 'all_losses'}
        print(f"Epoch {epoch} stats: {stats_to_print}")
    
    # Close loss log file
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
    parser = argparse.ArgumentParser('DINO', add_help=False)
    
    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                       choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'],
                       help='Architecture')
    parser.add_argument('--image_size', default=96, type=int, help='Image size')
    parser.add_argument('--out_dim', default=8192, type=int,
                       help='Dimensionality of the DINO head output')
    parser.add_argument('--bottleneck_dim', default=256, type=int,
                       help='Dimensionality of bottleneck in projection head')
    parser.add_argument('--norm_last_layer', default=True, type=bool,
                       help='Whether to weight normalize the last layer')
    parser.add_argument('--drop_path_rate', default=None, type=float,
                       help='Stochastic depth rate (drop path). Default: 0.1 for tiny/small, 0.15 for base')
    
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
                       help='Whether to use mixed precision training (default: True)')
    parser.add_argument('--weight_decay', default=0.04, type=float,
                       help='Initial weight decay')
    parser.add_argument('--weight_decay_end', default=0.4, type=float,
                       help='Final weight decay')
    parser.add_argument('--clip_grad', default=3.0, type=float,
                       help='Maximal parameter gradient norm')
    parser.add_argument('--batch_size', default=64, type=int,
                       help='Per-GPU batch size')
    parser.add_argument('--epochs', default=100, type=int,
                       help='Number of epochs')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                       help='Number of epochs to freeze last layer')
    
    # Augmentation parameters
    parser.add_argument('--local_crops_number', default=6, type=int,
                       help='Number of small local views')
    
    # Optimizer parameters
    parser.add_argument('--lr', default=0.0005, type=float,
                       help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                       help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                       help='Number of epochs for learning rate warmup')
    parser.add_argument('--lr_last_layer_scale', default=1.0, type=float,
                       help='Learning rate scale for last layer')
    
    # Misc
    parser.add_argument('--data_path', default='/mnt/user-data/uploads/pretrain/',
                       type=str, help='Path to pretraining data')
    parser.add_argument('--output_dir', default='./checkpoints', type=str,
                       help='Path to save checkpoints')
    parser.add_argument('--save_freq', default=10, type=int,
                       help='Save checkpoint every n epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                       help='Number of data loading workers')
    
    #Resume
    parser.add_argument('--resume', default='', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    main(args)
