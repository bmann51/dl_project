import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from dino_ssl import (
    get_backbone, DINOHead, MultiCropWrapper, DINOLoss, 
    DataAugmentation, cosine_scheduler, update_momentum,
    cancel_gradients_last_layer, count_parameters
)


class UnlabeledImageDataset(Dataset):
    """
    Dataset for unlabeled pretraining images
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_paths.extend(self.root_dir.rglob(ext))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, 
                   data_loader, optimizer, lr_schedule, wd_schedule, 
                   momentum_schedule, epoch, fp16_scaler, args):
    """
    Train for one epoch
    """
    metric_logger = {}
    
    for it, (images, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
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
        
        # Logging
        torch.cuda.synchronize()
        metric_logger['loss'] = loss.item()
        metric_logger['lr'] = optimizer.param_groups[0]["lr"]
        metric_logger['wd'] = optimizer.param_groups[0]["weight_decay"]
    
    return {k: v for k, v in metric_logger.items()}


def main(args):
    print("=" * 80)
    print("DINO Self-Supervised Learning")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation
    transform = DataAugmentation(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=args.local_crops_number,
        size=args.image_size
    )
    
    # Dataset and DataLoader
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
    student_backbone, embed_dim = get_backbone(args.arch, img_size=args.image_size)
    teacher_backbone, _ = get_backbone(args.arch, img_size=args.image_size)
    
    student_head = DINOHead(
        embed_dim,
        args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
        norm_last_layer=args.norm_last_layer,
    )
    teacher_head = DINOHead(
        embed_dim,
        args.out_dim,
        bottleneck_dim=args.bottleneck_dim,
    )
    
    student = MultiCropWrapper(student_backbone, student_head)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head)
    
    # Move to GPU
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
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
        
        # Print statistics
        print(f"Epoch {epoch} stats: {train_stats}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_checkpoint.pth')
    torch.save({
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'args': args,
    }, final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")


if __name__ == '__main__':
    import sys
    import math
    
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
    parser.add_argument('--use_fp16', default=True, type=bool,
                       help='Whether to use mixed precision training')
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