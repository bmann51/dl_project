import os
import math
import argparse
import torch
from torch.utils.data import DataLoader

from ibot_ssl_v10 import (
    MultiCropSSLImageDataset,
    generate_block_mask,
    apply_image_mask,
    StudentTeacherIBOT,
    IBOTMultiCropLoss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Path to pretrain/ directory")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=300)  # multi-crop is heavier
    p.add_argument("--batch_size", type=int, default=192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_local_crops", type=int, default=4)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to .pth checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset: 2 global + N local
    dataset = MultiCropSSLImageDataset(
        args.data_root,
        num_global_crops=2,
        num_local_crops=args.num_local_crops,
        global_size=96,
        local_size=64,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    model = StudentTeacherIBOT(
        img_size=96,
        patch_size=16,
        embed_dim=768,
        out_dim=8192,
    )
    model.to(device)

    # Optimizer: student backbone + head
    optim = torch.optim.AdamW(
        model.student_backbone.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    optim.add_param_group({"params": model.head.parameters()})

    # Loss
    ibot_loss = IBOTMultiCropLoss(
        out_dim=8192,
        student_temp_cls=0.1,
        student_temp_patch=0.1,
        teacher_temp_cls=(0.04, 0.07),
        teacher_temp_patch=(0.04, 0.07),
        warmup_teacher_temp_epochs=30,
        nepochs=args.epochs,
    ).to(device)

    # Optionally resume
    start_epoch = 0
    global_step = 0

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"No checkpoint found at {args.resume}")
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        # "epoch" in checkpoint is the next epoch to run
        start_epoch = ckpt.get("epoch", 0)
        print(f" -> start_epoch set to {start_epoch}")

        model.student_backbone.load_state_dict(ckpt["student_backbone"])
        model.teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
        model.head.load_state_dict(ckpt["head"])
        model.teacher_head.load_state_dict(ckpt["teacher_head"])
        optim.load_state_dict(ckpt["optimizer"])
        ibot_loss.load_state_dict(ckpt["ibot_loss"])

        global_step = ckpt.get("global_step", 0)

    def lr_schedule(epoch):
        # Cosine schedule over total epochs; epoch is the current index
        return 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))

    num_global = 2

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_mim = 0.0
        n_batches = 0

        for crops in loader:
            # crops is a list of length (2 + num_local), each [B, C, H, W]
            # First 2 are global, rest are local
            assert isinstance(crops, list)
            assert len(crops) >= num_global

            global_views = [v.to(device, non_blocking=True)
                            for v in crops[:num_global]]
            local_views = [v.to(device, non_blocking=True)
                           for v in crops[num_global:]]

            B = global_views[0].size(0)

            # Masking on GLOBAL student views only
            mask1 = generate_block_mask(
                B, img_size=96, patch_size=16,
                mask_ratio=args.mask_ratio, device=device
            )
            mask2 = generate_block_mask(
                B, img_size=96, patch_size=16,
                mask_ratio=args.mask_ratio, device=device
            )

            g1_student = apply_image_mask(global_views[0], mask1, patch_size=16)
            g2_student = apply_image_mask(global_views[1], mask2, patch_size=16)

            # Student forward: globals (CLS + patch)
            s_cls1, s_patch1 = model.forward_student(g1_student)
            s_cls2, s_patch2 = model.forward_student(g2_student)

            # Student forward: locals (CLS only)
            s_cls_locals = []
            for lv in local_views:
                cls_l, _ = model.forward_student(lv)
                s_cls_locals.append(cls_l)

            # Teacher forward: GLOBAL unmasked views only
            with torch.no_grad():
                t_cls1, t_patch1 = model.forward_teacher(global_views[0])
                t_cls2, t_patch2 = model.forward_teacher(global_views[1])

            student_cls_views = [s_cls1, s_cls2] + s_cls_locals
            teacher_cls_views = [t_cls1, t_cls2]
            student_patch_views = [s_patch1, s_patch2]
            teacher_patch_views = [t_patch1, t_patch2]
            masks = [mask1, mask2]

            loss, loss_dict = ibot_loss(
                student_cls_views=student_cls_views,
                teacher_cls_views=teacher_cls_views,
                student_patch_views=student_patch_views,
                teacher_patch_views=teacher_patch_views,
                masks=masks,
                epoch=epoch,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            # EMA update of teacher (ramp to 0.999)
            m = 0.996 + (1 - 0.996) * (epoch / max(1, args.epochs - 1))
            m = min(m, 0.999)
            model.update_teacher(momentum=m)

            epoch_loss += float(loss.item())
            epoch_cls += loss_dict["loss_cls"]
            epoch_mim += loss_dict["loss_mim"]
            n_batches += 1
            global_step += 1

        # Adjust LR after epoch
        for g in optim.param_groups:
            g["lr"] = args.lr * lr_schedule(epoch)

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={epoch_loss/n_batches:.4f} "
            f"cls={epoch_cls/n_batches:.4f} "
            f"mim={epoch_mim/n_batches:.4f}"
        )

        # Save checkpoint every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            state = {
                # store "next epoch to run"
                "epoch": epoch + 1,
                "student_backbone": model.student_backbone.state_dict(),
                "teacher_backbone": model.teacher_backbone.state_dict(),
                "head": model.head.state_dict(),
                "teacher_head": model.teacher_head.state_dict(),
                "optimizer": optim.state_dict(),
                "ibot_loss": ibot_loss.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            }
            ckpt_path = os.path.join(
                args.output_dir,
                f"ibot_v10_epoch{epoch+1}.pth",
            )
            torch.save(state, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
