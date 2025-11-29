import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm
import numpy as np


# ======================
#   Dataset & Augmentations
# ======================

class SSLImageDataset(Dataset):
    """
    Unlabeled SSL dataset.
    Returns two augmented views: (img1, img2).
    """
    def __init__(self, root):
        self.root = Path(root)
        self.paths = sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")

        # Strong SimCLR / DINO-style augmentations, output size 96x96
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        im1 = self.transform(img)
        im2 = self.transform(img)
        return im1, im2


# ======================
#   Masking (image-space, blockwise)
# ======================

def generate_block_mask(batch_size, img_size=96, patch_size=16, mask_ratio=0.3, device="cpu"):
    """
    Returns:
      mask: [B, N] with 1 for masked patches, 0 for visible.
    """
    num_patches_per_dim = img_size // patch_size  # e.g. 96/16 = 6
    N = num_patches_per_dim ** 2
    num_mask = int(mask_ratio * N)
    mask = torch.zeros(batch_size, N, device=device)
    for b in range(batch_size):
        idx = torch.randperm(N, device=device)[:num_mask]
        mask[b, idx] = 1.0
    return mask  # [B, N]

def apply_image_mask(x, mask, patch_size=16):
    """
    x: [B, 3, H, W]
    mask: [B, N] with 1=masked
    Zero out masked patches in image space.
    """
    B, C, H, W = x.shape
    assert H == W
    num_patches_per_dim = H // patch_size
    N = num_patches_per_dim ** 2
    assert mask.shape[1] == N

    x = x.clone()
    mask_flat = mask.view(B, num_patches_per_dim, num_patches_per_dim)  # [B, Gh, Gw]
    for b in range(B):
        for i in range(num_patches_per_dim):
            for j in range(num_patches_per_dim):
                if mask_flat[b, i, j] == 1:
                    h0 = i * patch_size
                    w0 = j * patch_size
                    x[b, :, h0:h0+patch_size, w0:w0+patch_size] = 0.0
    return x


# ======================
#   ViT backbone + projection head
# ======================

class ViTBackbone(nn.Module):
    """
    Wrap timm ViT-Base/16 to expose CLS and patch tokens.
    """
    def __init__(self, img_size=96, patch_size=16, embed_dim=768):
        super().__init__()
        # vit_base_patch16_224 is standard; we override img_size & num_classes
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,  # no classifier
            pretrained=False,
        )
        assert self.vit.embed_dim == embed_dim

    def forward(self, x):
        """
        Return:
          cls:   [B, D]
          patch: [B, N, D]
        """
        # timm ViT forward_features returns [B, 1+N, D]
        tokens = self.vit.forward_features(x)  # [B, 1+N, D]
        cls = tokens[:, 0]          # [B, D]
        patch = tokens[:, 1:]       # [B, N, D]
        return cls, patch

class ProjectionHead(nn.Module):
    """
    Shared projection head for CLS and patch tokens.
    3-layer MLP with l2-normalized bottleneck, output dim K.
    """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=8192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        # l2 normalization
        x = F.normalize(x, dim=-1)
        return x

class StudentTeacherIBOT(nn.Module):
    """
    Student-Teacher wrapper:
      - Both share same backbone architecture and projection head architecture.
      - Teacher parameters are EMA of student.
    """
    def __init__(self, img_size=96, patch_size=16, embed_dim=768, out_dim=8192):
        super().__init__()
        self.student_backbone = ViTBackbone(img_size, patch_size, embed_dim)
        self.teacher_backbone = ViTBackbone(img_size, patch_size, embed_dim)
        self.head = ProjectionHead(embed_dim, hidden_dim=2048, out_dim=out_dim)
        # Teacher uses same head, but we do not backprop through it directly (EMA updates).
        self.teacher_head = ProjectionHead(embed_dim, hidden_dim=2048, out_dim=out_dim)
        # Initialize teacher with student weights
        self._init_teacher()

    @torch.no_grad()
    def _init_teacher(self):
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False
        for ps, pt in zip(self.head.parameters(), self.teacher_head.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        """
        EMA update of teacher params from student params.
        """
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(momentum).add_(ps.data * (1.0 - momentum))
        for ps, pt in zip(self.head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data * (1.0 - momentum))

    def forward_student(self, x):
        cls, patch = self.student_backbone(x)
        cls_proj = self.head(cls)
        patch_proj = self.head(patch)  # [B, N, K]
        return cls_proj, patch_proj

    @torch.no_grad()
    def forward_teacher(self, x):
        cls, patch = self.teacher_backbone(x)
        cls_proj = self.teacher_head(cls)
        patch_proj = self.teacher_head(patch)
        return cls_proj, patch_proj


# ======================
#   iBOT Loss (CLS + MIM)
# ======================

class IBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim=8192,
        student_temp_cls=0.1,
        student_temp_patch=0.1,
        teacher_temp_cls=(0.04, 0.07),
        teacher_temp_patch=(0.04, 0.07),
        warmup_teacher_temp_epochs=30,
        nepochs=400,
        center_momentum_cls=0.9,
        center_momentum_patch=0.9,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp_cls = student_temp_cls
        self.student_temp_patch = student_temp_patch
        self.teacher_temp_cls_start, self.teacher_temp_cls_end = teacher_temp_cls
        self.teacher_temp_patch_start, self.teacher_temp_patch_end = teacher_temp_patch
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.nepochs = nepochs

        # Centers
        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, out_dim))
        self.center_momentum_cls = center_momentum_cls
        self.center_momentum_patch = center_momentum_patch

    def _teacher_temp(self, epoch, kind="cls"):
        if kind == "cls":
            start, end = self.teacher_temp_cls_start, self.teacher_temp_cls_end
        else:
            start, end = self.teacher_temp_patch_start, self.teacher_temp_patch_end
        if epoch < self.warmup_teacher_temp_epochs:
            return start + (end - start) * epoch / max(1, self.warmup_teacher_temp_epochs - 1)
        else:
            return end

    @torch.no_grad()
    def _update_center_cls(self, teacher_cls_all):
        """
        teacher_cls_all: [B_total, K]
        """
        batch_center = teacher_cls_all.mean(dim=0, keepdim=True)
        self.center_cls = self.center_cls * self.center_momentum_cls + batch_center * (1.0 - self.center_momentum_cls)

    @torch.no_grad()
    def _update_center_patch(self, teacher_patch_all):
        """
        teacher_patch_all: [B_total * N_masked, K]
        """
        batch_center = teacher_patch_all.mean(dim=0, keepdim=True)
        self.center_patch = self.center_patch * self.center_momentum_patch + batch_center * (1.0 - self.center_momentum_patch)

    def forward(
        self,
        student_cls_views,   # list of [B, K], masked views (two global crops)
        teacher_cls_views,   # list of [B, K], unmasked views
        student_patch_views, # list of [B, N, K], masked
        teacher_patch_views, # list of [B, N, K], unmasked
        masks,               # list of [B, N] {0,1}, for student views
        epoch,
    ):
        """
        Two global views: index 0 and 1.
        We do cross-view CLS distillation and in-view MIM.
        """
        # --- CLS loss (DINO-style cross-view) ---
        t_temp_cls = self._teacher_temp(epoch, kind="cls")
        teacher_cls_all = []
        loss_cls = 0.0
        n_loss_terms = 0

        # cross-view: (student v0, teacher v1) and (student v1, teacher v0)
        for i_s, i_t in [(0, 1), (1, 0)]:
            s = student_cls_views[i_s]  # [B, K]
            t = teacher_cls_views[i_t].detach()  # [B, K]

            # teacher: center + temp + softmax
            t = F.softmax((t - self.center_cls) / t_temp_cls, dim=-1)  # [B, K]

            # student: temp + log_softmax
            s = F.log_softmax(s / self.student_temp_cls, dim=-1)       # [B, K]

            ce = -(t * s).sum(dim=-1).mean()
            loss_cls += ce
            n_loss_terms += 1
            teacher_cls_all.append(t)  # for center update

        loss_cls = loss_cls / max(1, n_loss_terms)

        # --- MIM loss (patch-level, in-view) ---
        t_temp_patch = self._teacher_temp(epoch, kind="patch")
        loss_mim = 0.0
        n_mim_terms = 0
        teacher_patch_tokens_all = []

        for idx in range(len(student_patch_views)):
            sp = student_patch_views[idx]      # [B, N, K]
            tp = teacher_patch_views[idx].detach()  # [B, N, K]
            mask = masks[idx]                  # [B, N]
            B, N, K = sp.shape
            assert mask.shape == (B, N)

            # teacher
            tp = (tp - self.center_patch) / t_temp_patch
            tp = F.softmax(tp, dim=-1)  # [B, N, K]

            # student
            sp = F.log_softmax(sp / self.student_temp_patch, dim=-1)  # [B, N, K]

            ce = -(tp * sp).sum(dim=-1)  # [B, N]

            # average over masked tokens per image
            masked = mask > 0
            num_masked = masked.sum(dim=1)  # [B]

            # Avoid divide by zero: if image has no masked patches (shouldn't happen if ratio>0) skip
            valid = num_masked > 0
            if valid.any():
                ce_valid = (ce * mask).sum(dim=1)[valid] / num_masked[valid]
                loss_mim += ce_valid.mean()
                n_mim_terms += 1

            # accumulate teacher patch tokens on masked locations for center update
            teacher_patch_tokens_all.append(tp[masked])  # [num_masked_total, K_total]

        if n_mim_terms > 0:
            loss_mim = loss_mim / n_mim_terms
        else:
            loss_mim = torch.tensor(0.0, device=student_cls_views[0].device)

        # --- Update centers ---
        with torch.no_grad():
            if teacher_cls_all:
                teacher_cls_cat = torch.cat(teacher_cls_all, dim=0)  # [B_total, K]
                self._update_center_cls(teacher_cls_cat)
            if teacher_patch_tokens_all:
                teacher_patch_cat = torch.cat(teacher_patch_tokens_all, dim=0)
                self._update_center_patch(teacher_patch_cat)

        # Total loss (no scaling between CLS and MIM, as in paper)
        loss = loss_cls + loss_mim
        return loss, {"loss_cls": loss_cls.item(), "loss_mim": loss_mim.item()}

