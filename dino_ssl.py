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
    Wrapper to handle multiple crops (global + local views)
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        # If input is a list of crops, concatenate them
        if isinstance(x, list):
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True)[1], 0)
            start_idx = 0
            output = []
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx:end_idx]))
                _out = self.head(_out)
                output.append(_out)
                start_idx = end_idx
            return torch.cat(output)
        else:
            _out = self.backbone(x)
            _out = self.head(_out)
            return _out


class DINOLoss(nn.Module):
    """
    DINO loss with teacher centering and sharpening
    """
    def __init__(self, out_dim, ncrops=8, warmup_teacher_temp=0.04,
                 teacher_temp=0.04, warmup_teacher_temp_epochs=30,
                 nepochs=100, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Temperature schedule
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global views for teacher
        
        total_loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip cases where student and teacher operate on same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output (DINOv2 uses exponential moving average)
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentation:
    """
    Multi-crop augmentation strategy from DINO
    """
    def __init__(self, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4),
                 local_crops_number=6, size=96):
        # Global crops (2)
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


def get_backbone(arch='vit_small', patch_size=16, img_size=96):
    """
    Create a backbone network. ViT recommended for DINO.
    For 96x96 images, we use smaller models to stay under 100M params.
    """
    if 'vit' in arch:
        # Using timm for ViT models
        if arch == 'vit_tiny':
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False, 
                                     img_size=img_size, num_classes=0)  # num_classes=0 removes head
        elif arch == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=False,
                                     img_size=img_size, num_classes=0)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        embed_dim = model.embed_dim
    else:
        # ResNet backbone alternative
        model = timm.create_model('resnet50', pretrained=False, num_classes=0)
        embed_dim = 2048
    
    return model, embed_dim


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=10,
                    start_warmup_value=0):
    """
    Cosine learning rate schedule with warmup
    """
    warmup_schedule = torch.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    
    iters = torch.arange(epochs * niter_per_ep - warmup_epochs * niter_per_ep)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + torch.cos(math.pi * iters / len(iters)))
    
    schedule = torch.cat((warmup_schedule, schedule))
    return schedule


@torch.no_grad()
def update_momentum(student, teacher, m):
    """
    Momentum update of the teacher network (DINOv2 uses cosine schedule for momentum)
    """
    for param_q, param_k in zip(student.parameters(), teacher.parameters()):
        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """
    Cancel gradients for the last layer during initial training (DINO trick)
    """
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


# Model parameter counting utility
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing DINO components...")
    
    # Test backbone
    backbone, embed_dim = get_backbone('vit_small', img_size=96)
    print(f"Backbone embed_dim: {embed_dim}")
    print(f"Backbone parameters: {count_parameters(backbone):,}")
    
    # Test head
    head = DINOHead(embed_dim, out_dim=8192, bottleneck_dim=256)
    print(f"Head parameters: {count_parameters(head):,}")
    
    # Test full model
    student = MultiCropWrapper(backbone, head)
    print(f"Total student parameters: {count_parameters(student):,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 96, 96)
    features = backbone(x)
    output = head(features)
    print(f"Output shape: {output.shape}")
    
    print("\nAll tests passed!")