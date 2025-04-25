import os
import datetime
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.amp import GradScaler, autocast
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.utils import save_image

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Like ImageFolder, but __getitem__ returns (img, label, path)."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _   = self.samples[index]
        return img, label, path
###########################################
# 1) Helper: Replicate subfolder and save image
###########################################
def replicate_subfolder_and_save(img_tensor: torch.Tensor,
                                 original_path: str,
                                 source_root: str,
                                 target_root: str,
                                 epoch: int = None):
    rel_path = os.path.relpath(original_path, start=source_root)
    subdir   = os.path.dirname(rel_path)  # e.g. "cat" or "dog"
    base_name = os.path.basename(original_path)
    name_no_ext, ext = os.path.splitext(base_name)

    # build filename as before…
    parts = [name_no_ext]
    if epoch is not None:
        parts.append(f"epoch{epoch}")
    parts.append(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    new_filename = "_".join(parts) + ext

    # **here** we now include the epoch level in the directory
    out_subdir = os.path.join(target_root, f"epoch{epoch}", subdir)
    os.makedirs(out_subdir, exist_ok=True)
    final_path = os.path.join(out_subdir, new_filename)

    save_image(img_tensor.clamp(0, 1), final_path)
    return final_path


###########################################
# 2) Load frozen Classifier 1 (EfficientNet-B6 pretrained on clean images)
###########################################
def load_frozen_classifier1(path, device, num_classes=3):
    model = efficientnet_b6(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    state = torch.load(path, map_location=device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model

###########################################
# 3) Define UNet+MaxViT Watermark Generator
###########################################
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.seq(x)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x_conv = self.conv(x)          # same spatial size, channels = out_c
        x_down = self.pool(x_conv)     # spatial downsample by factor 2
        return x_conv, x_down

class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_c + skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)  # upsample (spatial x2)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

# --- Attention blocks for MaxViT ---
class SimpleAttention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = heads
        self.head_dim = dim // heads
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).reshape(B, N, C)
        return self.proj(out)

def partition_windows(x, ws=8):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * (H // ws) * (W // ws), ws, ws, C)

def unpartition_windows(x, ws, B, H, W):
    n = x.shape[0]
    bHW = (H // ws) * (W // ws)
    b = n // bHW
    x = x.view(b, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, H, W, -1)

def block_attn(x, attn, ws=8):
    B, H, W, C = x.shape
    blocks = partition_windows(x, ws)
    blocks = blocks.view(-1, ws * ws, C)
    out = attn(blocks)
    out = out.view(-1, ws, ws, C)
    out = unpartition_windows(out, ws, B, H, W)
    return out

class SimpleMaxViTBlock(nn.Module):
    def __init__(self, dim=512, heads=8, ws=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.ws = ws
    def forward(self, x):
        B, H, W, C = x.shape
        # Block attention
        x_ = x.view(B, H * W, C)
        x_ = self.ln1(x_)
        x_ = x_.view(B, H, W, C)
        x_ = block_attn(x_, self.attn, self.ws)
        x = x + x_
        # Feed-forward
        x_ = x.view(B, H * W, C)
        x_ = self.ln2(x_)
        x_ = self.ffn(x_)
        x_ = x_.view(B, H, W, C)
        return x + x_

class UNetWithRealMaxViT(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, mid_dim=512):
        super().__init__()
        # Down path: 4 blocks
        self.down1 = DownBlock(in_channels, base_dim)          # 3->64; 512->256
        self.down2 = DownBlock(base_dim, base_dim * 2)           # 64->128; 256->128
        self.down3 = DownBlock(base_dim * 2, base_dim * 4)         # 128->256; 128->64
        self.down4 = DownBlock(base_dim * 4, mid_dim)            # 256->512; 64->32

        # Bottleneck: MaxViT block on (B,512,32,32)
        self.maxvit = SimpleMaxViTBlock(dim=mid_dim, heads=8, ws=8)

        # To match channel dimensions in UpBlock1, reduce skip from d4 from 512 to base_dim*4 (i.e. 256)
        self.reduce_d4 = nn.Conv2d(mid_dim, base_dim * 4, kernel_size=1)

        # Up path: 4 blocks
        self.up1 = UpBlock(in_c=mid_dim, skip_c=base_dim * 4, out_c=base_dim * 4)   # (512->256)
        self.up2 = UpBlock(in_c=base_dim * 4, skip_c=base_dim * 4, out_c=base_dim * 2)  # (256->128)
        self.up3 = UpBlock(in_c=base_dim * 2, skip_c=base_dim * 2, out_c=base_dim)      # (128->64)
        self.up4 = UpBlock(in_c=base_dim, skip_c=base_dim, out_c=base_dim)              # (64->64)

        self.final_conv = nn.Conv2d(base_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # Down path
        d1, d1p = self.down1(x)
        d2, d2p = self.down2(d1p)
        d3, d3p = self.down3(d2p)
        d4, d4p = self.down4(d3p)

        # Bottleneck: apply MaxViT block
        B, C, H, W = d4p.shape
        x_b = d4p.permute(0, 2, 3, 1)
        x_b = self.maxvit(x_b)
        x_b = x_b.permute(0, 3, 1, 2)

        # Up path
        d4_reduced = self.reduce_d4(d4)
        u1 = self.up1(x_b, d4_reduced)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        # Instead of generating a full image, generate a residual watermark
        out = x + self.final_conv(u4)

        return out

###########################################
# 4) Define Classifier 2 (EfficientNet-B6 based)
###########################################
def build_classifier2(num_classes=3):
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

###########################################
# 5) Stage 1: Joint (Concurrent) Training with AMP for Memory Optimization
###########################################
###########################################
# 5) Stage 1: Joint (Concurrent) Training
###########################################
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Subclass ImageFolder so __getitem__ returns (img, label, path)

def stage1_joint_train(data_dir: str,
                       classifier1_path: str,
                       partial_save_root: str,
                       joint_out_dir: str,
                       epochs=1,
                       batch_size=4,
                       base_lr=1e-4):
    """
    Stage 1:
      - Jointly train the watermark generator (UNet+MaxViT) and classifier 2.
      - Saves every watermarked image this epoch into:
         partial_save_root/epoch{epoch}/{class_name}/<original_name>_epoch{epoch}_{timestamp}.ext
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    c1    = load_frozen_classifier1(classifier1_path, device, num_classes=3)
    wm_gen = UNetWithRealMaxViT().to(device)
    c2     = build_classifier2(num_classes=3).to(device)
    if torch.cuda.device_count() > 1:
        wm_gen = nn.DataParallel(wm_gen)
        c2     = nn.DataParallel(c2)

    # -- DATASET & LOADER --
    transform = T.Compose([T.Resize((512,512)), T.ToTensor()])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(list(wm_gen.parameters()) + list(c2.parameters()), lr=base_lr)
    ce_loss   = nn.CrossEntropyLoss()
    scaler    = GradScaler()

    # hyperparams
    lower_ratio        = 0.10
    upper_ratio        = 0.25
    target_c2_clean    = 1.10
    alpha1, alpha2, gamma = 1.00, 0.75, 0.0
    import time
    start_epoch = time.time()

    for epoch in range(1, epochs+1):
        wm_gen.train(); c2.train()
        running = dict(loss=0, c1_clean=0, c1_wm=0, c2_wm=0, c2_clean=0)
        for batch_idx, (imgs, labels, paths) in enumerate(loader):
            t0 = time.time()
            imgs   = imgs.to(device)
            labels = labels.to(device)
            load_time = time.time() - t0

            optimizer.zero_grad()

            t1 = time.time()


            with autocast(device_type='cuda'):
                wm_imgs = wm_gen(imgs)

                # classifier1 losses (frozen)
                with torch.no_grad():
                    out1_clean = c1(imgs)
                l1_clean = ce_loss(out1_clean, labels)
                l1_wm    = ce_loss(c1(wm_imgs), labels)
                diff1    = l1_wm - l1_clean
                penalty1 = alpha1 * (F.relu(lower_ratio*l1_clean - diff1) +
                                     F.relu(diff1 - upper_ratio*l1_clean))

                # classifier2 losses
                out2_wm    = c2(wm_imgs)
                l2_wm      = ce_loss(out2_wm, labels)
                l2_clean   = ce_loss(c2(imgs), labels)
                penalty2   = alpha2 * F.relu(target_c2_clean - l2_clean)

                # subtlety loss
                mse_diff = F.mse_loss(wm_imgs, imgs)
                total_loss = l2_wm + penalty1 + penalty2 + gamma * mse_diff


                # … compute wm_imgs, losses, backward, optimizer.step() …
            compute_time = time.time() - t1



            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 1 == 0:
                    print(f"[Epoch {epoch} Batch {batch_idx+1}] "
                          f"load={load_time:.3f}s, compute={compute_time:.3f}s")

            # accumulate for logging
            running['loss']     += total_loss.item()
            running['c1_clean'] += l1_clean.item()
            running['c1_wm']    += l1_wm.item()
            running['c2_wm']    += l2_wm.item()
            running['c2_clean'] += l2_clean.item()

            # save **all** watermarked images this batch
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for wm_i, orig_path in zip(wm_imgs.detach().cpu(), paths):
                replicate_subfolder_and_save(
                    wm_i, original_path=orig_path,
                    source_root=data_dir,
                    target_root=partial_save_root,
                    epoch=epoch
                )

        # epoch log
        batches = len(loader)
        print(f"Epoch {epoch}: Total Loss={running['loss']/batches:.4f}, "
              f"C1 Clean={running['c1_clean']/batches:.4f}, C1 WM={running['c1_wm']/batches:.4f}, "
              f"C2 WM={running['c2_wm']/batches:.4f}, C2 Clean={running['c2_clean']/batches:.4f}")

    # save final models
    out_dir = os.path.join(joint_out_dir, "stage1_models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(wm_gen.state_dict(), os.path.join(out_dir, "wm_gen_final.pth"))
    torch.save(c2.state_dict(),   os.path.join(out_dir, "c2_temp_final.pth"))
    print("[INFO] Stage1 complete.")
    return wm_gen

###########################################
# 6) Stage 2: Generate full watermarked dataset
###########################################
def stage2_generate_final_dataset(wm_gen, data_dir: str, final_wm_root: str):
    device = next(wm_gen.parameters()).device
    wm_gen.eval()
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor()
    ])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, (img, _, orig_path) in enumerate(loader):
        img = img.to(device)
        with torch.no_grad():
            wm = wm_gen(img)
        wm_cpu = wm[0].detach().cpu()
        orig_path, _ = dataset.samples[i]
        replicate_subfolder_and_save(wm_cpu, orig_path, source_root=data_dir,
                                     target_root=final_wm_root, epoch=None)
    print(f"[INFO] Full watermarked dataset saved at: {final_wm_root}")

###########################################
# 7) Stage 3: Train final Classifier 2 on full watermarked dataset
###########################################
def stage3_train_classifier2(watermarked_dir: str,
                             out_model_path: str,
                             epochs=1,
                             batch_size=4,
                             lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        # 1) Randomly crop & resize to 512×512
        transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
        # 2) Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.55),
        # 3) Random vertical flip
        transforms.RandomVerticalFlip(p=0.35),
        # 4) Random rotation up to ±30°
        transforms.RandomRotation(degrees=(0, 30)),
        # 5) Color jitter (brightness, contrast, saturation, hue)
        transforms.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1)
        ),
        # 6) Randomly convert to grayscale
        transforms.RandomGrayscale(p=0.35),
        # 7) Occasionally blur
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        # 8) To tensor + normalize to [–1, +1]
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # 9) Randomly erase a patch
        transforms.RandomErasing(p=0.25),
    ])
    dataset = ImageFolder(watermarked_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    c2 = build_classifier2(num_classes=len(dataset.classes))
    if torch.cuda.device_count() > 1:
        c2 = nn.DataParallel(c2)
    c2 = c2.to(device)
    optimizer = optim.Adam(c2.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        c2.train()
        total_loss, correct = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = c2(imgs)
            loss = ce_loss(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
        epoch_loss = total_loss / len(dataset)
        epoch_acc = correct / len(dataset)
        print(f"[Final C2] Epoch {epoch}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    torch.save(c2.state_dict(), out_model_path)
    print(f"[INFO] Final classifier 2 saved at {out_model_path}")

###########################################
# 8) MAIN SCRIPT
###########################################
def main():
    # paths (adjust as needed)
    classifier1_path   = r"E:\watermarking\afhq\trained_model_class1\efficientnet_b6_classifier.pth"
    data_dir           = r"E:\watermarking\afhq\tiny_train"
    partial_save_root  = r"E:\watermarking\afhq\epoch_images"             # now saves *all* wm images per epoch
    joint_out_dir      = r"E:\watermarking\afhq\joint_training_outputs"
    final_wm_root      = r"E:\watermarking\afhq\full_final_watermarked"
    final_c2_model     = r"E:\watermarking\afhq\class2_model\imagenetb6_class2.pth"

    print("[INFO] Stage 1: Joint training (saving all per‑epoch watermarks)…")
    wm_gen = stage1_joint_train(
        data_dir,
        classifier1_path,
        partial_save_root,
        joint_out_dir,
        epochs=1,
        batch_size=4,
        base_lr=1e-4
    )

    print("[INFO] Stage 2: Generating full watermarked dataset…")
    stage2_generate_final_dataset(
        wm_gen,
        data_dir,
        final_wm_root
    )

    print("[INFO] Stage 3: Training final Classifier 2 on watermarked data…")
    stage3_train_classifier2(
        watermarked_dir=final_wm_root,
        out_model_path=final_c2_model,
        epochs=1,
        batch_size=4,
        lr=1e-4
    )
if __name__ == "__main__":
    main()
