import os
import glob
import random
import datetime
import shutil
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.models import efficientnet_b6

###########################################
# 1) Helper: replicate subfolder + save
###########################################

def replicate_subfolder_and_save(
    img_tensor: torch.Tensor,
    original_path: str,
    source_root: str,
    target_root: str,
    epoch: int = None
):
    """
    Replicates subfolder structure from `source_root` to `target_root`.
    Optionally appends epoch + datetime to the filename.
    """
    rel_path = os.path.relpath(original_path, start=source_root)
    subdir = os.path.dirname(rel_path)
    base_name = os.path.basename(rel_path)
    name_no_ext, ext = os.path.splitext(base_name)

    out_subdir = os.path.join(target_root, subdir)
    os.makedirs(out_subdir, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if epoch is not None:
        new_filename = f"{name_no_ext}_epoch{epoch}_{now_str}{ext}"
    else:
        new_filename = f"{name_no_ext}_{now_str}{ext}"

    final_path = os.path.join(out_subdir, new_filename)

    # clamp & save
    img_tensor = img_tensor.clamp(0,1)
    save_image(img_tensor, final_path)
    return final_path

###########################################
# 2) Load / Freeze Classifier #1
###########################################

def load_frozen_classifier1(path, device, num_classes=3):
    model = efficientnet_b6(weights = None)

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
# 3) UNet+MaxViT Watermark Generator
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
        x_conv = self.conv(x)          # keep same spatial size, channels=out_c
        x_down = self.pool(x_conv)     # downsample => half spatial
        return x_conv, x_down         # we return both: skip (x_conv) + pooled (x_down)


class UpBlock(nn.Module):
    """
    UpBlock that:
      1) ConvTranspose2d(in_c -> out_c) to double spatial size
      2) Concat skip (skip_c channels) and the upsampled x (out_c channels)
      3) DoubleConv((skip_c + out_c) -> out_c)
    """
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_c + skip_c, out_c)

    def forward(self, x, skip):
        """
        x:    the feature map from the previous up/bottleneck, shape (B, in_c, H, W)
        skip: the skip feature from the down path, shape (B, skip_c, 2H, 2W)
        1) upsample x => (B, out_c, 2H, 2W)
        2) concat => (B, (out_c + skip_c), 2H, 2W)
        3) double conv => (B, out_c, 2H, 2W)
        """
        x = self.up(x)                            # => (B, out_c, 2H, 2W)
        x = torch.cat([skip, x], dim=1)           # => (B, skip_c + out_c, 2H, 2W)
        x = self.conv(x)                          # => (B, out_c, 2H, 2W)
        return x


class SimpleAttention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj= nn.Linear(dim, dim)
        self.num_heads = heads
        self.head_dim = dim//heads
    def forward(self, x):
        B,N,C = x.shape
        qkv = self.qkv(x).view(B,N,3,self.num_heads,self.head_dim)
        q,k,v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        scores = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).reshape(B,N,C)
        return self.proj(out)

def partition_windows(x, ws=8):
    B,H,W,C = x.shape
    x = x.view(B,H//ws,ws,W//ws,ws,C)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.reshape(B*(H//ws)*(W//ws), ws, ws, C)

def unpartition_windows(x, ws, B,H,W):
    n = x.shape[0]
    bHW = (H//ws)*(W//ws)
    b = n//bHW
    x = x.view(b,H//ws,W//ws, ws,ws,-1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.view(b,H,W,-1)

def block_attn(x, attn, ws=8):
    B,H,W,C = x.shape
    blocks = partition_windows(x, ws)
    blocks = blocks.view(-1, ws*ws, C)
    out = attn(blocks)
    out = out.view(-1, ws, ws, C)
    out = unpartition_windows(out, ws, B,H,W)
    return out

class SimpleMaxViTBlock(nn.Module):
    def __init__(self, dim=512, heads=8, ws=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim)
        )
        self.ws = ws
    def forward(self, x):
        B,H,W,C = x.shape
        # block attention
        x_ = x.view(B,H*W,C)
        x_ = self.ln1(x_)
        x_ = x_.view(B,H,W,C)
        x_ = block_attn(x_, self.attn, self.ws)
        x = x + x_
        # feed-forward
        x_ = x.view(B,H*W,C)
        x_ = self.ln2(x_)
        x_ = self.ffn(x_)
        x_ = x_.view(B,H,W,C)
        x = x + x_
        return x

class UNetWithRealMaxViT(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, mid_dim=512):
        super().__init__()
        # 4 DownBlocks
        self.down1 = DownBlock(in_channels, base_dim)          # (3->64)
        self.down2 = DownBlock(base_dim, base_dim*2)           # (64->128)
        self.down3 = DownBlock(base_dim*2, base_dim*4)         # (128->256)
        self.down4 = DownBlock(base_dim*4, mid_dim)            # (256->512)

        # MaxViT block in bottleneck
        self.maxvit = SimpleMaxViTBlock(dim=mid_dim, heads=8, ws=8)

        # 4 UpBlocks (specify skip channels + out channels)
        self.up1 = UpBlock(in_c=mid_dim, skip_c=mid_dim, out_c=base_dim*4)  # (512->256) skip=512 => total=768 => conv(768->256)
        self.up2 = UpBlock(in_c=base_dim*4, skip_c=base_dim*4, out_c=base_dim*2) # (256->128), skip=256 => total=384 => conv(384->128)
        self.up3 = UpBlock(in_c=base_dim*2, skip_c=base_dim*2, out_c=base_dim)   # (128->64), skip=128 => total=192 => conv(192->64)
        self.up4 = UpBlock(in_c=base_dim, skip_c=base_dim, out_c=base_dim)       # (64->64), skip=64 => total=128 => conv(128->64)

        self.final_conv = nn.Conv2d(base_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # down 1
        d1, d1p = self.down1(x)   # d1: (B,64,512,512), d1p: (B,64,256,256)
        d2, d2p = self.down2(d1p) # d2: (B,128,256,256), d2p: (B,128,128,128)
        d3, d3p = self.down3(d2p) # d3: (B,256,128,128), d3p: (B,256,64,64)
        d4, d4p = self.down4(d3p) # d4: (B,512,64,64),  d4p: (B,512,32,32)

        # MaxViT at bottleneck
        B,C,H,W = d4p.shape  # (B,512,32,32)
        x_ = d4p.permute(0,2,3,1)  # => (B,32,32,512)
        x_ = self.maxvit(x_)       # => (B,32,32,512)
        x_ = x_.permute(0,3,1,2)   # => (B,512,32,32)

        # up path
        up1_out = self.up1(x_, d4)  # in=512, skip=512 -> out=256
        up2_out = self.up2(up1_out, d3) # in=256, skip=256 -> out=128
        up3_out = self.up3(up2_out, d2) # in=128, skip=128 -> out=64
        up4_out = self.up4(up3_out, d1) # in=64, skip=64  -> out=64

        out = self.final_conv(up4_out) # => (B,3,512,512)
        return out

############################################
# 4) SimpleMaxViTClassifier for second classifier
############################################

class SimpleMaxViTClassifier(nn.Module):
    def __init__(self, num_classes=3, dim=256, heads=8, ws=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),  # 512->256
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),# 256->128
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),# 128->64
            nn.ReLU()
        )
        self.block = SimpleMaxViTBlock(dim, heads, ws)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.stem(x)  # (B,dim,64,64)
        _,C,H,W = x.shape
        x = x.permute(0,2,3,1)
        x = self.block(x)
        x = x.view(B,H*W,C)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)

#######################################################
# 5) TRAIN WATERMARK + TEMP CLASSIFIER W/LEARNABLE THRESHOLDS
#######################################################

def train_watermark_with_auto_thresholds(
    data_dir:str,
    device:torch.device,
    classifier1_path:str,
    partial_save_root:str,
    final_wm_root:str,
    out_dir:str,
    epochs=10,
    batch_size=8,
    base_lr=2e-4
):
    """
    Full approach:
    1) Loads classifier #1 (frozen).
    2) Builds UNet+MaxViT (wm_gen) & a second classifier (c2).
    3) We define 4 constraints:
       - c1_loss_wm <= eps_c1
       - c2_loss_wm <= eps_c2_wm
       - c2_loss_clean >= eps_c2_clean
       - diff <= eps_diff
    4) Each eps_i is LEARNABLE (via softplus).
       Each constraint has a Lagrange multiplier that adapts.
    5) Save partial (1/25) watermarked images
       E:\\watermarking\afhq\\<class>\\epoch\
    6) Generate final watermarked dataset into final_wm_root
    """
    os.makedirs(partial_save_root, exist_ok=True)
    os.makedirs(final_wm_root, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load c1
    c1 = load_frozen_classifier1(classifier1_path, device, num_classes=3)

    # 2) Build wm_gen & c2
    wm_gen = UNetWithRealMaxViT()
    c2 = SimpleMaxViTClassifier(num_classes=3)

    if torch.cuda.device_count()>1:
        wm_gen = nn.DataParallel(wm_gen)
        c2 = nn.DataParallel(c2)

    wm_gen = wm_gen.to(device)
    c2 = c2.to(device)

    # 3) Dataloader
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor()
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 4) Lagrange multipliers
    lambda_c1 = 0.0
    lambda_c2_wm = 0.0
    lambda_c2_clean=0.0
    lambda_diff=0.0
    lam_step = 1e-4

    # -------------------------------------------------------------------------
    # 5) LEARNABLE THRESHOLDS (eps_i)
    #    We can start them at bigger or more lenient values. Also add optional clamp.
    # -------------------------------------------------------------------------
    # Original were ~0.2,1.0,0.02. Let’s make them bigger or more lenient:
    eps_c1_raw = nn.Parameter(torch.tensor(1.0))  # => c1_loss_wm <= ~1.0
    eps_c2_wm_raw = nn.Parameter(torch.tensor(1.0))  # => c2_loss_wm <= ~1.0
    eps_c2_clean_raw = nn.Parameter(torch.tensor(1.0))  # => c2_loss_clean >= ~1.0
    eps_diff_raw = nn.Parameter(torch.tensor(0.1))  # => diff <= ~0.1 initially

    def softplus(x):
        return torch.log(1+torch.exp(x))

    def clamp_eps(val, minv, maxv):
        return torch.clamp(val, min=minv, max=maxv)

    # We'll define an optional small penalty for threshold regularization
    # so they don't blow up or vanish.
    threshold_reg_weight = 1e-3

    # -------------------------------------------------------------------------
    # 6) Single optimizer for (wm_gen, c2) + threshold params
    # -------------------------------------------------------------------------

    # 6) Single optimizer for (wm_gen, c2) + threshold params
    all_params = list(wm_gen.parameters()) + list(c2.parameters()) + [
        eps_c1_raw, eps_c2_wm_raw, eps_c2_clean_raw, eps_diff_raw
    ]
    opt = optim.Adam(all_params, lr=base_lr)
    ce_loss = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # 7) (Optional) MSE penalty to reduce "too dark" images.
    #    We'll define a scalar gamma for the MSE. You can tune this.
    # -------------------------------------------------------------------------

    gamma = 0.1  # or 0.05, 0.01, etc. - this is an additional control range parameter

    step_count = 0
    for epoch in range(1, epochs+1):
        wm_gen.train()
        c2.train()

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # forward
            x_wm = wm_gen(imgs)

            # c1 on x_wm => want success => c1_loss
            with torch.no_grad():
                out1_wm = c1(x_wm)
            loss_c1_wm = ce_loss(out1_wm, labels)

            # c2 on x_wm => want success => c2_wm
            out2_wm = c2(x_wm)
            loss_c2_wm = ce_loss(out2_wm, labels)

            # c2 on clean => want fail => => c2_clean
            out2_clean= c2(imgs)
            loss_c2_clean = ce_loss(out2_clean, labels)

            # difference
            diff = (x_wm - imgs).pow(2).mean()

            # compute eps_i
            eps_c1_val      = softplus(eps_c1_raw)
            eps_c2_wm_val   = softplus(eps_c2_wm_raw)
            eps_c2_clean_val= softplus(eps_c2_clean_raw)
            eps_diff_val    = softplus(eps_diff_raw)



            eps_c1_val = clamp_eps(eps_c1_val, 0.1, 2.0)
            eps_c2_wm_val = clamp_eps(eps_c2_wm_val, 0.1, 2.0)
            eps_c2_clean_val = clamp_eps(eps_c2_clean_val, 0.1, 2.0)
            eps_diff_val = clamp_eps(eps_diff_val, 0.0, 0.5)  # 0.5 if we want a cap

            # constraints (in-graph)
            c_c1 = (loss_c1_wm - eps_c1_val).clamp(min=0)  # want <= eps_c1
            c_c2wm = (loss_c2_wm - eps_c2_wm_val).clamp(min=0)  # want <= eps_c2_wm
            c_c2cl = (eps_c2_clean_val - loss_c2_clean).clamp(min=0)  # want >= eps_c2_clean
            c_diff = (diff - eps_diff_val).clamp(min=0)  # want <= eps_diff

            # penalty
            penalty = (lambda_c1*c_c1 +
                       lambda_c2_wm*c_c2wm +
                       lambda_c2_clean*c_c2cl +
                       lambda_diff*c_diff)

            # optional threshold regularization
            # e.g. we want them not to blow up => small penalty
            # sum of eps_i or sum of raw
            reg_eps = threshold_reg_weight * (eps_c1_val + eps_c2_wm_val + eps_c2_clean_val + eps_diff_val)

            # (Optional) MSE penalty to discourage very dark images:
            # If you see images are extremely dark, a small gamma can keep them near original.
            mse_term = gamma * F.mse_loss(x_wm, imgs)

            total_loss = penalty + reg_eps + mse_term

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            # update multipliers after forward
            c_c1_val_f = (loss_c1_wm.item() - eps_c1_val.item())
            c_c2wm_val_f = (loss_c2_wm.item() - eps_c2_wm_val.item())
            c_c2cl_val_f = (eps_c2_clean_val.item() - loss_c2_clean.item())
            c_diff_val_f = (diff.item() - eps_diff_val.item())
            # We changed lam_step => 1e-4

            if c_c1_val_f > 0:
                lambda_c1 += lam_step
            else:
                lambda_c1 = max(lambda_c1 - lam_step * 0.5, 0.0)

            if c_c2wm_val_f > 0:
                lambda_c2_wm += lam_step
            else:
                lambda_c2_wm = max(lambda_c2_wm - lam_step * 0.5, 0.0)

            if c_c2cl_val_f > 0:
                lambda_c2_clean += lam_step
            else:
                lambda_c2_clean = max(lambda_c2_clean - lam_step * 0.5, 0.0)

            if c_diff_val_f > 0:
                lambda_diff += lam_step
            else:
                lambda_diff = max(lambda_diff - lam_step * 0.5, 0.0)

            step_count += 1

            if batch_idx % 25 == 0:
                print(f"[Epoch {epoch}, Batch {batch_idx}] c1_wm={loss_c1_wm:.3f}, c2_wm={loss_c2_wm:.3f}, c2_clean={loss_c2_clean:.3f}, diff={diff:.4f}")
                print(f"  eps_c1={eps_c1_val.item():.3f}, eps_c2_wm={eps_c2_wm_val.item():.3f}, eps_c2_cl={eps_c2_clean_val.item():.3f}, eps_diff={eps_diff_val.item():.4f}")
                print(f"  lambdas=({lambda_c1:.3f},{lambda_c2_wm:.3f},{lambda_c2_clean:.3f},{lambda_diff:.3f})")

            # Save 1 in 25 partial images
            if (batch_idx%25)==0:
                idx0 = 0
                wm0 = x_wm[idx0].detach().cpu()
                sample_idx = batch_idx*batch_size + idx0
                if sample_idx < len(dataset.samples):
                    orig_path, cls_idx = dataset.samples[sample_idx]
                    class_name = dataset.classes[cls_idx]
                    # replicate E:\\watermarking\\afhq\\<class_name>\\<epoch>\\
                    out_subdir = os.path.join(partial_save_root, class_name, str(epoch))
                    os.makedirs(out_subdir, exist_ok=True)
                    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"wm_{sample_idx}_{now_str}.png"
                    final_path = os.path.join(out_subdir, filename)
                    save_image(wm0.clamp(0,1), final_path)

        print(f"End of epoch {epoch}: eps_c1={eps_c1_val.item():.3f}, eps_c2_wm={eps_c2_wm_val.item():.3f}, eps_c2_clean={eps_c2_clean_val.item():.3f}, eps_diff={eps_diff_val.item():.4f}")

    print("[INFO] Done training watermark generator + c2 with auto-thresholds.")
    # save final states
    wm_gen_path = os.path.join(out_dir, "wm_generator_final.pth")
    c2_path     = os.path.join(out_dir, "temp_c2_final.pth")
    torch.save(wm_gen.state_dict(), wm_gen_path)
    torch.save(c2.state_dict(), c2_path)

    # Generate final watermarked dataset
    print("[INFO] Generating full watermarked dataset ...")
    wm_gen.eval()
    final_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for i,(img, lbl) in enumerate(final_loader):
            img = img.to(device)
            wm = wm_gen(img)
            wm_cpu = wm[0].cpu()
            orig_path, _ = dataset.samples[i]
            replicate_subfolder_and_save(
                wm_cpu, orig_path, source_root=data_dir,
                target_root=final_wm_root,
                epoch=None
            )

    print("[INFO] Full watermarked dataset saved at:", final_wm_root)
    return wm_gen_path, c2_path

##########################################################
# 6) Train final classifier #2 on the full watermarked set
##########################################################

class FinalMaxViTClassifier(nn.Module):
    """
    Possibly the same architecture or a new one. We'll reuse SimpleMaxViTClassifier.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = SimpleMaxViTClassifier(num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

def train_final_classifier2(
    watermarked_dir:str,
    device:torch.device,
    out_model_path:str = r"E:\\watermarking\\afhq\\class2_model\\watermarked_classifier.pth",
    epochs=10,
    batch_size=16,
    lr=1e-4

):
    """
    Standard classification training on the final watermarked dataset.
    """
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    transform = T.Compose([
        T.Resize((512,512)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    dataset = ImageFolder(watermarked_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    c2 = FinalMaxViTClassifier(num_classes=len(dataset.classes))
    if torch.cuda.device_count()>1:
        c2 = nn.DataParallel(c2)
    c2 = c2.to(device)

    optimizer = optim.Adam(c2.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
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

            total_loss += loss.item()*imgs.size(0)
            _, preds = torch.max(out,1)
            correct += (preds==labels).sum().item()
        epoch_loss = total_loss/len(dataset)
        epoch_acc  = correct/len(dataset)
        print(f"[FinalC2] Epoch {epoch}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

    torch.save(c2.state_dict(), out_model_path)
    print(f"[INFO] Final classifier #2 saved at {out_model_path}")

##########################################################
# 7) MAIN
##########################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # a) paths
    classifier1_path = r"E:\\watermarking\\afhq\\trained_model\\efficientnet_b6_classifier.pth"
    train_data_dir   = r"E:\\watermarking\\afhq\\train"   # unwatermarked data
    partial_save_root= r"E:\\watermarking\\afhq"         # for partial 1-in-25 images
    final_wm_root    = r"E:\\watermarking\\afhq\\full_final_watermarked"
    out_dir          = r"E:\\watermarking\\afhq\\dual_training_outputs"

    # b) Train watermark generator + temp c2 with auto thresholds
    train_watermark_with_auto_thresholds(
        data_dir=train_data_dir,
        device=device,
        classifier1_path=classifier1_path,
        partial_save_root=partial_save_root,
        final_wm_root=final_wm_root,
        out_dir=out_dir,
        epochs=10,
        batch_size=8,
        base_lr=1e-4
    )

    # c) Now train final classifier #2 on the full watermarked
    final_c2_path = r"E:\\watermarking\\afhq\\class2_model\\watermarked_classifier.pth"
    train_final_classifier2(
        watermarked_dir=final_wm_root,
        device=device,
        out_model_path=final_c2_path,
        epochs=10,
        batch_size=8,
        lr=1e-4
    )

if __name__=="__main__":
    main()