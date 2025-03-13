import os
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

###############################################################################
# 1) DATASET: Loads images, returning (image, filename)
###############################################################################

class FilenameImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        self.image_paths = []
        for ext in exts:
            self.image_paths += glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True)
        self.image_paths = [p for p in self.image_paths if os.path.isfile(p)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(img_path)
        return img, filename

###############################################################################
# 2) WINDOW/GRID PARTITION HELPERS for MaxViT (local window + grid attention)
###############################################################################

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size, window_size, C)
    return windows

def window_unpartition(windows, window_size, H, W):
    B_ = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(
        B_,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B_, H, W, -1)
    return x

###############################################################################
# 3) MULTI-HEAD SELF ATTENTION
###############################################################################

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = out.permute(0, 2, 1, 3).contiguous()  # (B, N, num_heads, head_dim)
        out = out.view(B, N, C)
        out = self.proj(out)
        return out

###############################################################################
# 4) BLOCK ATTENTION + GRID ATTENTION
###############################################################################

def block_attention(x, attn, window_size):
    B, H, W, C = x.shape
    windows = window_partition(x, window_size)
    windows = windows.view(-1, window_size*window_size, C)
    out = attn(windows)
    out = out.view(-1, window_size, window_size, C)
    out = window_unpartition(out, window_size, H, W)
    return out

def grid_attention(x, attn, grid_size):
    B, H, W, C = x.shape
    cell_h = H // grid_size
    cell_w = W // grid_size
    assert cell_h == cell_w, "For simplicity, must be square cells."

    windows = window_partition(x, cell_h)
    windows = windows.view(-1, cell_h*cell_w, C)
    out = attn(windows)
    out = out.view(-1, cell_h, cell_w, C)
    out = window_unpartition(out, cell_h, H, W)
    return out

###############################################################################
# 5) FEED-FORWARD (MLP)
###############################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

###############################################################################
# 6) ONE MaxViT BLOCK (block attn -> MLP -> grid attn -> MLP)
###############################################################################

class MaxViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, grid_size, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.grid_size = grid_size

        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn_block = MultiHeadSelfAttention(dim, num_heads)
        self.ffn_block = FeedForward(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.norm3 = nn.LayerNorm(dim)
        self.attn_grid = MultiHeadSelfAttention(dim, num_heads)
        self.ffn_grid = FeedForward(dim, hidden_dim)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, x):
        B, H, W, C = x.shape

        # block attention
        shortcut = x
        x_ = x.view(B, H*W, C)
        x_ = self.norm1(x_)
        x_ = x_.view(B, H, W, C)
        x_ = block_attention(x_, self.attn_block, self.window_size)
        x = shortcut + x_

        # FFN
        shortcut = x
        x_ = x.view(B, H*W, C)
        x_ = self.norm2(x_)
        x_ = self.ffn_block(x_)
        x_ = x_.view(B, H, W, C)
        x = shortcut + x_

        # grid attention
        shortcut = x
        x_ = x.view(B, H*W, C)
        x_ = self.norm3(x_)
        x_ = x_.view(B, H, W, C)
        x_ = grid_attention(x_, self.attn_grid, self.grid_size)
        x = shortcut + x_

        # FFN
        shortcut = x
        x_ = x.view(B, H*W, C)
        x_ = self.norm4(x_)
        x_ = self.ffn_grid(x_)
        x_ = x_.view(B, H, W, C)
        x = shortcut + x_

        return x

###############################################################################
# 7) STACK MULTIPLE BLOCKS => MaxViT
###############################################################################

class MaxViT(nn.Module):
    def __init__(self, dim, depth=2, num_heads=4, window_size=8, grid_size=8, mlp_ratio=4.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                MaxViTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    grid_size=grid_size,
                    mlp_ratio=mlp_ratio
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

###############################################################################
# 8) UNet building blocks
###############################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

###############################################################################
# 9) Watermark Extractor for the universal code
###############################################################################

class WatermarkExtractor(nn.Module):
    def __init__(self, code_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32*32, code_dim)
        )

    def forward(self, x):
        return self.cnn(x)

###############################################################################
# 10) UNetWithRealMaxViT (Option B: Implicit Watermark)
###############################################################################

class UNetWithRealMaxViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_dim=64,
        maxvit_dim=512,
        maxvit_depth=2,
        maxvit_heads=8,
        window_size=8,
        grid_size=8
    ):
        super().__init__()

        # Down
        self.down1 = DownBlock(in_channels, base_dim)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DownBlock(base_dim, base_dim*2)
        self.down3 = DownBlock(base_dim*2, base_dim*4)
        self.down4 = DownBlock(base_dim*4, maxvit_dim)

        # MaxViT
        self.maxvit = MaxViT(
            dim=maxvit_dim,
            depth=maxvit_depth,
            num_heads=maxvit_heads,
            window_size=window_size,
            grid_size=grid_size
        )

        # Up
        self.up1 = UpBlock(maxvit_dim, base_dim*4)
        self.up2 = UpBlock(base_dim*4, base_dim*2)
        self.up3 = UpBlock(base_dim*2, base_dim)
        self.final_conv = nn.Conv2d(base_dim, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)          # (B, 64, 512, 512)
        x2 = self.pool(x1)
        x2 = self.down2(x2)         # (B, 128, 256, 256)
        x3 = self.pool(x2)
        x3 = self.down3(x3)         # (B, 256, 128, 128)
        x4 = self.pool(x3)
        x4 = self.down4(x4)         # (B, 512, 64, 64)

        B, C, H, W = x4.shape
        x4_m = x4.permute(0, 2, 3, 1).contiguous()  # (B, 64, 64, 512)
        x4_m = self.maxvit(x4_m)                    # (B, 64, 64, 512)
        x4_out = x4_m.permute(0, 3, 1, 2).contiguous()  # (B, 512, 64, 64)

        x = self.up1(x4_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.final_conv(x)   # (B, 3, 512, 512)
        return out

###############################################################################
# 11) SAVE WATERMARKED IMAGE
###############################################################################

def save_watermarked_image(img_tensor, original_filename, output_dir="D:\\Dropbox\\UMA Augusta\\PhD\\Research Thesis\\brain_tumor_mri_dataset\\watermarked_outputs_MaxVIT_noDiscrim"):
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}_watermarked_{timestamp}{ext}"
    save_path = os.path.join(output_dir, new_filename)
    clamped = img_tensor.clamp(0,1)
    vutils.save_image(
        img_tensor,
        save_path,
        normalize=True,  # scale values for better visual
        value_range=(0, 1)
    )# if the data is theoretically in 0..1

    #vutils.save_image(clamped, save_path)
    return save_path

###############################################################################
# 12) TRAINING WITH nn.DataParallel FOR MULTI-GPU
###############################################################################

def train_unet_real_maxvit_implicit(
    data_dir,
    epochs=2,
    batch_size=4,
    lr=1e-5,
    code_dim=128,
    model_params=None
):
    """
    1) DataParallel for multi-GPU usage
    2) If batch_size=2 and you have 2 GPUs, each GPU gets 1 image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A single universal watermark code
    universal_code = torch.randn(code_dim, device=device)

    transform = T.Compose([T.Resize((512,512)), T.ToTensor()])
    dataset = FilenameImageDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model creation
    if model_params is None:
        model_params = {
            'in_channels': 3,
            'base_dim': 64,
            'maxvit_dim': 512,
            'maxvit_depth': 2,
            'maxvit_heads': 8,
            'window_size': 8,
            'grid_size': 8
        }
    model = UNetWithRealMaxViT(**model_params)

    # Watermark extractor
    extractor = WatermarkExtractor(code_dim=code_dim)

    # Wrap them in nn.DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs with nn.DataParallel.")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert bytes to GB
            compute_capability = torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor
            print(f"GPU {i}: {gpu_name}")
            print(f"    Memory: {gpu_memory:.2f} GB")
            print(f"    Compute Capability: {compute_capability[0]}.{compute_capability[1]}\n")

        model = nn.DataParallel(model, device_ids=[0, 1])
        extractor = nn.DataParallel(extractor, device_ids=[0, 1])


    model.to(device)
    extractor.to(device)

    # Losses
    recon_loss_fn = nn.MSELoss()
    wm_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(list(model.parameters()) + list(extractor.parameters()), lr=lr)

    print("Starting multi-GPU training with Real MaxViT in the Bottleneck (Option B)...")
    for epoch in range(1, epochs+1):
        model.train()
        extractor.train()

        total_recon = 0.0
        total_wm = 0.0

        for batch_idx, (images, filenames) in enumerate(loader):
            images = images.to(device)

            # Forward
            watermarked = model(images)

            # Recon Loss
            loss_recon = recon_loss_fn(watermarked, images)

            # Watermark Loss
            extracted_code = extractor(watermarked)
            target_code = universal_code.unsqueeze(0).expand(extracted_code.size(0), -1)
            loss_wm = wm_loss_fn(extracted_code, target_code)

            loss = loss_recon + 0.15*loss_wm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_wm += loss_wm.item()

            # Save sample occasionally
            if (batch_idx+1) % 2 == 0:
                print(f"Epoch[{epoch}/{epochs}], Batch {batch_idx+1}/{len(loader)}")
                print(f"  Recon Loss: {loss_recon.item():.4f}, WM Loss: {loss_wm.item():.4f}")
                if epoch >= 2:
                    saved_img = watermarked[0].detach().cpu()
                    fname = filenames[0]
                    out_path = save_watermarked_image(saved_img, fname)
                    print(f"  -> Saved watermarked image: {out_path} from epoch {epoch}")

        avg_recon = total_recon / len(loader)
        avg_wm = total_wm / len(loader)
        print(f"=> Epoch {epoch}/{epochs} done. Recon: {avg_recon:.4f}, WM: {avg_wm:.4f}")

    print("Training complete. Multi-GPU run finished!")

###############################################################################
# 13) USAGE EXAMPLE
###############################################################################

if __name__ == "__main__":
    data_directory = "D:\\Dropbox\\UMA Augusta\\PhD\\Research Thesis\\brain_tumor_mri_dataset\\Training"

    # If you have 2 GPUs and want each GPU to process exactly 1 image => batch_size=2
    # If you have e.g. 2 GPUs and want each GPU to process 2 images => batch_size=4
    train_unet_real_maxvit_implicit(
        data_dir=data_directory,
        epochs=15,
        batch_size=10,
        lr=1e-5,
        code_dim=128,
        model_params={
            'in_channels': 3,
            'base_dim': 64,
            'maxvit_dim': 512,
            'maxvit_depth': 2,
            'maxvit_heads': 8,
            'window_size': 8,
            'grid_size': 8
        }
    )
