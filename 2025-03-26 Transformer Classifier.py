import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", message="PyTorch is not compiled with NCCL support")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
import timm  # for transformer-based models (e.g. Swin Transformer)

###############################################################################
# 1) DATASET FOR WATERMARKING (unchanged): Returns (image, full_img_path)
###############################################################################
class FilenameImageDataset(Dataset):
    """
    Recursively loads images from 'root_dir' and returns (transformed_image, full_img_path).
    We'll use the full path to replicate subfolder structure in the output folder.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

        self.image_paths = []
        for ext in exts:
            self.image_paths += glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True)
        # Filter out anything that isn't a file
        self.image_paths = [p for p in self.image_paths if os.path.isfile(p)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path


###############################################################################
# 2) Utility: Reproduce subfolder structure in target directory, now with epoch
###############################################################################
def get_output_path(original_img_path, data_root, target_root, epoch):
    rel_path = os.path.relpath(original_img_path, start=data_root)
    subdir = os.path.dirname(rel_path)
    base_name = os.path.basename(rel_path)
    name_no_ext, ext = os.path.splitext(base_name)
    final_subdir = os.path.join(target_root, subdir)
    os.makedirs(final_subdir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{name_no_ext}_epoch{epoch}_watermarked_{timestamp}{ext}"
    final_path = os.path.join(final_subdir, new_filename)
    return final_path


###############################################################################
# 3) WINDOW/GRID PARTITION HELPERS (unchanged)
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
# 4) MULTI-HEAD SELF ATTENTION (unchanged)
###############################################################################
import torch.nn.functional as F

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
# 5) BLOCK & GRID ATTENTION (unchanged)
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
# 6) FEED-FORWARD (unchanged)
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
# 7) MaxViTBlock (unchanged)
###############################################################################
class MaxViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, grid_size, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.grid_size = grid_size

        hidden_dim = int(dim * mlp_ratio)

        # block attn
        self.norm1 = nn.LayerNorm(dim)
        self.attn_block = MultiHeadSelfAttention(dim, num_heads)
        self.ffn_block = FeedForward(dim, hidden_dim)
        self.norm2 = nn.LayerNorm(dim)

        # grid attn
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
# 8) MaxViT (unchanged)
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
# 9) UNet components (unchanged)
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
# 10) WatermarkExtractor (unchanged)
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
# 11) UNetWithRealMaxViT (Option B)
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
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)

        B, C, H, W = x4.shape
        x4_m = x4.permute(0, 2, 3, 1).contiguous()
        x4_m = self.maxvit(x4_m)
        x4_out = x4_m.permute(0, 3, 1, 2).contiguous()

        x = self.up1(x4_out, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.final_conv(x)
        return out


###############################################################################
# 12) Save watermarked image with epoch in filename
###############################################################################
def save_watermarked_image(img_tensor, original_img_path, data_root, target_root, epoch):
    final_path = get_output_path(original_img_path, data_root, target_root, epoch)
    clamped = img_tensor.clamp(0,1)
    vutils.save_image(clamped, final_path)
    return final_path


###############################################################################
# 13) TRAINING Watermark with MaxViT
###############################################################################
def train_unet_real_maxvit_implicit(
    data_dir,
    target_dir,
    epochs=5,
    batch_size=2,
    lr=1e-5,
    code_dim=128,
    model_params=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single universal watermark code
    universal_code = torch.randn(code_dim, device=device)

    transform = T.Compose([T.Resize((512,512)), T.ToTensor()])
    dataset = FilenameImageDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
    extractor = WatermarkExtractor(code_dim=code_dim)

    # DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)
        extractor = nn.DataParallel(extractor)

    model.to(device)
    extractor.to(device)

    recon_loss_fn = nn.MSELoss()
    wm_loss_fn = nn.MSELoss()

    optimizer = optim.Adam(list(model.parameters()) + list(extractor.parameters()), lr=lr)

    print("Starting multi-GPU training with real MaxViT (Option B), preserving subfolders & epoch in filenames.")
    for epoch in range(1, epochs+1):
        model.train()
        extractor.train()

        total_recon = 0.0
        total_wm = 0.0

        for batch_idx, (images, full_paths) in enumerate(loader):
            images = images.to(device)
            watermarked = model(images)

            # Reconstruction loss
            loss_recon = recon_loss_fn(watermarked, images)

            # Watermark loss
            extracted_code = extractor(watermarked)
            target_code = universal_code.unsqueeze(0).expand(extracted_code.size(0), -1)
            loss_wm = wm_loss_fn(extracted_code, target_code)

            loss = loss_recon + loss_wm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_wm += loss_wm.item()

            # Print progress every 50 batches
            if (batch_idx % 50) == 0:
                print(f"Epoch[{epoch}/{epochs}] Batch {batch_idx+1}/{len(loader)}")
                print(f"  Recon Loss: {loss_recon.item():.4f} | WM Loss: {loss_wm.item():.4f}")

                # Optionally save image from second epoch onward
                if epoch >= 2:
                    saved_img = watermarked[0].detach().cpu()
                    original_path = full_paths[0]
                    out_path = save_watermarked_image(saved_img, original_path, data_dir, target_dir, epoch)
                    print(f"  -> Saved watermarked image: {out_path}")

        avg_recon = total_recon / len(loader)
        avg_wm = total_wm / len(loader)
        print(f"=> Epoch {epoch}/{epochs} complete. Recon: {avg_recon:.4f} | WM: {avg_wm:.4f}")

    print("Training complete. Subfolder structure + epoch in filenames are now preserved!")


###############################################################################
# 14) CLASSIFIER CODE (TRANSFORMER-BASED) + Evaluate on Testing
###############################################################################
class MRI4ClassDataset(Dataset):
    """
    Loads 4 classes from subfolders:
      root/
        glioma/
        meningioma/
        pituitary/
        notumor/
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.classes = ["glioma", "meningioma", "pituitary", "notumor"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            class_idx = self.class_to_idx[class_name]
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                for f in glob.glob(os.path.join(class_folder, f"*{ext}")):
                    self.image_paths.append(f)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_classifier(model, data_loader, device):
    """
    Returns classification accuracy on the given data_loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_4class_transformer_classifier(
    train_root,
    test_root,
    model_path="mri_4class_transformer.pth",
    test_acc_path="mri_4class_transformer_test_accuracy.txt",
    epochs=5,
    lr=1e-4
):
    """
    Use a Swin Transformer from timm as a 4-class classifier.
    If `model_path` + `test_acc_path` exist, skip training & testing.

    We'll use timm's "swin_tiny_patch4_window7_224" as an example.
    """
    if os.path.isfile(model_path) and os.path.isfile(test_acc_path):
        print("[Transformer Classifier] Found model & test accuracy files. Skipping training & testing.")
        return None

    import timm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Typically, Swin models expect 224x224 (or 384, etc.). We'll do 224x224.
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        # Optionally, use normal ImageNet means/stdev if desired
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MRI4ClassDataset(train_root, transform=transform)
    test_dataset  = MRI4ClassDataset(test_root,  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False, num_workers=2)

    # 1) Create a Swin Transformer from timm, pre-trained on ImageNet
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=4)

    # 2) Replace the final classification head with 4 outputs
    # For Swin, the last layer is `model.head`
    #in_features = model.head.in_features
    #model.head = nn.Linear(in_features, 4)

    if torch.cuda.device_count() > 1:
        print(f"[Transformer Classifier] Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("[Transformer Classifier] Starting training...")
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Transformer Classifier] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    # Evaluate on test set
    test_acc = evaluate_classifier(model, test_loader, device)
    print(f"[Transformer Classifier] Final Test Accuracy: {test_acc*100:.2f}%")

    # Save model & test accuracy
    torch.save(model.state_dict(), model_path)
    with open(test_acc_path, 'w') as f:
        f.write(f"{test_acc:.4f}\n")

    print(f"[Transformer Classifier] Model saved to {model_path}")
    print(f"[Transformer Classifier] Test accuracy saved to {test_acc_path}")

    return model


###############################################################################
# 15) MAIN
###############################################################################
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # A) CLASSIFIER (Transformer) Data & Output
    # -------------------------------------------------------------------------
    classifier_train_dir = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Training"
    classifier_test_dir  = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Testing"

    # We'll save the transformer model here
    transformer_model_path    = "mri_4class_transformer.pth"
    transformer_testacc_path  = "mri_4class_transformer_test_accuracy.txt"

    # -------------------------------------------------------------------------
    # B) WATERMARKING Data & Output
    # -------------------------------------------------------------------------
    data_dir   = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset"
    target_dir = r"E:\watermarking\MaxVIt50percent"

    # 1) Train & Test Transformer Classifier (only if needed)
    train_4class_transformer_classifier(
        train_root=classifier_train_dir,
        test_root=classifier_test_dir,
        model_path=transformer_model_path,
        test_acc_path=transformer_testacc_path,
        epochs=5,
        lr=1e-4
    )

    # 2) Proceed with Watermarking (MaxViT)
    print("[Main] Now running the MaxViT watermark training...")
    train_unet_real_maxvit_implicit(
        data_dir=data_dir,
        target_dir=target_dir,
        epochs=5,
        batch_size=2,
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

    print("[Main] Pipeline complete!")
