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
import torch.nn.functional as F
from PIL import Image

############################################
# 1) CLASSIFIER: EfficientNet-B3 + Data Aug
############################################
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights, efficientnet_b6, EfficientNet_B6_Weights

class MRI4ClassDataset(Dataset):
    """
    Loads images from subfolders under root_dir:
      root_dir/
        glioma/
        meningioma/
        pituitary/
        notumor/
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_idx = {
            "notumor": 0,
            "glioma": 1,
            "meningioma": 2,
            "pituitary": 3
        }
        self.classes = list(self.class_to_idx.keys())

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


def create_efficientnet_b3(num_classes=4):
    """
    Loads EfficientNet-B3 (pretrained on ImageNet),
    replaces final layer with a small MLP for your 4-class MRI problem.
    """
    weights = EfficientNet_B5_Weights.IMAGENET1K_V1
    model = efficientnet_b5(weights=weights)  # torchvision >= 2.0

    # Original final layer has 1536 in_features for B3
    in_features = model.classifier[1].in_features

    # Example: small MLP after the dropout
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model


def train_efficientnet_b3_classifier(
    train_root,
    test_root,
    model_path="efficientnet_b5_classifier.pth",
    epochs=10,
    lr=1e-5,
    batch_size=4,
    num_workers=2
):
    """
    1) If model_path exists, skip training (or load).
    2) Otherwise:
       - Data augment on training images
       - Evaluate test accuracy each epoch
       - Save final model
    """
    if os.path.isfile(model_path):
        print(f"[Classifier] Found {model_path}, skipping training. (Or load if desired.)")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A) Data augmentation for training
    train_transform = T.Compose([
        T.Resize((300,300)),
        T.RandomResizedCrop((224,224), scale=(0.8,1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Test transform
    test_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # B) Datasets & loaders
    train_dataset = MRI4ClassDataset(train_root, transform=train_transform)
    test_dataset  = MRI4ClassDataset(test_root,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # C) Model: EfficientNet-B3
    model = create_efficientnet_b3(num_classes=4)
    if torch.cuda.device_count() > 1:
        print(f"[Classifier] Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)

    model.to(device)

    # D) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # E) Training loop with test each epoch
    for epoch in range(1, epochs+1):
        # TRAIN
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total

        # TEST
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}%")

    # Save final
    torch.save(model.state_dict(), model_path)
    print(f"[Classifier] Saved final model to {model_path}")
    return model


##########################################
# 2) MAXViT WATERMARKING CODE (UNCHANGED)
##########################################
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
        return img, img_path


def save_watermarked_image(img_tensor, original_full_path, data_root, target_root, epoch):
    final_path = get_output_path(original_full_path, data_root, target_root, epoch)
    clamped = img_tensor.clamp(0,1)
    vutils.save_image(clamped, final_path)
    return final_path

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

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N, C)
        out = self.proj(out)
        return out

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
    assert cell_h == cell_w, "Must be square cells."

    windows = window_partition(x, cell_h)
    windows = windows.view(-1, cell_h*cell_w, C)
    out = attn(windows)
    out = out.view(-1, cell_h, cell_w, C)
    out = window_unpartition(out, cell_h, H, W)
    return out

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
        self.down1 = DownBlock(in_channels, base_dim)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DownBlock(base_dim, base_dim*2)
        self.down3 = DownBlock(base_dim*2, base_dim*4)
        self.down4 = DownBlock(base_dim*4, maxvit_dim)

        self.maxvit = MaxViT(
            dim=maxvit_dim,
            depth=maxvit_depth,
            num_heads=maxvit_heads,
            window_size=window_size,
            grid_size=grid_size
        )

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

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel for Watermarker!")
        model = nn.DataParallel(model)
        extractor = nn.DataParallel(extractor)

    model.to(device)
    extractor.to(device)

    recon_loss_fn = nn.MSELoss()
    wm_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(list(model.parameters()) + list(extractor.parameters()), lr=lr)

    print("[MaxViT Watermarker] Starting training...")
    for epoch in range(1, epochs+1):
        model.train()
        extractor.train()

        total_recon = 0.0
        total_wm = 0.0

        for batch_idx, (images, full_paths) in enumerate(loader):
            images = images.to(device)
            watermarked = model(images)

            loss_recon = recon_loss_fn(watermarked, images)
            extracted_code = extractor(watermarked)
            target_code = universal_code.unsqueeze(0).expand(extracted_code.size(0), -1)
            loss_wm = wm_loss_fn(extracted_code, target_code)

            loss = loss_recon + loss_wm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_wm += loss_wm.item()

            if (batch_idx % 50) == 0:
                print(f"Epoch[{epoch}/{epochs}], Batch {batch_idx+1}/{len(loader)} "
                      f"Recon: {loss_recon.item():.4f}, WM: {loss_wm.item():.4f}")

                # Save sample watermarked image from 2nd epoch onward
                if epoch >= 2:
                    saved_img = watermarked[0].detach().cpu()
                    original_path = full_paths[0]
                    out_path = save_watermarked_image(saved_img, original_path, data_dir, target_dir, epoch)
                    print(f"  -> Saved watermarked image: {out_path}")

        avg_recon = total_recon / len(loader)
        avg_wm = total_wm / len(loader)
        print(f"=> Epoch {epoch}/{epochs} done. Recon: {avg_recon:.4f} | WM: {avg_wm:.4f}")

    print("[MaxViT Watermarker] Training complete!")


#####################################################
# 3) MAIN: Train Classifier -> Freeze -> Watermark
#####################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    #  A) Classifier Training
    # -------------------------------------------------------------------------
    classifier_train_dir = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Training"
    classifier_test_dir  = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Testing"
    classifier_path      = "efficientnet_b5_classifier.pth"

    # Train the classifier if not found, or skip if found
    b3_classifier = train_efficientnet_b3_classifier(
        train_root=classifier_train_dir,
        test_root=classifier_test_dir,
        model_path=classifier_path,
        epochs=10,    # 10 epochs
        lr=1e-5       # learning rate
    )

    # Freeze the classifier (if we have it in memory)
    if b3_classifier is not None:
        print("[Main] Freezing classifier parameters now.")
        for param in b3_classifier.parameters():
            param.requires_grad = False

    # -------------------------------------------------------------------------
    #  B) Watermarking with MaxViT
    # -------------------------------------------------------------------------
    data_dir   = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset"
    target_dir = r"E:\watermarking\MaxVItNoDescrim"

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

    print("[Main] Pipeline complete! Classifier is trained & frozen, MaxViT Watermarking done.")
