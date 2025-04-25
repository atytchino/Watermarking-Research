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
# 1) CLASSIFIER CODE (unchanged)
############################################
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b5, EfficientNet_B5_Weights

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
    Same code as yours, using B5, returning an MLP head with 4 classes.
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from keys if the model was trained with DataParallel
    and is being loaded into a non-DataParallel model.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd

def train_efficientnet_b3_classifier(
    train_root,
    test_root,
    model_path="efficientnet_b5_classifier.pth",
    epochs=15,
    lr=1e-5,
    batch_size=4,
    num_workers=2
):
    if os.path.isfile(model_path):
        print(f"[Classifier] Found {model_path}, loading it instead of training.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Create the same model architecture
        model = create_efficientnet_b3(num_classes=4)

        # 2) Load the checkpoint
        checkpoint_sd = torch.load(model_path, map_location=device)

        # 3) Remove "module." from keys if present
        checkpoint_sd = remove_module_prefix(checkpoint_sd)

        # 4) Load into model
        model.load_state_dict(checkpoint_sd)

        model.to(device)
        return model

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

    train_dataset = MRI4ClassDataset(train_root, transform=train_transform)
    test_dataset  = MRI4ClassDataset(test_root,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_efficientnet_b3(num_classes=4)
    if torch.cuda.device_count() > 1:
        print(f"[Classifier] Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    torch.save(model.state_dict(), model_path)
    print(f"[Classifier] Saved final model to {model_path}")
    return model

##########################################
# 2) DATASET FOR WATERMARKING w/ LABELS
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
    return os.path.join(final_subdir, new_filename)

class LabeledFilenameDataset(Dataset):
    """
    Returns (image, label, full_path) so we can:
      - measure classification accuracy on watermarked images
      - replicate subfolder structure for saving
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
        for c in self.classes:
            cdir = os.path.join(root_dir, c)
            if not os.path.isdir(cdir):
                continue
            label_idx = self.class_to_idx[c]
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                for f in glob.glob(os.path.join(cdir, f"*{ext}")):
                    self.image_paths.append(f)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path

def save_watermarked_image(img_tensor, original_img_path, data_root, target_root, epoch):
    final_path = get_output_path(original_img_path, data_root, target_root, epoch)
    clamped = img_tensor.clamp(0,1)
    vutils.save_image(clamped, final_path)
    return final_path

##########################################
# 3) MAXVIT, BLOCKS, EXTRACTOR, etc.
##########################################
# (Paste your existing classes: MultiHeadSelfAttention, block_attention, grid_attention,
#  FeedForward, MaxViTBlock, MaxViT, DoubleConv, DownBlock, UpBlock, WatermarkExtractor,
#  UNetWithRealMaxViT)
#
# For brevity, we won't re-paste them here. We'll assume they match your code EXACTLY.
#
# e.g.:
# class MultiHeadSelfAttention(nn.Module): ...
# class MaxViTBlock(nn.Module): ...
# class UNetWithRealMaxViT(nn.Module): ...
# etc.

##########################################
# 4) White-Box Watermark w/ Classifier Feedback
##########################################

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

def train_unet_real_maxvit_feedback(
    data_dir,
    target_dir,
    classifier,            # frozen classifier
    epochs=5,
    batch_size=2,
    lr=1e-5,
    code_dim=128,
    model_params=None
):
    """
    - Labeled dataset => measure classifier accuracy on watermarked images each epoch
    - If accuracy > 55%, apply confusion loss (negative cross-entropy) to degrade it
      to 45-55% range.
    - Otherwise, skip confusion loss, only do recon + watermark extraction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Freeze classifier
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False
    classifier.to(device)

    # Build model
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
        print(f"[WatermarkFeedback] Using {torch.cuda.device_count()} GPUs with nn.DataParallel!")
        model = nn.DataParallel(model)
        extractor = nn.DataParallel(extractor)

    model.to(device)
    extractor.to(device)

    # Labeled dataset
    transform = T.Compose([T.Resize((512,512)), T.ToTensor()])
    dataset = LabeledFilenameDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Single universal watermark code
    universal_code = torch.randn(code_dim, device=device)

    # Loss functions
    recon_loss_fn = nn.MSELoss()
    wm_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()  # for negative cross-entropy => confusion

    optimizer = optim.Adam(list(model.parameters()) + list(extractor.parameters()), lr=lr)

    # Helper to measure classifier accuracy on entire set (watermarked)
    def measure_accuracy():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (imgs, labels, paths) in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                wimgs = model(imgs)
                logits = classifier(wimgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0


    print("[WatermarkFeedback] Starting training with classifier feedback (45-55% goal).")
    consecutive_count = 0
    for epoch in range(1, epochs+1):
        model.train()
        extractor.train()

        total_recon = 0.0
        total_wm = 0.0
        total_conf = 0.0
        total_batches = 0

        # We'll measure batch-based accuracy to see if we degrade
        batch_correct = 0
        batch_total = 0

        for batch_idx, (imgs, labels, full_paths) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            wimgs = model(imgs)  # generate watermarked images

            # Reconstruction
            loss_recon = recon_loss_fn(wimgs, imgs)

            # Watermark extraction
            extracted = extractor(wimgs)
            target_code = universal_code.unsqueeze(0).expand(extracted.size(0), -1)
            loss_wm = wm_loss_fn(extracted, target_code)

            # We'll measure the current accuracy on this batch
            logits = classifier(wimgs)
            preds = torch.argmax(logits, dim=1)
            batch_correct += (preds == labels).sum().item()
            batch_total += labels.size(0)

            # Confusion: only if accuracy > 55%
            # We'll estimate batch_acc just for a quick check
            batch_acc = (preds == labels).float().mean().item() * 100.0

            # Negative cross-entropy => degrade classifier
            if batch_acc > 55.0:
                # apply confusion
                ce = ce_loss_fn(logits, labels)
                confusion_loss = -1.0 * ce
            else:
                confusion_loss = torch.tensor(0.0, device=device)

            total_loss = loss_recon + loss_wm + confusion_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_recon += loss_recon.item()
            total_wm += loss_wm.item()
            total_conf += confusion_loss.item()
            total_batches += 1

            # Optionally save a sample watermarked image
            if (batch_idx % 50) == 0:
                print(f"Epoch[{epoch}/{epochs}], Batch {batch_idx+1}/{len(loader)} "
                      f"Recon={loss_recon.item():.4f}, WM={loss_wm.item():.4f}, Conf={confusion_loss.item():.4f}")
                if epoch >= 2:
                    saved_img = wimgs[0].detach().cpu()
                    out_path = save_watermarked_image(saved_img, full_paths[0], data_dir, target_dir, epoch)
                    print(f"  -> Saved watermarked image: {out_path}")

        avg_recon = total_recon / total_batches
        avg_wm = total_wm / total_batches
        avg_conf = total_conf / total_batches
        batch_overall_acc = 100.0 * batch_correct / batch_total

        # Evaluate full accuracy
        test_acc = measure_accuracy() * 100.0

        print(f"=> Epoch {epoch}/{epochs} done. "
              f"Recon={avg_recon:.4f} | WM={avg_wm:.4f} | Conf={avg_conf:.4f} | "
              f"BatchAcc={batch_overall_acc:.2f}% | FullAcc={test_acc:.2f}%")



        if 45.0 <= test_acc <= 55.0:
            consecutive_count += 1
            if consecutive_count >= 2:  #  2 epochs stable
                print("[Stopping] Accuracy stable in [45-55]% for 2 epochs.")
                break
        else:
            consecutive_count = 0

        print("[WatermarkFeedback] Done. Final accuracy:", measure_accuracy())



    print("[WatermarkFeedback] Training complete! Now your classifier should be in 45-55% range (if successful).")
    return model  # optionally return extractor as well


#####################################################
# 5) MAIN
#####################################################
if __name__ == "__main__":
    # 1) Train or load classifier
    classifier_train_dir = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Training"
    classifier_test_dir  = r"D:\Dropbox\UMA Augusta\PhD\Research Thesis\brain_tumor_mri_dataset\Testing"
    classifier_path      = "efficientnet_b5_classifier.pth"

    b5_classifier = train_efficientnet_b3_classifier(
        train_root=classifier_train_dir,
        test_root=classifier_test_dir,
        model_path=classifier_path,
        epochs=15,
        lr=1e-5

    )
    if b5_classifier is not None:
        # Freeze it
        print("[Main] Freezing classifier parameters now.")
        for param in b5_classifier.parameters():
            param.requires_grad = False

    # 2) Watermark with Feedback
    if b5_classifier is not None:
        data_dir   = r"D:\\Dropbox\\UMA Augusta\\PhD\\Research Thesis\\brain_tumor_mri_dataset\\Testing"
        target_dir = r"E:\\watermarking\\MaxVItNoDescrim"

        # Now do the feedback-based approach
        model = train_unet_real_maxvit_feedback(
            data_dir=data_dir,
            target_dir=target_dir,
            classifier=b5_classifier,   # pass the frozen classifier
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
        print("[Main] White-box feedback watermarking complete!")
    else:
        print("[Main] No classifier loaded. Skipping watermark feedback step.")

    print("[Main] Pipeline complete!")
