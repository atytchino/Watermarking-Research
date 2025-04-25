import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


# -------------------------------
# 1) Define Helper Blocks
# -------------------------------

class SimpleAttention(nn.Module):
    def __init__(self, dim=256, heads=8):
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
    def __init__(self, dim=256, heads=8, ws=8):
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
        # Attention sub-block
        x_ = x.view(B, H * W, C)
        x_ = self.ln1(x_)
        x_ = x_.view(B, H, W, C)
        x_ = block_attn(x_, self.attn, self.ws)
        x = x + x_
        # Feed-forward sub-block
        x_ = x.view(B, H * W, C)
        x_ = self.ln2(x_)
        x_ = self.ffn(x_)
        x_ = x_.view(B, H, W, C)
        return x + x_


# -------------------------------
# 2) Define the Classifier Architecture
# -------------------------------

class SimpleMaxViTClassifier(nn.Module):
    def __init__(self, num_classes=3, dim=256, heads=8, ws=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),  # (B,3,512,512) -> (B,dim,256,256)
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),  # (B,dim,256,256) -> (B,dim,128,128)
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),  # (B,dim,128,128) -> (B,dim,64,64)
            nn.ReLU()
        )
        self.block = SimpleMaxViTBlock(dim, heads, ws)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.stem(x)  # (B, dim, 64, 64)
        _, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.block(x)
        x = x.view(B, H * W, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)


# If the checkpoint was saved from a wrapper, define the wrapper:
class FinalMaxViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = SimpleMaxViTClassifier(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


# -------------------------------
# 3) Loading the Pretrained Model
# -------------------------------

def load_classifier(model_path, device):
    # Instantiate the same wrapper used during training.
    model = FinalMaxViTClassifier(num_classes=3)
    # Wrap in DataParallel for multi-GPU support.
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel before load.")
        model = nn.DataParallel(model)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# -------------------------------
# 4) Evaluation Function: Per-Class Accuracy and Confusion Matrix
# -------------------------------

def evaluate_classifier(model, data_loader, device, dataset_name):
    all_preds = []
    all_labels = []
    class_correct = {}
    class_total = {}

    for class_name in data_loader.dataset.classes:
        class_correct[class_name] = 0
        class_total[class_name] = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, dim=1)
            for i, label in enumerate(labels):
                class_name = data_loader.dataset.classes[label.item()]
                class_total[class_name] += 1
                if preds[i] == label:
                    class_correct[class_name] += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_acc = sum(class_correct[c] for c in class_correct) / sum(class_total[c] for c in class_total)
    print(f"\n[{dataset_name}] Overall Accuracy: {overall_acc:.4f}")
    print(f"[{dataset_name}] Per-Class Accuracy:")
    for class_name in data_loader.dataset.classes:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name]
            print(f"    {class_name}: {acc:.4f} ({class_correct[class_name]}/{class_total[class_name]})")

    print(f"\n[{dataset_name}] Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=data_loader.dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data_loader.dataset.classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"{dataset_name} Confusion Matrix (Acc={overall_acc:.2%})")
    plt.tight_layout()
    plt.show()


# -------------------------------
# 5) Main: Evaluate on Original & Watermarked Images
# -------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path to your saved classifier 2 checkpoint.
    classifier_path = r"E:\watermarking\afhq\class2_model\imagenetb6_class2.pth"

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Define dataset paths
    original_val_path = r"E:\watermarking\afhq\val"
    watermarked_path = r"E:\watermarking\afhq\full_final_watermarked"

    dataset_original = ImageFolder(original_val_path, transform=transform)
    dataset_watermarked = ImageFolder(watermarked_path, transform=transform)

    loader_original = DataLoader(dataset_original, batch_size=8, shuffle=False, num_workers=2)
    loader_watermarked = DataLoader(dataset_watermarked, batch_size=8, shuffle=False, num_workers=2)

    model = load_classifier(classifier_path, device)

    print("Evaluating on Original Validation Images:")
    evaluate_classifier(model, loader_original, device, "Original Validation")

    print("Evaluating on Watermarked Images:")
    evaluate_classifier(model, loader_watermarked, device, "Watermarked")


if __name__ == "__main__":
    main()
