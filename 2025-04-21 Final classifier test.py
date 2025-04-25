import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

# ——— Rebuild exactly the same Classifier 2 architecture ———
def build_classifier2(num_classes=3):
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model

# ——— Load with DataParallel first, then load_state_dict “as‑is” ———
def load_classifier2(path, num_classes=3):
    # 1) instantiate
    model = build_classifier2(num_classes)

    # 2) wrap for true multi‑GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 3) load the saved state (with “module.” prefixes intact!)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)

    return model

# ——— Standard multi‑GPU‐aware evaluation ———
def evaluate(model, data_dir, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    ds     = ImageFolder(data_dir, transform=transform)
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total

# ——— Main ———
if __name__ == "__main__":
    final_c2_path = r"E:\watermarking\afhq\class2_model\imagenetb6_class2.pth"
    num_classes   = 3

    # load + parallelize
    model = load_classifier2(final_c2_path, num_classes)

    # evaluate on both datasets
    wm_acc    = evaluate(model, r"E:\watermarking\afhq\epoch_images_2025-04-21-05-00\Epoch4")
    clean_acc = evaluate(model, r"E:\watermarking\afhq\tiny_train")

    print(f"Watermarked accuracy: {wm_acc:.4f}")
    print(f"Clean accuracy:       {clean_acc:.4f}")
