import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_classifier2(model_path, device, num_classes=3):
    # Build EfficientNet-B6 classifier
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    # Wrap for multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_dir, device, title):
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    all_preds, all_labels = [], []
    correct, total = 0,0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out,1)
            correct += (preds==labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = correct/total
    print(f"\n[{title}]  Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=dataset.classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"{title} Confusion Matrix (Acc={acc:.2%})")
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Path to your trained Classifier 2
    classifier2_path = r"E:\watermarking\afhq\class2_model\imagenetb6_class2.pth"
    model = load_classifier2(classifier2_path, device)

    # 1) Evaluate each epoch folder
    epoch_root = r"E:\watermarking\afhq\epoch_images"
    for epoch_dir in sorted(glob.glob(os.path.join(epoch_root, "epoch*"))):
        epoch_name = os.path.basename(epoch_dir)
        print(f"\n=== Evaluating epoch images: {epoch_name} ===")
        evaluate_model(model, epoch_dir, device, title=f"Epoch {epoch_name}")

    # 2) Evaluate the final watermarked dataset
    final_wm = r"E:\watermarking\afhq\full_final_watermarked"
    print("\n=== Evaluating final watermarked dataset ===")
    evaluate_model(model, final_wm, device, title="Final Watermarked Set")

    # 3) Evaluate the original test set
    test_dir = r"E:\watermarking\afhq\test"
    print("\n=== Evaluating original test images ===")
    evaluate_model(model, test_dir, device, title="Original Test Set")

if __name__ == "__main__":
    main()
