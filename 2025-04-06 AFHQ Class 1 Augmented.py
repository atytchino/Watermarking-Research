import os
import glob
import random
import shutil
import datetime
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights, efficientnet_b6, EfficientNet_B6_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b6
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################

def count_images_in_directory(base_dir):
    """
    Counts images in `base_dir` for each subfolder (class).
    Returns dict: class_name -> count
    """
    class_counts = {}
    if not os.path.exists(base_dir):
        return class_counts
    for cls_name in os.listdir(base_dir):
        cls_path = os.path.join(base_dir, cls_name)
        if os.path.isdir(cls_path):
            num_images = 0
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                num_images += len(glob.glob(os.path.join(cls_path, ext)))
            class_counts[cls_name] = num_images
    return class_counts

def maybe_create_test_set(train_dir, test_dir, fraction=0.33):
    """
    If `test_dir` does not exist, create it by moving ~33% of images
    from each class in `train_dir`.
    Otherwise, do nothing.
    """
    if os.path.exists(test_dir):
        print(f"[INFO] Test folder '{test_dir}' already exists. Skipping test creation.")
        return

    print(f"[INFO] Creating test set in '{test_dir}' by moving {int(fraction*100)}% of training images...")
    os.makedirs(test_dir, exist_ok=True)

    for cls_name in os.listdir(train_dir):
        cls_train_dir = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_train_dir):
            cls_test_dir = os.path.join(test_dir, cls_name)
            os.makedirs(cls_test_dir, exist_ok=True)

            # Gather images
            all_images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                all_images.extend(glob.glob(os.path.join(cls_train_dir, ext)))

            random.shuffle(all_images)
            test_count = int(len(all_images) * fraction)
            images_to_move = all_images[:test_count]

            for img_path in images_to_move:
                img_name = os.path.basename(img_path)
                new_path = os.path.join(cls_test_dir, img_name)
                shutil.move(img_path, new_path)

    print("[INFO] Test set creation complete.")

def plot_loss_curves(train_losses, val_losses, save_path=None):
    """
    Plots train & val losses vs. epoch. Optionally saves the figure if save_path is given.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved loss plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_confusion(labels, preds, class_names, title="Confusion Matrix", save_path=None):
    """
    Displays a confusion matrix with scikit-learn.
    """
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, values_format='d', cmap="Blues", colorbar=False)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()

###############################################################################
# 2) TRAINING THE EFFICIENTNET-B6 CLASSIFIER
###############################################################################

def train_efficientnet_b6_classifier(
    train_dir,
    val_dir,
    test_dir,
    model_save_path,
    epochs=5,
    batch_size=8,
    lr=1e-4,
    plot_dir=None
):
    """
    Trains an EfficientNet-B6 classifier for cat/dog/wild at 512x512.
    - Saves the best model to `model_save_path`.
    - Plots train/val loss curves.
    - At the end, evaluates on the entire test set and plots confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training EfficientNet-B6 classifier on device: {device}")

    # ================== 8 AUGMENTATIONS FOR TRAIN ===================
    transform_train = transforms.Compose([
        # 1) RandomResizedCrop: randomly crops & scales the image
        transforms.RandomResizedCrop((512, 512), scale=(0.8, 1.0)),
        # 2) RandomHorizontalFlip
        transforms.RandomHorizontalFlip(p=0.55),
        # 3) RandomVerticalFlip
        transforms.RandomVerticalFlip(p=0.35),
        # 4) RandomRotation: rotates the image randomly between 0 to 30 degrees
        transforms.RandomRotation(degrees=(0, 30)),
        # 5) ColorJitter: random adjustments within specified ranges
        transforms.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1)
        ),
        # 6) RandomGrayscale
        transforms.RandomGrayscale(p=0.35),
        # 7) Random Gaussian Blur (using RandomApply)
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # 8) RandomErasing
        transforms.RandomErasing(p=0.25),
        ])
    # ================== EVAL/TEST TRANSFORMS (no heavy augment) ==================
    transform_eval = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # Datasets / Dataloaders
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds   = torchvision.datasets.ImageFolder(val_dir,   transform=transform_eval)
    test_ds  = torchvision.datasets.ImageFolder(test_dir,  transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Print data info
    print(f"[INFO] #Train = {len(train_ds)}, #Val = {len(val_ds)}, #Test = {len(test_ds)}")
    print("[INFO] Classes:", train_ds.classes)

    # Model
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")
        # ------------------- TRAIN -------------------
        model.train()
        running_loss, running_corrects = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = running_corrects / len(train_loader.dataset)
        print(f"  [Train] Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")

        # ------------------- VALIDATE -------------------
        model.eval()
        val_loss, val_corrects = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels).item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc  = val_corrects / len(val_loader.dataset)
        print(f"  [Val]   Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Checkpoint if best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Saving best model (val_loss={best_val_loss:.4f})")

    # ------------------- Plot Loss Curves -------------------
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        loss_plot_path = os.path.join(plot_dir, "train_val_loss.png")
        plot_loss_curves(train_losses, val_losses, save_path=loss_plot_path)
    else:
        plot_loss_curves(train_losses, val_losses)

    # ------------------- Evaluate on Test Set (Full) -------------------
    # Load best checkpoint
    best_model = efficientnet_b6(pretrained=False)
    best_model.classifier[1] = nn.Linear(num_features, 3)
    best_model.load_state_dict(torch.load(model_save_path))
    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(best_model)
    best_model = best_model.to(device)
    best_model.eval()

    test_corrects = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = best_model(images)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels).item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = test_corrects / len(test_ds)
    print(f"[TEST] Accuracy: {test_acc:.4f}")

    # Plot confusion matrix for the entire test set
    if plot_dir:
        cm_path = os.path.join(plot_dir, "test_confusion_matrix_full.png")
        plot_confusion(all_labels, all_preds, train_ds.classes,
                       title="Test Set Confusion Matrix", save_path=cm_path)
    else:
        plot_confusion(all_labels, all_preds, train_ds.classes, title="Test Set Confusion Matrix")


###############################################################################
# 3) MAIN SCRIPT
###############################################################################

def main():
    random.seed(42)
    torch.manual_seed(42)

    # Folder paths
    train_dir = r"E:\\watermarking\\afhq\\train"
    val_dir   = r"E:\\watermarking\\afhq\\val"
    test_dir  = r"E:\\watermarking\\afhq\\test"
    model_dir = r"E:\\watermarking\\afhq\\trained_model_class1"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "efficientnet_b6_classifier.pth")

    # We'll define a plot directory for saving figures
    plot_dir = os.path.join(model_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Maybe create test set
    if not os.path.exists(test_dir):
        maybe_create_test_set(train_dir, test_dir, fraction=0.33)

    # 2) Check if training is needed
    test_exists = os.path.exists(test_dir)
    model_exists = os.path.isfile(model_path)

    if test_exists and model_exists:
        print("[INFO] Model file & test folder found. Skipping training.")
        print("[INFO] The classifier is presumably trained. Exiting.")
    else:
        print("[INFO] Starting training of EfficientNet-B6 classifier for 5 epochs...")
        train_efficientnet_b6_classifier(
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            model_save_path=model_path,
            epochs=5,
            batch_size=8,   # Adjust to your GPU
            lr=1e-4,
            plot_dir=plot_dir
        )
        print("[INFO] Training & Evaluation Complete.")

if __name__ == "__main__":
    main()
