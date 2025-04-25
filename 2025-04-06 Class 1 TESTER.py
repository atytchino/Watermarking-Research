from PIL import Image
import pytest
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b6
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random, os, glob
import torch
from torch.utils.data import DataLoader


# -------------------------------------------------------------------
# 1) CUSTOM DATASET FOR THE RANDOM SAMPLES
# -------------------------------------------------------------------
class RandomSubsetDataset(Dataset):
    """
    A simple dataset that expects a list of (image_path, class_index) tuples,
    plus a transform function.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# -------------------------------------------------------------------
# 2) FIXTURES FOR PATHS
# -------------------------------------------------------------------
@pytest.fixture
def model_path():
    return r"E:\watermarking\afhq\trained_model_class1\efficientnet_b6_classifier.pth"

@pytest.fixture
def test_dir():
    return r"E:\watermarking\afhq\test"

# -------------------------------------------------------------------
# 3) TEST FUNCTION: LOAD MODEL, INFER ON RANDOM 100 IMAGES PER CLASS
# -------------------------------------------------------------------



def test_classifier_on_random_images(
    model_path,
    test_dir,
    num_images_per_class=1000,
    batch_size=1,
    random_seed=42
):
    """
    1) Loads the trained EfficientNet-B6 model from `model_path`.
    2) Gathers up to `num_images_per_class` random images from each class folder in `test_dir`.
    3) Runs inference, prints accuracy, and shows a confusion matrix.

    NOTE: This checkpoint expects 3 classes in the final layer.
    """
    random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather class names
    class_names = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )

    # Build samples
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    samples = []
    for cls_name in class_names:
        cls_path = os.path.join(test_dir, cls_name)
        all_images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            all_images.extend(glob.glob(os.path.join(cls_path, ext)))
        random.shuffle(all_images)
        subset = all_images[:num_images_per_class]
        lbl = class_to_idx[cls_name]
        for img_path in subset:
            samples.append((img_path, lbl))

    print(f"[INFO] Found {len(samples)} total images across {len(class_names)} classes.")
    for cls_name in class_names:
        count_for_this_class = sum(1 for s in samples if s[1] == class_to_idx[cls_name])
        print(f"  -> Class '{cls_name}' has {count_for_this_class} images selected.")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Dataset / Dataloader
    dataset = RandomSubsetDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build the EfficientNet-B6 model
    # ----------------------------------------------------------
    # IMPORTANT: Force final layer to have 3 outputs (matching your checkpoint)
    # ----------------------------------------------------------
    print(f"[INFO] Loading model from {model_path}")
    model = efficientnet_b6(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)  # 3 classes

    # Optionally wrap in DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel before load.")
        model = nn.DataParallel(model)

    # Load the checkpoint
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    print("[INFO] Running inference on the selected images...")
    all_labels = []
    all_preds = []
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()

    total = len(dataset)
    accuracy = correct / total
    print(f"\n[RESULT] Tested on {total} images total. Accuracy = {accuracy:.4f}")


# -------------------------------------------------------------------
# 4) MAIN
# -------------------------------------------------------------------
def test_classifier_on_random_images(
    model_path,
    test_dir,
    num_images_per_class=1000,
    batch_size=1,
    random_seed=42
):
    """
    1) Loads the trained EfficientNet-B6 model from `model_path`.
    2) Gathers up to `num_images_per_class` random images from each class folder in `test_dir`.
    3) Runs inference, prints *per-class* accuracy and a confusion matrix.

    NOTE: This checkpoint expects 3 classes in the final layer.
    """


    random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather class names
    class_names = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )

    # Build samples
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    samples = []
    for cls_name in class_names:
        cls_path = os.path.join(test_dir, cls_name)
        all_images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            all_images.extend(glob.glob(os.path.join(cls_path, ext)))
        random.shuffle(all_images)
        subset = all_images[:num_images_per_class]
        lbl = class_to_idx[cls_name]
        for img_path in subset:
            samples.append((img_path, lbl))

    print(f"[INFO] Found {len(samples)} total images across {len(class_names)} classes.")
    for cls_name in class_names:
        count_for_this_class = sum(1 for s in samples if s[1] == class_to_idx[cls_name])
        print(f"  -> Class '{cls_name}' has {count_for_this_class} images selected.")

    # Data transforms
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Use your RandomSubsetDataset
    dataset = RandomSubsetDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build the EfficientNet-B6 model (3 outputs)
    from torchvision.models import efficientnet_b6
    import torch.nn as nn

    print(f"[INFO] Loading model from {model_path}")
    model = efficientnet_b6(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))  # 3 classes

    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel before load.")
        model = nn.DataParallel(model)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    print("[INFO] Running inference on the selected images...")

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # -----------------------------
    # 1) Compute per-class accuracy
    # -----------------------------
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    for lbl, prd in zip(all_labels, all_preds):
        class_total[lbl] += 1
        if lbl == prd:
            class_correct[lbl] += 1

    print("\n[RESULT] Accuracy per class:")
    for i, cls_name in enumerate(class_names):
        if class_total[i] == 0:
            acc = 0.0
        else:
            acc = class_correct[i] / class_total[i]
        print(f"  - Class '{cls_name}': {acc:.4f} ({class_correct[i]}/{class_total[i]})")
        if class_correct[i]/class_total[i] < 0.9:
            print("At least one class has accuracy level below 90%. Please re-train the classifier")
            exit(1)

    # -----------------------------
    # 2) Display Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(all_labels, all_preds)
    print("\n[INFO] Confusion Matrix:")
    print(cm)



def main():
    # Paths / parameters
    model_path = r"E:\watermarking\afhq\trained_model_class1\efficientnet_b6_classifier.pth"
    test_dir   = r"E:\watermarking\afhq\test"
    num_images_per_class = 1000
    batch_size = 1
    random_seed = 42

    # Run test
    test_classifier_on_random_images(
        model_path=model_path,
        test_dir=test_dir,
        num_images_per_class=num_images_per_class,
        batch_size=batch_size,
        random_seed=random_seed
    )

if __name__ == "__main__":
    main()
