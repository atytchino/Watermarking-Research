import os
import random

def delete_half_images(root_dir: str, seed: int = 42):
    """
    Traverse all subdirectories under root_dir. In each directory,
    randomly delete half of the image files.

    :param root_dir: top-level directory containing subfolders of images
    :param seed: random seed for reproducibility
    """
    # Define which extensions to treat as images
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    random.seed(seed)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter to image files only
        images = [f for f in filenames if os.path.splitext(f)[1].lower() in image_exts]
        if not images:
            continue

        # Compute how many to delete
        num_to_delete = len(images) // 2
        to_delete = random.sample(images, num_to_delete)

        print(f"In '{dirpath}': found {len(images)} images, deleting {num_to_delete} of them.")
        for fname in to_delete:
            full_path = os.path.join(dirpath, fname)
            try:
                os.remove(full_path)
                print(f"  • Deleted: {full_path}")
            except Exception as e:
                print(f"  ! Error deleting {full_path}: {e}")

if __name__ == "__main__":
    ROOT = r"E:\watermarking\afhq\tiny_train"
    # ⚠️ Be sure you have a backup before running!
    delete_half_images(ROOT, seed=12345)