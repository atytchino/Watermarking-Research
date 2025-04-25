from pathlib import Path
import shutil

# --- CONFIGURATION ---
INPUT_ROOT   = Path(r"E:\watermarking\afhq\epoch_images")
CLEAN_ROOT   = Path(r"E:\watermarking\afhq\tiny_train")
OUTPUT_ROOT  = Path(r"E:\watermarking\afhq\watermarked_clean_by_epoch")

# IDs to match per class
PATTERNS = {
    "cat":  ["000112", "000348", "000406", "000343", "0004"],
    "dog":  ["00006",  "00043",  "00101",  "000037", "000053"],
    "wild": ["000145", "00035",  "000447", "00058",  "001075"]
}

def copy_epoch_files():
    """Copy matching files from each epochX/class subfolder."""
    for epoch_dir in INPUT_ROOT.iterdir():
        if not epoch_dir.is_dir() or not epoch_dir.name.startswith("epoch"):
            continue

        for cls, ids in PATTERNS.items():
            src_folder = epoch_dir / cls
            if not src_folder.exists():
                print(f"[WARN] missing folder: {src_folder}")
                continue

            dst_folder = OUTPUT_ROOT / epoch_dir.name / cls
            dst_folder.mkdir(parents=True, exist_ok=True)

            for img_path in src_folder.iterdir():
                if not img_path.is_file():
                    continue
                if any(pid in img_path.name for pid in ids):
                    dst_file = dst_folder / img_path.name
                    if dst_file.exists():
                        # skip existing
                        continue
                    shutil.copy2(img_path, dst_file)
                    print(f"Copied {img_path.name} → {dst_folder}")

def copy_clean_files_to_epoch0():
    """Copy matching clean images into epoch0/class subfolders."""
    epoch0_folder = OUTPUT_ROOT / "epoch0"
    for cls, ids in PATTERNS.items():
        src_folder = CLEAN_ROOT / cls
        if not src_folder.exists():
            print(f"[WARN] missing clean folder: {src_folder}")
            continue

        dst_folder = epoch0_folder / cls
        dst_folder.mkdir(parents=True, exist_ok=True)

        for img_path in src_folder.iterdir():
            if not img_path.is_file():
                continue
            if any(pid in img_path.name for pid in ids):
                dst_file = dst_folder / img_path.name
                if dst_file.exists():
                    # skip existing
                    continue
                shutil.copy2(img_path, dst_file)
                print(f"Copied clean {img_path.name} → {dst_folder}")

if __name__ == "__main__":
    copy_epoch_files()
    copy_clean_files_to_epoch0()
    print("Done.")
