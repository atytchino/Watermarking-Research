from pathlib import Path
from PIL import Image, ImageChops, ImageFilter
from datetime import datetime

# === CONFIGURATION ===
INPUT_ROOT  = Path(r"E:\watermarking\afhq\watermarked_clean_by_epoch")
OUTPUT_ROOT = Path(r"E:\watermarking\afhq\extracted_watermarks")


def extract_and_save(orig_path: Path, wm_path: Path, out_dir: Path, epoch_name: str):
    # 1) Load images
    orig = Image.open(orig_path).convert("RGB")
    wm   = Image.open(wm_path).convert("RGB")

    # 2) Compute absolute diff
    diff = ImageChops.difference(wm, orig)
    # 2.1) boost visibility by scaling pixel differences (tweak scale as needed)
    diff = diff.point(lambda px: min(255, int(px * 8)))
    w, h = orig.size

    # 3) White canvas + global mask
    white    = Image.new("RGB", (w, h), "white")
    mask_all = diff.convert("L").point(lambda px: 255 if px > 0 else 0)

    # 4) Colored watermark overlay (RGB diff on white)
    color_overlay = white.copy()
    color_overlay.paste(diff, mask=mask_all)

    # 5) Grayscale difference (for reference)
    gray      = diff.convert("L")
    gray_rgb  = Image.merge("RGB", (gray, gray, gray))
    extracted_gray = white.copy()
    extracted_gray.paste(gray_rgb, mask=mask_all)

    # 6) Per-channel tinted outlines (1px border)
    channels = diff.split()
    tints    = [(255,0,0), (0,255,0), (0,0,255)]
    outlines = []
    for chan, tint in zip(channels, tints):
        mask_ch = chan.point(lambda px: 255 if px > 0 else 0)
        eroded  = mask_ch.filter(ImageFilter.MinFilter(3))
        outline = ImageChops.subtract(mask_ch, eroded)
        layer   = Image.new("RGB", (w, h), tint)
        img     = white.copy()
        img.paste(layer, mask=outline)
        outlines.append(img)

    # 7) Build 2×3 grid:
    #    Row 1: Original | Watermarked | Difference (color overlay)
    #    Row 2: Red diff  | Green diff   | Blue diff
    canvas = Image.new("RGB", (3*w, 2*h), "white")
    imgs   = [orig, wm, color_overlay] + outlines
    positions = [
        (0,   0), (w,   0), (2*w,   0),  # top row
        (0,   h), (w,   h), (2*w,   h),  # bottom row
    ]
    for img, pos in zip(imgs, positions):
        canvas.paste(img, pos)

    # 8) Save with timestamp and epoch
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{orig_path.stem}_{epoch_name}_{ts}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / fname)


def main():
    epoch0 = INPUT_ROOT / "epoch0"
    if not epoch0.exists():
        raise FileNotFoundError(f"Could not find epoch0 at {epoch0}")

    for cls_dir in epoch0.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name

        for orig_img in cls_dir.iterdir():
            if not orig_img.is_file():
                continue
            stem = orig_img.stem

            for epoch_dir in INPUT_ROOT.iterdir():
                if not (epoch_dir.is_dir() and epoch_dir.name.startswith("epoch")):
                    continue

                wm_folder = epoch_dir / cls
                if not wm_folder.exists():
                    continue

                # Match by stem prefix
                match = next(
                    (f for f in wm_folder.iterdir() if f.is_file() and f.stem.startswith(stem)),
                    None
                )
                if match is None:
                    continue

                out_dir = OUTPUT_ROOT / epoch_dir.name / cls
                extract_and_save(orig_img, match, out_dir, epoch_dir.name)

if __name__ == "__main__":
    main()
