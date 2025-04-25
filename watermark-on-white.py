from pathlib import Path
from PIL import Image, ImageChops
from datetime import datetime

# === CONFIGURATION ===
# Originals are in extracted_watermarks/epoch0
ORIG_ROOT   = Path(r"E:\watermarking\afhq\extracted_watermarks\epoch0")
# All epochN folders with watermarked images are in extracted_watermarks (except epoch0 and grids)
WM_ROOT     = Path(r"E:\watermarking\afhq\extracted_watermarks")
# Output grids into a subfolder 'grids' under extracted_watermarks
GRID_ROOT   = WM_ROOT / "grids"


def extract_and_save(orig_path: Path, wm_path: Path, out_dir: Path, epoch_name: str):
    """
    Build and save a 3×2 grid:
      Top row:    Original | Watermarked | Difference (grayscale)
      Bottom row: Red diff  | Green diff   | Blue diff
    """
    orig = Image.open(orig_path).convert("RGB")
    wm   = Image.open(wm_path).convert("RGB")

    # compute absolute-difference
    diff = ImageChops.difference(wm, orig)
    w, h = orig.size

    # prepare white canvas and mask
    white    = Image.new("RGB", (w, h), "white")
    mask_all = diff.convert("L").point(lambda px: 255 if px > 0 else 0)

    # grayscale difference on white
    extracted = white.copy()
    extracted.paste(diff, mask=mask_all)

    # per-channel colored diffs on white
    channel_imgs = []
    for tint, chan in zip([(255,0,0), (0,255,0), (0,0,255)], diff.split()):
        mask_ch     = chan.point(lambda px: 255 if px > 0 else 0)
        color_layer = Image.new("RGB", (w, h), tint)
        img         = white.copy()
        img.paste(color_layer, mask=mask_ch)
        channel_imgs.append(img)

    # assemble 3×2 grid: Original, Watermarked, Difference; Red, Green, Blue
    canvas = Image.new("RGB", (3*w, 2*h), "white")
    elements = [orig, wm, extracted] + channel_imgs
    positions = [(i%3 * w, i//3 * h) for i in range(6)]
    for img, pos in zip(elements, positions):
        canvas.paste(img, pos)

    # save grid
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{orig_path.stem}_{epoch_name}_{ts}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / fname)


def main():
    # build map of original stems per class
    orig_map = {}
    for cls_dir in ORIG_ROOT.iterdir():
        if cls_dir.is_dir():
            orig_map[cls_dir.name] = {p.stem: p for p in cls_dir.iterdir() if p.is_file()}

    # process each epoch folder except epoch0 and grids
    for epoch_dir in WM_ROOT.iterdir():
        if not (epoch_dir.is_dir() and epoch_dir.name.startswith("epoch") and epoch_dir.name != "epoch0"):
            continue
        epoch_name = epoch_dir.name

        for cls_dir in epoch_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            class_name = cls_dir.name
            class_orig = orig_map.get(class_name, {})

            # for each watermarked image, skip those without a matching original
            for wm_path in cls_dir.iterdir():
                if not wm_path.is_file():
                    continue
                # derive original stem before '_epoch'
                stem = wm_path.stem.split("_epoch", 1)[0]
                orig_path = class_orig.get(stem)
                if not orig_path:
                    continue  # skip unmatched

                grid_dir = GRID_ROOT / epoch_name / class_name
                extract_and_save(orig_path, wm_path, grid_dir, epoch_name)

if __name__ == "__main__":
    main()
