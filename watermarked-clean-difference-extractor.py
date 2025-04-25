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

    # 2) Absolute diff
    diff = ImageChops.difference(wm, orig)

    # 3) White canvas
    white = Image.new("RGB", orig.size, "white")

    # 4) Full grayscale diff (may be faint if your delta is small)
    mask_all = diff.convert("L").point(lambda px: 255 if px > 0 else 0)
    extracted = white.copy()
    extracted.paste(diff, mask=mask_all)

    # 5) Per‑channel outline masks & tinted borders
    channels = diff.split()  # R, G, B
    tints    = [(255,0,0), (0,255,0), (0,0,255)]
    channel_outlines = []

    for chan, tint in zip(channels, tints):
        # binary mask of where this channel changed
        mask_ch = chan.point(lambda px: 255 if px > 0 else 0)
        # erode it by one pixel
        eroded  = mask_ch.filter(ImageFilter.MinFilter(3))
        # subtract to leave only the 1‑px border
        outline = ImageChops.subtract(mask_ch, eroded)

        # paste a solid‑color layer through that border
        color_layer = Image.new("RGB", orig.size, tint)
        img = white.copy()
        img.paste(color_layer, mask=outline)
        channel_outlines.append(img)

    # 6) Assemble 2×3 grid: [orig, wm, extracted, R‑outline, G‑outline, B‑outline]
    w, h = orig.size
    canvas = Image.new("RGB", (2*w, 3*h), "white")
    imgs   = [orig, wm, extracted] + channel_outlines
    positions = [
        (0,   0), (w,   0),
        (0,   h), (w,   h),
        (0, 2*h), (w, 2*h),
    ]
    for img, pos in zip(imgs, positions):
        canvas.paste(img, pos)

    # 7) Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{orig_path.stem}_{epoch_name}_{timestamp}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / fname)
    print(f"Saved → {out_dir/ fname}")

def main():
    epoch0 = INPUT_ROOT / "epoch0"
    if not epoch0.exists():
        raise RuntimeError(f"Could not find epoch0 at {epoch0}")

    # walk each class folder under epoch0
    for cls_dir in epoch0.iterdir():
        if not cls_dir.is_dir(): continue
        cls = cls_dir.name

        # each original in epoch0/<class>
        for orig_img in cls_dir.iterdir():
            if not orig_img.is_file(): continue
            stem = orig_img.stem

            # scan every epochX
            for epoch_dir in INPUT_ROOT.iterdir():
                if not (epoch_dir.is_dir() and epoch_dir.name.startswith("epoch")):
                    continue

                wm_folder = epoch_dir / cls
                if not wm_folder.exists():
                    continue

                # find the matching watermarked file
                match = next(
                    (f for f in wm_folder.iterdir()
                     if f.is_file() and f.stem.startswith(stem)),
                    None
                )
                if match is None:
                    print(f"[WARN] no match for {stem} in {epoch_dir.name}/{cls}")
                    continue

                out_dir = OUTPUT_ROOT / epoch_dir.name / cls
                extract_and_save(orig_img, match, out_dir, epoch_dir.name)

if __name__ == "__main__":
    main()
