from pathlib import Path
from PIL import Image, ImageChops, ImageFilter, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === CONFIGURATION ===
INPUT_ROOT  = Path(r"E:\watermarking\afhq\watermarked_clean_by_epoch")
OUTPUT_ROOT = Path(r"E:\watermarking\afhq\extracted_watermarks")

# Default font for labels
try:
    FONT = ImageFont.truetype("arial.ttf", size=14)
except IOError:
    FONT = ImageFont.load_default()


def extract_and_save(orig_path: Path, wm_path: Path, out_dir: Path, epoch_name: str, class_name: str):
    # Load images
    orig = Image.open(orig_path).convert("RGB")
    wm   = Image.open(wm_path).convert("RGB")
    # Compute diff and boost visibility
    diff = ImageChops.difference(wm, orig)
    diff = diff.point(lambda px: min(255, int(px * 8)))
    w, h = orig.size

    # White canvas + mask
    white = Image.new("RGB", (w, h), "white")
    mask  = diff.convert("L").point(lambda px: 255 if px > 0 else 0)

    # Overlay diff
    overlay = white.copy()
    overlay.paste(diff, mask=mask)
    # Generate RGB channel outlines
    outlines = []
    for chan, tint in zip(diff.split(), [(255,0,0),(0,255,0),(0,0,255)]):
        m = chan.point(lambda px: 255 if px > 0 else 0)
        er = m.filter(ImageFilter.MinFilter(3))
        border = ImageChops.subtract(m, er)
        layer = Image.new("RGB", (w, h), tint)
        img = white.copy()
        img.paste(layer, mask=border)
        outlines.append(img)

    # Prepare labels
    base_label = f"Epoch:{epoch_name} Class:{class_name} File:{orig_path.name}"
    main_labels = [
        f"{orig_path.name} (original)",
        f"Watermarked Image: {epoch_name} (watermarked)",
        f"Extracted Watermark: {epoch_name} (extracted watermark)",
        "RGB Red (watermark channel)",
        "RGB Green (watermark channel)",
        "RGB Blue (watermark channel)",
    ]

    # Grid dimensions with padding
    top_h = 40
    bot_h = 20
    canvas = Image.new("RGB", (3*w, 2*h + top_h + bot_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Place images and draw labels
    imgs = [orig, wm, overlay] + outlines
    for idx, img in enumerate(imgs):
        col = idx % 3
        row = idx // 3
        x = col * w
        y = top_h + row * h
        canvas.paste(img, (x, y))
        # Draw base label at top
        draw.text((x + 5, 2), base_label, fill="black", font=FONT)
        # Draw specific label
        label = main_labels[idx]
        if row == 0:
            # position under base within top_h
            bbox = draw.textbbox((x+5, 2), base_label, font=FONT)
            label_y = bbox[3] + 4
            draw.text((x + 5, label_y), label, fill="black", font=FONT)
        else:
            # position in bottom padding
            bbox = draw.textbbox((0,0), label, font=FONT)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            lx = x + (w - tw)//2
            ly = y + h + 2
            draw.text((lx, ly), label, fill="black", font=FONT)

    # Save main grid
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    main_fname = f"{orig_path.stem}_{epoch_name}_grid_{ts}.png"
    canvas.save(out_dir / main_fname)

    # Prepare grayscale array for colormaps
    gray_arr = np.array(diff.convert('L'), dtype=np.float32) / 255.0
    cmaps = ['inferno', 'magma', 'plasma', 'viridis', 'Reds', 'Blues']

    # Create colormap grid with legends
    legend_h = 10
    cmap_canvas = Image.new("RGB", (3*w, 2*h + top_h + legend_h), "white")
    draw2 = ImageDraw.Draw(cmap_canvas)
    for idx, cmap_name in enumerate(cmaps):
        col = idx % 3
        row = idx // 3
        x = col * w
        y = top_h + row * h
        # Top base label
        draw2.text((x + 5, 2), base_label, fill="black", font=FONT)
        # Cmap name on new line
        bbox_base = draw2.textbbox((x+5, 2), base_label, font=FONT)
        name_y = bbox_base[3] + 4
        draw2.text((x + 5, name_y), cmap_name, fill="black", font=FONT)
        # Paste cmap image
        arr = plt.get_cmap(cmap_name)(gray_arr)[:, :, :3]
        cmap_img = Image.fromarray((arr * 255).astype(np.uint8))
        cmap_canvas.paste(cmap_img, (x, y))
                # Draw legend bar
        grad = np.tile(np.linspace(1, 0, w)[None, :], (1, 1))
        grad_rgb = plt.get_cmap(cmap_name)(grad)[:, :, :3]
        legend_img = Image.fromarray((grad_rgb * 255).astype(np.uint8)).resize((w, legend_h))
        cmap_canvas.paste(legend_img, (x, y + h))

        # Annotate ticks on legend
        tick_y = y + h + legend_h + 2
        # left (max) value
        draw2.text((x, tick_y), "1.0", fill="black", font=FONT)
        # right (min) value
        bbox0 = draw2.textbbox((0,0), "0.0", font=FONT)
        tw0 = bbox0[2] - bbox0[0]
        draw2.text((x + w - tw0, tick_y), "0.0", fill="black", font=FONT)

        # Save colormap grid
    cmap_fname = f"{orig_path.stem}_{epoch_name}_cmap_grid_{ts}.png"
    cmap_canvas.save(out_dir / cmap_fname)


def main():
    for cls_dir in (INPUT_ROOT / "epoch0").iterdir():
        if not cls_dir.is_dir():
            continue
        class_name = cls_dir.name
        for orig in cls_dir.iterdir():
            if not orig.is_file():
                continue
            stem = orig.stem
            for epoch_dir in INPUT_ROOT.iterdir():
                if not (epoch_dir.is_dir() and epoch_dir.name.startswith("epoch")):
                    continue
                wm_folder = epoch_dir / class_name
                if not wm_folder.exists():
                    continue
                match = next((f for f in wm_folder.iterdir() if f.is_file() and f.stem.startswith(stem)), None)
                if not match:
                    continue
                out_dir = OUTPUT_ROOT / epoch_dir.name / class_name
                extract_and_save(orig, match, out_dir, epoch_dir.name, class_name)

if __name__ == "__main__":
    main()
