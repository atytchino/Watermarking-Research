import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_watermark_diff(orig_path: str, wm_path: str, amplify: float = 10.0):
    """
    Displays the original image, the watermarked image, and a heatmap of the
    absolute difference (watermark pattern) amplified for visibility.

    Args:
        orig_path: Path to the original (clean) image.
        wm_path:   Path to the corresponding watermarked image.
        amplify:   Factor by which to scale the difference for visualization.
    """
    # Load images and normalize to [0,1]
    orig = np.array(Image.open(orig_path).convert('RGB')).astype(np.float32) / 255.0
    wm = np.array(Image.open(wm_path).convert('RGB')).astype(np.float32) / 255.0

    # Compute difference
    diff = wm - orig

    # Absolute magnitude (average over channels)
    mag = np.mean(np.abs(diff), axis=2)

    # Amplify for visibility and clip
    mag_ampl = np.clip(mag * amplify, 0.0, 1.0)

    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(wm)
    axes[1].set_title("Watermarked")
    axes[1].axis("off")

    im = axes[2].imshow(mag_ampl, cmap="inferno")
    axes[2].set_title(f"Amplified Abs Diff (×{amplify:.1f})")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


visualize_watermark_diff(
      orig_path=r"E:\watermarking\afhq\watermarked_clean_pairs\epoch10_c1.jpg",
      wm_path=  r"E:\watermarking\afhq\watermarked_clean_pairs\epoch10_wm1.png",
      amplify=15.0
  )