import numpy as np
from PIL import Image

orig = np.array(Image.open("E:/watermarking/afhq/tiny_train/classA/img001.png")).astype(np.float32)/255
wm   = np.array(Image.open("E:/watermarking/afhq/full_final_watermarked/classA/img001…png")).astype(np.float32)/255

print("mean abs diff:", np.mean(np.abs(orig - wm)))
print("max abs diff: ", np.max(np.abs(orig - wm)))
