import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from PIL import Image
import os

def generate_superpixels(image_path, output_dir, segmented=False):
    os.makedirs(output_dir, exist_ok=True)

    # print("superpixel start")

    image = Image.open(image_path).convert("RGB")
    image_np = img_as_float(np.array(image))
    segments = slic(image_np, n_segments=100, compactness=10, sigma=1)
    marked = mark_boundaries(image_np, segments)
    marked_image = (marked * 255).astype(np.uint8)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    suffix = "_seg" if segmented else "_raw"
    cv2.imwrite(os.path.join(output_dir, f"{filename}_superpixels{suffix}.jpg"), marked_image)
    print("superpixel end")
