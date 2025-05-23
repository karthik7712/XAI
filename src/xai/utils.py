import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.segmentation import slic
from src.xai.gradcam import GradCAM


def save_gradcam_output(gradcam_output, label, save_dir, image_id):
    """
    Save the Grad-CAM output as an image.

    :param gradcam_output: Grad-CAM output (heatmap)
    :param label: The true label of the image
    :param save_dir: Directory to save the Grad-CAM output
    :param image_id: Unique ID for the image to be saved
    """
    # Convert Grad-CAM output to numpy and scale
    gradcam_output = gradcam_output.cpu().detach().numpy()
    gradcam_output = cv2.applyColorMap(np.uint8(255 * gradcam_output), cv2.COLORMAP_JET)

    # Plot Grad-CAM heatmap
    plt.imshow(gradcam_output)
    plt.axis('off')
    gradcam_path = os.path.join(save_dir, f'gradcam_{label}_{image_id}.png')
    plt.savefig(gradcam_path, bbox_inches='tight', pad_inches=0)

    print(f"Saved Grad-CAM for {label} image {image_id}")


def apply_superpixels(image):
    """
    Apply Superpixel segmentation using SLIC.

    :param image: Input image (Tensor)
    :return: Superpixels segmentation mask
    """
    # Convert image tensor to numpy
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    # Apply SLIC (Simple Linear Iterative Clustering) for superpixels
    segments = slic(image, n_segments=100, compactness=10, sigma=1)

    return segments


def save_superpixels_output(superpixels, label, save_dir, image_id):
    """
    Save the Superpixels output as an image with the segmentation boundaries.

    :param superpixels: Superpixel segmentation output
    :param label: The true label of the image
    :param save_dir: Directory to save the superpixels output
    :param image_id: Unique ID for the image to be saved
    """
    # Create a mask where superpixels boundaries are
    boundaries = np.zeros_like(superpixels)
    boundaries[superpixels == superpixels.min()] = 1

    # Convert to color for visibility
    colored_boundaries = np.zeros((superpixels.shape[0], superpixels.shape[1], 3), dtype=np.uint8)
    colored_boundaries[boundaries == 1] = [255, 0, 0]  # Red boundaries

    # Save the image with boundaries
    superpixels_path = os.path.join(save_dir, f'superpixels_{label}_{image_id}.png')
    cv2.imwrite(superpixels_path, colored_boundaries)

    print(f"Saved Superpixels for {label} image {image_id}")


