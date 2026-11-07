"""
find_fixed_pattern.py

Finds the fixed pattern noise and saves it in a pickle file.
"""

import pickle
import numpy as np
import utils

BLACK_LEVEL: int = 16
"""The black level of the camera, known to be 16."""

IMG_DIR: str = "fixed_pattern"
"""Directory containing the images. Image names should be in the format 1.dng, 2.dng, etc."""

IMG_COUNT: int = 10
"""Number of images in IMG_DIR to average."""

SAVE_NAME: str = "fixed_pattern.pkl"
"""File to save pickle as."""

if __name__ == "__main__":
    # Variable to dump
    fixed_pattern: np.ndarray

    # Load images into numpy array as a stack, subtracting black level while doing so
    raw_image_list: list[np.ndarray] = []
    raw_images: np.ndarray

    for i in range(IMG_COUNT):
        img_path: str = f"{IMG_DIR}/{i+1}.dng"
        black_image: np.ndarray =  utils.read_raw(img_path) - BLACK_LEVEL
        raw_image_list.append(black_image)

    raw_images = np.stack(raw_image_list)

    # Calculate average as fixed-pattern noise, clipping bottom values
    fixed_pattern = np.clip(np.mean(raw_images, axis=0), 0, utils.MAX_PIXEL)
    fixed_pattern = utils.debayer_raw(fixed_pattern)[:,:,1]

    # Sanity checks
    min_val: float = np.min(fixed_pattern)
    max_val: float = np.max(fixed_pattern)
    ave_val: float = np.mean(fixed_pattern)

    print(f"Shape: {fixed_pattern.shape}")
    print(
        f"Min: {min_val}, Max: {max_val}, Ave: {ave_val}"
    )

    # Save as pickle
    with open(SAVE_NAME, "wb") as f:
        pickle.dump(fixed_pattern, f)
