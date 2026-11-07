"""
General utility functions used for the lab.
"""

import sys
import re
import pickle
from pathlib import Path
import numpy as np
import cv2
import rawpy

# ========================================
# Module constants and members
# ========================================

MAX_PIXEL : float = 1023
"""Max value of RAW pixels."""

BLACK_LEVEL: int = 16
"""Black level of sensor."""

FIXED_PATTERN_PICKLE_PATH: str = "fixed_pattern.pkl"
"""Path to pickle file with fixed pattern noise"""

fixed_pattern: np.ndarray = None
"""Fixed pattern of sensor."""

# ========================================
# Initialise module
# ========================================

# Load pickle
try:
    with open(FIXED_PATTERN_PICKLE_PATH, "rb") as f:
        fixed_pattern = pickle.load(f)
    print("Info: Fixed pattern pickle loaded.")
except FileNotFoundError:
    print("Warning: Fixed pattern pickle not found.")


# ========================================
# Module defs
# ========================================

def _natural_key(path: Path):
    """Sort heat_2 before heat_10 (plain sorted() would put heat_10 before heat_2)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]

def read_raw(filename: str) -> np.ndarray:
    """Reads the DNG file and returns as an np.ndarray."""
    try:
        # Load and view the raw bayer image
        with rawpy.imread(filename) as raw:
            return np.array(raw.raw_image, dtype=np.int16, copy=True, order="C")

    except FileNotFoundError as e:
        print(f"Error loading {filename}: {e}")
        sys.exit(1)

def debayer_raw(raw: np.ndarray) -> np.ndarray:
    """Convert raw DNG into a float32 debayered RGB image."""
    # pylint: disable=no-member
    raw = np.asarray(raw, dtype=np.int16, order="C")
    raw = np.clip(raw, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    rgb = cv2.cvtColor(raw, cv2.COLOR_BayerGR2RGB).astype(np.float32)
    if np.max(rgb) >= MAX_PIXEL - 8:
        print("SATURATION!!!")
    # print(np.max(rgb))
    rgb_norm = rgb / MAX_PIXEL

    return rgb_norm

def rotate_image(img: np.ndarray, k: int = -1) -> np.ndarray:
    """Rotate the image, default value of -1 rotates to match lab setup."""
    return np.rot90(img, k=k)

def process_raw(filename: str) -> np.ndarray:
    """Do full pre-processing on raw image from provided file path."""
    raw = read_raw(filename)
    debayer = debayer_raw(raw)
    green = debayer[:, :, 1]
    if fixed_pattern is not None:
        green -= fixed_pattern
    rotated = rotate_image(green)

    # Jank crop
    crop_size = 512
    h, w = rotated.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    cropped = rotated[top:top + crop_size, left:left + crop_size]

    return cropped

def load_images(directory: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load the ambient and heat image stacks (natural filename order) and the heat file names."""
    files = sorted(directory.iterdir(), key=_natural_key)
    ambient_files = [
        f for f in files
        if f.is_file() and f.name.startswith("ambient_")
    ]
    heat_files = [f for f in files if f.is_file() and f.name.startswith("heat_")]

    ambient = [process_raw(str(f)) for f in ambient_files]
    heat = [process_raw(str(f)) for f in heat_files]

    return ambient, heat, [f.name for f in heat_files]
