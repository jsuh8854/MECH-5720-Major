"""
Generate fixed background noise for the imaging.
"""

import numpy as np
from matplotlib import pyplot as plt

IMAGE_WIDTH:  int = 1920
IMAGE_HEIGHT: int = 1080
OUT_DIR: str = "noise"

def generate_vertical_stripes():
    """Make da vertical stripes"""
    x = np.random.randint(2, size=IMAGE_WIDTH) * 255
    image = np.tile(x, (IMAGE_HEIGHT, 1))
    plt.imsave(f"{OUT_DIR}/vertical_noise.png", image, cmap="gray")

def generate_horizontal_stripes():
    """Make da sideways stripes"""
    y = np.random.randint(2, size=IMAGE_HEIGHT) * 255
    image = np.tile(y[:, np.newaxis], (1, IMAGE_WIDTH))
    plt.imsave(f"{OUT_DIR}/horizontal_noise.png", image, cmap="gray")

def generate_noise():
    """psssh"""
    noise: np.ndarray = np.random.randint(2, size=IMAGE_HEIGHT * IMAGE_WIDTH) * 255
    plt.imsave(f"{OUT_DIR}/noise.png", noise.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)), cmap="gray")

if __name__ == "__main__":
    generate_horizontal_stripes()
    generate_vertical_stripes()
    generate_noise()
