"""
The TV-L1 solver sub-module.

Wrapper for creating and processing OpenCV's TV-L1 engine.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class TVL1Settings:
    """Settings for the TV-L1 solver."""
    l: float = 0.05 # Lower for smoother, higher for higher trust in data
    """Lambda"""
    tau: float = 0.25
    scales_number: int = 5
    warpings_number: int = 5
    epsilon: float = 0.01
    inner_iterations: int = 30
    outer_iterations: int = 10
    blur_sigma: float = 2.0  # Gaussian pre-blur applied to BOTH images (0 = disabled)

def _create_tvl1(settings: TVL1Settings):
    """Create and initialise tv-l1 from provided settings."""
    # pylint:disable=no-member
    tv_l1 = cv2.optflow.createOptFlow_DualTVL1()

    tv_l1.setLambda(settings.l)
    tv_l1.setTau(settings.tau)
    tv_l1.setScalesNumber(settings.scales_number)
    tv_l1.setWarpingsNumber(settings.warpings_number)
    tv_l1.setEpsilon(settings.epsilon)
    tv_l1.setInnerIterations(settings.inner_iterations)
    tv_l1.setOuterIterations(settings.outer_iterations)

    return tv_l1

def perform_tv_l1(
    ambient_images: list[np.ndarray],
    heat_images: list[np.ndarray],
    tv_l1_settings: TVL1Settings
) -> list[np.ndarray]:
    """
    Takes in stacks of ambient and heat source images and returns a list of displacement images.
    
    Each displacement image has shape (H, W, 2) having displacement data of (dx, dy) in pixels.
    """
    # pylint:disable=no-member

    # Average ambient to reduce noise
    ref = np.mean(ambient_images, axis=0)

    # TV-L1's data/smoothness balance depends on the intensity range, so scale both
    # images to [0, 1] with ONE shared factor (applied identically to every image).
    scale = 1.0 / max(float(ref.max()), 1e-6)
    ref = ref * scale

    # Optional pre-blur, always applied to both images or neither.
    if tv_l1_settings.blur_sigma > 0:
        ref = cv2.GaussianBlur(ref, (0, 0), tv_l1_settings.blur_sigma)

    # Create the solver from settings
    tv_l1 = _create_tvl1(tv_l1_settings)

    results: list[np.ndarray] = [] # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")

        cur = img * scale

        if tv_l1_settings.blur_sigma > 0:
            cur = cv2.GaussianBlur(cur, (0, 0), tv_l1_settings.blur_sigma)

        # Match overall brightness to the reference (lamp drift, exposure)
        cur = cur * (ref.mean() / max(cur.mean(), 1e-6))

        flow = tv_l1.calc(ref, cur, None)  # (H, W, 2) with dx, dy

        # Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results
