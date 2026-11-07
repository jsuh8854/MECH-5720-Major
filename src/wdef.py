"""
The Window Deformation solver sub-module.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class WDefSettings:
    """Settings for the Window Deformation Solver"""
    window_sizes: tuple = (64, 32, 16)  # one pass per entry, coarse -> fine
    overlap: float = 0.5                # fractional window overlap
    iterations: int = 2                 # deformation iterations per pass
    max_shift_frac: float = 0.4         # peak search limited to +-frac*window
    median_thresh: float = 2.0          # normalised median test threshold
    min_std: float = 1e-3               # skip windows with almost no texture


def _grid_axes(h: int, w: int, win: int, overlap: float):
    """Window start coordinates for one pass."""
    step = max(1, int(round(win * (1 - overlap))))
    y0 = np.arange(0, h - win + 1, step)
    x0 = np.arange(0, w - win + 1, step)
    return step, y0, x0

def _subpixel_peak(corr: np.ndarray, max_shift: int) -> np.ndarray:
    """corr: (n, w, w) fftshifted correlation. Returns (n, 2) residual (dx, dy)."""
    n, w, _ = corr.shape
    c0 = w // 2
    lo, hi = c0 - max_shift, c0 + max_shift + 1
    sub = corr[:, lo:hi, lo:hi]
    flat = sub.reshape(n, -1)
    idx = np.argmax(flat, axis=1)
    py, px = np.divmod(idx, sub.shape[2])
    ar = np.arange(n)

    # 3-point Gaussian fit on log of the (normalised, clipped) correlation
    peak = np.maximum(flat[ar, idx], 1e-12)
    cn = np.clip(corr / peak[:, None, None], 1e-3, None)
    gy = np.clip(py + lo, 1, w - 2)
    gx = np.clip(px + lo, 1, w - 2)
    ln = np.log(cn)
    c = ln[ar, gy, gx]
    dx_sub = (ln[ar, gy, gx - 1] - ln[ar, gy, gx + 1]) / (
        2 * (ln[ar, gy, gx - 1] - 2 * c + ln[ar, gy, gx + 1]) - 1e-12)
    dy_sub = (ln[ar, gy - 1, gx] - ln[ar, gy + 1, gx]) / (
        2 * (ln[ar, gy - 1, gx] - 2 * c + ln[ar, gy + 1, gx]) - 1e-12)
    dx_sub = np.clip(dx_sub, -1, 1)
    dy_sub = np.clip(dy_sub, -1, 1)

    return np.stack([px + lo - c0 + dx_sub, py + lo - c0 + dy_sub], axis=1)

def _correlate_grid(
    a: np.ndarray, b: np.ndarray,
    win: int,
    y0: np.ndarray, x0: np.ndarray,
    settings: WDefSettings
) -> np.ndarray:
    """Cross-correlate all windows; returns residual (ny, nx, 2), NaN where no texture."""
    out = np.full((len(y0), len(x0), 2), np.nan, np.float32)
    hann = np.hanning(win).astype(np.float32)
    taper = np.outer(hann, hann)
    max_shift = max(1, int(settings.max_shift_frac * win))

    for iy, y in enumerate(y0):  # one row of windows at a time to limit memory
        wa = np.stack([a[y:y + win, x:x + win] for x in x0])
        wb = np.stack([b[y:y + win, x:x + win] for x in x0])
        wa = wa - wa.mean(axis=(1, 2), keepdims=True)
        wb = wb - wb.mean(axis=(1, 2), keepdims=True)
        textured = ((wa.std(axis=(1, 2)) > settings.min_std)
                    & (wb.std(axis=(1, 2)) > settings.min_std))

        fa = np.fft.rfft2(wa * taper)
        fb = np.fft.rfft2(wb * taper)
        corr = np.fft.irfft2(np.conj(fa) * fb, s=(win, win))
        corr = np.fft.fftshift(corr, axes=(1, 2))

        res = _subpixel_peak(corr, max_shift)
        res[~textured] = np.nan
        out[iy] = res
    return out

def _validate_and_fill(field: np.ndarray, thresh: float) -> np.ndarray:
    """Normalised median test (Westerweel & Scarano); outliers/NaNs -> neighbour median."""
    ny, nx, _ = field.shape
    out = field.copy()
    pad = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
    for c in range(2):
        nb = np.stack([
            pad[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx, c]
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
        ])
        med = np.nanmedian(nb, axis=0)
        rm = np.nanmedian(np.abs(nb - med), axis=0)
        r = np.abs(field[..., c] - med) / (rm + 0.1)
        bad = np.isnan(field[..., c]) | (r > thresh)
        out[..., c] = np.where(bad, med, field[..., c])
    # anything still NaN (large untextured regions) is set to 0
    return np.nan_to_num(out, nan=0.0)

def _grid_to_dense(
    grid: np.ndarray,
    win: int,
    step: int,
    y0: np.ndarray,
    x0: np.ndarray,
    h: int,
    w: int
) -> np.ndarray:
    """Cubic-interpolate the vector grid to one vector per pixel."""
    # pylint:disable=no-member
    cy0 = y0[0] + (win - 1) / 2
    cx0 = x0[0] + (win - 1) / 2
    gy = np.clip((np.arange(h, dtype=np.float32) - cy0) / step, 0, len(y0) - 1)
    gx = np.clip((np.arange(w, dtype=np.float32) - cx0) / step, 0, len(x0) - 1)
    mapx, mapy = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
    return cv2.remap(
        grid.astype(np.float32),
        mapx, mapy,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _window_deform_piv(ref: np.ndarray, cur: np.ndarray, settings: WDefSettings) -> np.ndarray:
    """Dense (H, W, 2) displacement (dx, dy) of cur relative to ref."""
    # pylint:disable=no-member, invalid-name
    h, w = ref.shape
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

    disp = np.zeros((h, w, 2), np.float32)  # current dense estimate

    for win in settings.window_sizes:
        step, y0, x0 = _grid_axes(h, w, win, settings.overlap)
        for it in range(settings.iterations):
            print(f"  window {win}px, iteration {it + 1}/{settings.iterations}")
            # Symmetric deformation: both images are moved by half the estimate
            half = 0.5 * disp
            a = cv2.remap(
                ref, X - half[..., 0], Y - half[..., 1],
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            b = cv2.remap(
                cur, X + half[..., 0], Y + half[..., 1],
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            resid = _correlate_grid(a, b, win, y0, x0, settings)
            resid = _validate_and_fill(resid, settings.median_thresh)
            disp = disp + _grid_to_dense(resid, win, step, y0, x0, h, w)

    return disp


def perform_window_deform(
    ambient_images: list[np.ndarray],
    heat_images: list[np.ndarray],
    wdef_settings: WDefSettings
) -> list[np.ndarray]:
    """
    Same inputs and outputs as perform_tv_l1, using window-deformation cross-correlation.

    Each displacement image has shape (H, W, 2) having displacement data of (dx, dy) in pixels.
    """
    # Average ambient to reduce noise
    ref = np.mean(ambient_images, axis=0)

    results: list[np.ndarray] = []  # To return

    for i, img in enumerate(heat_images):
        print(f"Processing image {i+1}...")

        # Match overall brightness to the reference (lamp drift, exposure)
        cur = img * (ref.mean() / max(img.mean(), 1e-6))

        flow = _window_deform_piv(ref, cur, wdef_settings)  # (H, W, 2) with dx, dy

        # Remove global rigid shift from vibration/drift
        flow[..., 0] -= np.median(flow[..., 0])
        flow[..., 1] -= np.median(flow[..., 1])

        results.append(flow.astype(np.float32))

    print("Done!")

    return results
