"""Imaging code for Background Orientated Schlieren Imaging"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import utils
import temp
from tvl1 import TVL1Settings, perform_tv_l1
from wdef import WDefSettings, perform_window_deform

# ----------------------------------------
# Analysis mode constants
# ----------------------------------------

# "tvl1", "wdef", "both" or "temp"
# Temperature estimate (METHOD = "temp")
METHOD: str = "temp"

# Solver used for the temperature analysis: "tvl1" or "wdef"
TEMP_METHOD: str = "wdef"

# ----------------------------------------
# Directory constants
# ----------------------------------------

IMAGE_DIRECTORY: str = "data"
OUT_DIRECTORY: str = "out"

# ----------------------------------------
# Temperature data constants
# ----------------------------------------

AMBIENT_TEMP_C: float = 30.0
MAX_TEMP_C:     float = 280.0 # Hot plate "MAX", from the manual

# The first group is the calibration setting that is assumed to be at MAX_TEMP_C.
# Positions [start, stop) in the *sorted* heat image list, one entry per hot-plate setting.
HEAT_GROUPS: dict = {
    "max": (0, 4),
    "medium": (4, 8),
    "low": (8, 12)
}
KNOWN_TEMPS_C = {"low": 150.0} # Hot plate "LOW", from the manual

GROUP_COLOURS: list[str] = ["tab:red", "tab:orange", "tab:blue", "tab:green"]

NOISE_TEST_FRAMES: int = 4  # first N ambient frames are the noise-floor test frames;
                            # they are compared against the mean of the remaining ambient frames

# ---------------------------------------------------------------------------
# Create the settings for the solvers.
# Adjust these to change the values sent to the solvers.
# ---------------------------------------------------------------------------

def make_tvl1_settings() -> TVL1Settings:
    """Settings used by the TV-L1 analyses."""
    return TVL1Settings(
        l=0.15,
        tau=0.25,
        scales_number=5,
        warpings_number=5,
        epsilon=0.01,
        inner_iterations=30,
        outer_iterations=10
    )

def make_wdef_settings() -> WDefSettings:
    """Settings used by the window-deformation analyses."""
    return WDefSettings(
        window_sizes=(64, 32, 16),
        overlap=0.5,
        iterations=8,
        max_shift_frac=0.4,
        median_thresh=2.0,
        min_std=1e-3
    )

# ---------------------------------------------------------------------------
# Plotting and analysis
# ---------------------------------------------------------------------------

def plot_displacements(
    heat_flow: np.ndarray, ambient_flow: np.ndarray,
    heat_image: np.ndarray, ambient_image: np.ndarray,
    method: str, save_path: Path,
    central_fraction: float = 0.99
) -> None:
    """
    4x2 plot: x displacement, y displacement and displacement magnitude for the
    heat image and the ambient noise floor, with the debayered source images underneath.

    The shared colour scales span only the middle `central_fraction` (default 99%) of all
    displacement values, so extreme outliers don't flatten the rest of the image.
    """
    img1 = heat_flow[:, :, 0]
    img2 = ambient_flow[:, :, 0]
    img3 = heat_flow[:, :, 1]
    img4 = ambient_flow[:, :, 1]
    mag_heat = np.hypot(heat_flow[:, :, 0], heat_flow[:, :, 1])
    mag_ambient = np.hypot(ambient_flow[:, :, 0], ambient_flow[:, :, 1])

    # Shared signed colour scale for dx/dy from the central fraction of all values
    tail = 100 * (1 - central_fraction) / 2
    all_vals = np.concatenate([img.ravel() for img in (img1, img2, img3, img4)])
    vmin, vmax = np.percentile(all_vals, [tail, 100 - tail])

    # Shared magnitude scale: 0 up to the upper percentile (magnitude is never negative)
    all_mag = np.concatenate([mag_heat.ravel(), mag_ambient.ravel()])
    mmax = np.percentile(all_mag, 100 - tail)

    # Shared grey scale for the two source images (1st to 99th percentile)
    both = np.concatenate([heat_image.ravel(), ambient_image.ravel()])
    gmin, gmax = np.percentile(both, [1, 99])

    # Third column is reserved for colorbars, so every row of image axes has the
    # same width and the bottom (source image) row lines up with the rows above.
    fig = plt.figure(figsize=(12, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, width_ratios=[1, 1, 0.04])
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(4)])
    cax_disp = fig.add_subplot(gs[0:2, 2])
    cax_mag = fig.add_subplot(gs[2, 2])

    # Subplot title formatting
    title_kwargs = {
        "fontsize": 16,
        "fontweight": "bold",
        "pad": 12
    }

    im1 = axes[0][0].imshow(img1, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0][0].set_title("Heat (x displacement)", **title_kwargs)

    axes[0][1].imshow(img2, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0][1].set_title("Ambient Comparison (x displacement)", **title_kwargs)

    axes[1][0].imshow(img3, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1][0].set_title("Heat (y displacement)", **title_kwargs)

    axes[1][1].imshow(img4, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1][1].set_title("Ambient Comparison (y displacement)", **title_kwargs)

    im_mag = axes[2][0].imshow(mag_heat, cmap="inferno", vmin=0, vmax=mmax)
    axes[2][0].set_title("Heat (displacement magnitude)", **title_kwargs)

    axes[2][1].imshow(mag_ambient, cmap="inferno", vmin=0, vmax=mmax)
    axes[2][1].set_title("Ambient Comparison (displacement magnitude)", **title_kwargs)

    axes[3][0].imshow(heat_image, cmap="gray", vmin=gmin, vmax=gmax)
    axes[3][0].set_title("Heat (debayered)", **title_kwargs)

    axes[3][1].imshow(ambient_image, cmap="gray", vmin=gmin, vmax=gmax)
    axes[3][1].set_title("Ambient (debayered)", **title_kwargs)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(method, fontsize=21, fontweight="bold")

    # Colour bars
    cb_disp = fig.colorbar(im1, cax=cax_disp)
    cb_disp.set_label("Displacement (px)", fontsize=15, labelpad=14)
    cb_disp.ax.tick_params(labelsize=13)

    cb_mag = fig.colorbar(im_mag, cax=cax_mag)
    cb_mag.set_label("Magnitude (px)", fontsize=15, labelpad=14)
    cb_mag.ax.tick_params(labelsize=13)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_collage(
    magnitudes: dict,
    heat_names: list[str],
    method: str,
    save_path: Path,
    image_shape: tuple[int, int],
    central_fraction: float = 0.95
) -> None:
    """
    Collage of displacement-magnitude images.
    One column per HEAT_GROUPS setting, one row per repeat within the setting.
    All panels share the pixel axes and one colour scale
    (0 up to the upper percentile of all panels pooled, so outliers don't wash everything out).

    magnitudes: {heat image index: 2D magnitude array, possibly downsampled}
    image_shape: (H, W) of the original images, so the axes stay in original pixel units.
    """
    h, w = image_shape
    ranges = {g: range(a, min(b, len(heat_names))) for g, (a, b) in HEAT_GROUPS.items()}
    ncols = len(ranges)
    nrows = max(len(r) for r in ranges.values())

    tail = 100 * (1 - central_fraction) / 2
    pooled = np.concatenate([m.ravel() for m in magnitudes.values()])
    vmax = float(np.percentile(pooled, 100 - tail))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows + 0.6),
        sharex=True, sharey=True,
        constrained_layout=True, squeeze=False
    )
    im = None
    for col, (gname, idx_range) in enumerate(ranges.items()):
        for row in range(nrows):
            ax = axes[row][col]
            if row < len(idx_range) and idx_range[row] in magnitudes:
                i = idx_range[row]
                im = ax.imshow(
                    magnitudes[i],
                    cmap="inferno", vmin=0, vmax=vmax,
                    extent=(0, w, h, 0), interpolation="nearest"
                )
                ax.set_title(f"{gname}: {Path(heat_names[i]).stem}", fontsize=16)
            else:
                ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("x (px)", fontsize=14)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (px)", fontsize=14)

    fig.suptitle(
        f"{method}: displacement magnitude, shared colour scale",
        fontsize=17, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=axes, label="Displacement magnitude (px)", shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Displacement magnitude (px)", fontsize=14, labelpad=12)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyse_all(
    name: str, out_subdir: str, perform, settings,
    ambient: list[np.ndarray], heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """Run `perform` on every heat image and save one plot per image.

    The ambient-vs-ambient noise floor is computed once and reused for all plots.
    Plots are saved in OUT_DIRECTORY/<out_subdir>/<out_subdir>_<index>.png, where the index
    follows the sorted order of the heat_* files.
    A collage of all heat images (one column per HEAT_GROUPS setting, shared axes and colour scale)
    is saved as <out_subdir>_collage.png.
    """
    # pylint:disable=no-member
    heat_names = heat_names or [f"heat_{i + 1}" for i in range(len(heat))]

    out_dir = Path(OUT_DIRECTORY) / out_subdir
    width = max(3, len(str(len(heat) - 1)))

    print(f"[{name}] computing ambient noise floor...")

    ambient_flow = perform(ambient, [ambient[0]], settings)[0]

    magnitudes = {}

    for i, heat_image in enumerate(heat):
        print(f"[{name}] heat image {i + 1}/{len(heat)}")

        flow = perform(ambient, [heat_image], settings)[0]

        plot_displacements(
            flow, ambient_flow,
            heat_image, ambient[0],
            name, out_dir / f"{out_subdir}_{i:0{width}d}.png"
        )

        mag = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32)
        magnitudes[i] = mag

    plot_collage(
        magnitudes,
        heat_names,
        name,
        out_dir / f"{out_subdir}_collage.png",
        heat[0].shape[:2]
    )

    print(f"[{name}] saved {len(heat)} plots and the collage to {out_dir}")

def summarise_temperature_groups(rows):
    """Aggregate per-image results into per-setting summary arrays."""
    groups = {}

    for group_name in HEAT_GROUPS:
        selected = [
            row for row in rows
            if row["group"] == group_name
        ]
        if selected:
            groups[group_name] = selected

    summary = {}

    for group_name, selected in groups.items():
        summary[group_name] = {
            "std": np.array([row["std_px"] for row in selected]),
            "r": np.array([row["r"] for row in selected]),
            "tl": np.array([row["T_linear_C"] for row in selected]),
            "td": np.array([row["T_density_C"] for row in selected]),
            "tp": np.array([row["T_power_C"] for row in selected]),
        }

    return groups, summary

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_displacement_spread(
    noise_stds, noise, groups, out_dir
):
    """Plot ambient noise and displacement spread for each heat image."""
    title_size = 17
    label_size = 15
    tick_size = 12
    legend_size = 12

    fig, ax = plt.subplots(
        figsize=(16, 8),
        constrained_layout=True,
    )

    x_positions = []
    labels = []

    for k, value in enumerate(noise_stds):
        ax.bar(
            k,
            value,
            color="tab:gray",
            label="Ambient (noise floor)" if k == 0 else None,
        )
        x_positions.append(k)
        labels.append(f"amb {k + 1}")

    position = len(noise_stds) + 1

    i = 0
    for group_name, selected in groups.items():
        start = position

        for j, row in enumerate(selected):
            ax.bar(
                position,
                row["std_px"],
                color=GROUP_COLOURS[i],
                label=group_name if j == 0 else None,
            )
            x_positions.append(position)
            labels.append(Path(row["file"]).stem)
            position += 1

        group_mean = np.mean([
            row["std_px"] for row in selected
        ])

        ax.hlines(
            group_mean,
            start - 0.4,
            position - 0.6,
            colors="black",
            linestyles="-",
            linewidth=2,
        )

        position += 1
        i += 1

    ax.axhline(
        noise,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
        fontsize=tick_size,
    )
    ax.set_ylabel(
        "Std of displacement magnitude (px)",
        fontsize=label_size,
        labelpad=10,
    )
    ax.set_xlabel(
        "Heat image",
        fontsize=label_size,
        labelpad=10,
    )
    ax.set_title(
        "Spread of displacement magnitude per image\n"
        "(black lines = group means)",
        fontsize=title_size,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.legend(fontsize=legend_size, loc="best")

    fig.savefig(
        out_dir / "displacement_spread.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_temperature_calibration(
    exponent,
    calibration_group,
    summary,
    out_dir,
):
    """Plot the temperature calibration models and group estimates."""
    title_size = 17
    label_size = 15
    tick_size = 12
    legend_size = 12
    annotation_size = 12

    fig, ax = plt.subplots(
        figsize=(12, 9),
        constrained_layout=True,
    )

    r_grid = np.linspace(0, 1.3, 200)

    ax.plot(
        r_grid,
        temp.temp_linear(r_grid, AMBIENT_TEMP_C, MAX_TEMP_C),
        color="tab:purple",
        linewidth=2.5,
        label="Linear in T",
    )
    ax.plot(
        r_grid,
        temp.temp_density(r_grid, AMBIENT_TEMP_C, MAX_TEMP_C),
        color="tab:green",
        linewidth=2.5,
        label="Linear in density deficit",
    )
    ax.plot(
        r_grid,
        temp.temp_power(r_grid, exponent, AMBIENT_TEMP_C, MAX_TEMP_C),
        color="tab:brown",
        linewidth=2.5,
        label=f"Power law, exponent {exponent:.2f}",
    )
    ax.scatter(
        [0, 1],
        [AMBIENT_TEMP_C, MAX_TEMP_C],
        color="black",
        s=80,
        zorder=5,
        label="Calibration points",
    )

    fixed_points = [
        (
            0.0,
            AMBIENT_TEMP_C,
            f"ambient (fixed): {AMBIENT_TEMP_C:.0f} C",
        ),
        (
            1.0,
            MAX_TEMP_C,
            f"{calibration_group} (fixed): {MAX_TEMP_C:.0f} C",
        ),
    ]

    for x, y, label in fixed_points:
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, -6),
            ha="left",
            va="top",
            color="black",
            fontsize=9,
            fontweight="bold",
        )

    for group_name, values in summary.items():
        if group_name == calibration_group:
            continue

        mean_ratio = values["r"].mean()

        estimates = (
            (values["tl"].mean(), "tab:purple", 8),
            (values["tp"].mean(), "tab:brown", -2),
            (values["td"].mean(), "tab:green", -14),
        )

        for mean_temp, colour, offset_y in estimates:
            ax.scatter(
                mean_ratio,
                mean_temp,
                color=colour,
                s=65,
                zorder=6,
            )


            ax.annotate(
                f"{group_name}: {mean_temp:.0f} C",
                (mean_ratio, mean_temp),
                textcoords="offset points",
                xytext=(8, offset_y),
                color=colour,
                fontsize=annotation_size,
                fontweight="bold",
            )

    ax.set_xlabel(
        "Normalised signal r = sqrt(std² - noise²) / "
        "max-setting signal",
        fontsize=label_size,
        labelpad=10,
    )
    ax.set_ylabel(
        "Estimated temperature (°C)",
        fontsize=label_size,
        labelpad=10,
    )
    ax.set_title(
        "Temperature estimate from displacement spread",
        fontsize=title_size,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=legend_size, loc="upper left")

    fig.savefig(
        out_dir / "temperature_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# ---------------------------------------------------------------------------
# Analysis calls for each algorithm
# ---------------------------------------------------------------------------


def tv_l1_analysis(
    ambient: list[np.ndarray], heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """
    Run TV-L1 on every heat image,
    with an ambient-vs-ambient noise-floor check to the side for comparison.
    """
    analyse_all(
        "TV-L1", "tvl1",
        perform_tv_l1, make_tvl1_settings(),
        ambient, heat, heat_names
    )

def wdef_analysis(
    ambient: list[np.ndarray],heat: list[np.ndarray],
    heat_names: list[str] | None = None
) -> None:
    """
    Run window-deformation PIV on every heat image,
    with an ambient-vs-ambient noise-floor check to the side for comparison.
    """
    analyse_all(
        "Window deformation", "wdef",
        perform_window_deform, make_wdef_settings(),
        ambient, heat, heat_names
    )

def temperature_analysis(
    ambient: list[np.ndarray],
    heat: list[np.ndarray],
    heat_names: list[str],
    perform,
    settings,
) -> None:
    """
    Estimate hot-plate temperatures from displacement-magnitude images.

    Uses ambient noise as the lower calibration point and the first
    HEAT_GROUPS entry as the maximum-temperature calibration point.
    """

    # Assign heat images to groups.
    group_of = temp.build_heat_groups(heat, heat_names, HEAT_GROUPS)

    # Measure ambient noise and heat-image displacement spread.
    noise_stds, noise = temp.measure_noise_floor(
        perform, settings, ambient, NOISE_TEST_FRAMES
    )
    stds, indices = temp.measure_heat_images(
        ambient, heat, heat_names, group_of, perform, settings
    )

    # Calculate noise-subtracted and normalised signals.
    calibration_group, normalised = (
        temp.calculate_normalised_signals(
            stds, indices, group_of, noise, HEAT_GROUPS
        )
    )

    # Fit the power-law temperature model.
    exponent = temp.fit_power_exponent(
        normalised, indices, group_of,
        AMBIENT_TEMP_C, MAX_TEMP_C,
        KNOWN_TEMPS_C
    )

    rows = temp.build_temperature_rows(
        indices,
        heat_names,
        group_of,
        stds,
        normalised,
        noise,
        exponent,
        AMBIENT_TEMP_C,
        MAX_TEMP_C
    )

    groups, summary = summarise_temperature_groups(rows)

    # 8. Generate plots.
    out_dir = Path(OUT_DIRECTORY) / "temperature"
    out_dir.mkdir(parents=True, exist_ok=True)


    plot_displacement_spread(
        noise_stds, noise, groups, out_dir
    )
    plot_temperature_calibration(
        exponent,
        calibration_group,
        summary,
        out_dir,
    )

    print(
        f"\nSaved {out_dir / 'displacement_spread.png'} and "
        f"{out_dir / 'temperature_calibration.png'}"
    )

# ---------------------------------------------------------------------------
# Main call
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load images to stack
    directory = Path(IMAGE_DIRECTORY)
    ambient_stack, heat_stack, heat_file_names = utils.load_images(directory)

    # Pick processing method so don't have to run all every time.
    if METHOD in ("tvl1", "both"):
        tv_l1_analysis(ambient_stack, heat_stack, heat_file_names)
    if METHOD in ("wdef", "both"):
        wdef_analysis(ambient_stack, heat_stack, heat_file_names)
    if METHOD == "temp":
        temperature_analysis(
            ambient_stack,
            heat_stack,
            heat_file_names,
            perform_tv_l1 if TEMP_METHOD == "tvl1" else perform_window_deform,
            make_tvl1_settings() if TEMP_METHOD == "tvl1" else make_wdef_settings()
        )
