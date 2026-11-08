"""
Temperature analysis module.
"""

import numpy as np


def magnitude_std(flow: np.ndarray) -> float:
    """Standard deviation of the displacement magnitude image."""
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float(mag.std())

def temp_linear(r: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    """Signal proportional to (T - T_ambient)."""
    return min_temp + (max_temp - min_temp) * r

def temp_density(r: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    """Signal proportional to the air density deficit, 1 - T_ambient/T (ideal gas law)."""
    ta = min_temp + 273.15
    tm = max_temp + 273.15
    r = np.clip(r, 0.0, 2.0)  # keep the denominator positive when extrapolating
    return ta / (1.0 - r * (1.0 - ta / tm)) - 273.15

def temp_power(r: np.ndarray, p: float, min_temp: float, max_temp: float) -> np.ndarray:
    """Signal proportional to (T - T_ambient)^p, so dT = dT_max * r^(1/p)."""
    r = np.clip(r, 0.0, None)
    return min_temp + (max_temp - min_temp) * r ** (1.0 / p)

def ambient_noise_stds(
    perform,
    settings,
    ambient: list[np.ndarray],
    noise_test_frames: int
) -> tuple[list[float], int]:
    """
    Calculate the noise floor for the ambient frames:

    std of |d| for the first NOISE_TEST_FRAMES ambient frames,
    each measured against the mean of the remaining ambient frames
    (the test frames are removed from the reference).

    Returns (stds, number of frames in the reference).
    """
    # pylint:disable=line-too-long

    test_frames, reference = ambient[:noise_test_frames], ambient[noise_test_frames:]
    flows = perform(reference, test_frames, settings)
    return [magnitude_std(f) for f in flows], len(reference)

# ---------------------------------------------------------------------
# Image grouping and measurements
# ---------------------------------------------------------------------

def build_heat_groups(heat, heat_names, heat_groups):
    """Map each heat-image index to its configured heat-setting group."""
    group_of = {}
    for group_name, (start, stop) in heat_groups.items():
        for i in range(start, min(stop, len(heat))):
            group_of[i] = group_name

    print("\nHeat images (natural sort order) and assigned group:")
    for i, name in enumerate(heat_names):
        print(
            f"  [{i + 1:2d}] {name:<30s} -> "
            f"{group_of.get(i, '(unassigned, skipped)')}"
        )

    return group_of

def measure_noise_floor(perform, settings, ambient, noise_test_frames):
    """Measure ambient noise and scale it to the heat-image reference size."""
    noise_stds, n_noise_ref = ambient_noise_stds(
        perform, settings, ambient, noise_test_frames
    )

    # Heat images use all ambient frames as the reference. Correct for the
    # difference in residual noise caused by the different reference sizes.
    n_heat_ref = len(ambient)
    factor = float(
        np.sqrt(
            (1 + 1 / n_heat_ref) /
            (1 + 1 / n_noise_ref)
        )
    )
    noise_stds = [value * factor for value in noise_stds]

    noise = float(np.mean(noise_stds))

    return noise_stds, noise

def measure_heat_images(ambient, heat, heat_names, group_of, perform, settings):
    """Measure displacement-magnitude standard deviation for each assigned image."""
    stds = {}
    indices = sorted(group_of)

    for count, i in enumerate(indices, start=1):
        print(
            f"[temp] heat image {count}/{len(indices)} "
            f"({heat_names[i]})"
        )
        displacement = perform(ambient, [heat[i]], settings)[0]
        stds[i] = magnitude_std(displacement)

    return stds, indices

# ---------------------------------------------------------------------
# Calibration and temperature calculations
# ---------------------------------------------------------------------

def calculate_signal(std, noise):
    """Remove independent background noise in quadrature."""
    return float(np.sqrt(max(std**2 - noise**2, 0.0)))


def calculate_normalised_signals(stds, indices, group_of, noise, heat_groups):
    """Normalise each noise-subtracted image signal to the calibration group."""
    calibration_group = next(iter(heat_groups))

    calibration_signal = float(np.mean([
        calculate_signal(stds[i], noise)
        for i in indices
        if group_of[i] == calibration_group
    ]))

    normalised = {
        i: calculate_signal(stds[i], noise) / calibration_signal
        for i in indices
    }

    return calibration_group, normalised

def fit_power_exponent(
    normalised,
    indices,
    group_of,
    low_temp,
    high_temp,
    known_temps,
):
    """Fit the power-law exponent with a reference middle temperature."""
    numerator = 0.0
    denominator = 0.0
    fit_lines = []

    for group_name, known_temp in known_temps.items():

        group_signal = float(np.mean([
            normalised[i]
            for i in indices
            if group_of[i] == group_name
        ]))

        x = (
            (known_temp - low_temp)
            / (high_temp - low_temp)
        )

        if group_signal > 0 and 0 < x < 1:
            log_signal = np.log(group_signal)
            log_temp_fraction = np.log(x)

            fit_lines.append(
                f"  {group_name}: datasheet {known_temp:.0f} C "
                f"-> r = {group_signal:.3f}, "
                f"implied exponent "
                f"{log_signal / log_temp_fraction:.2f}"
            )

            numerator += log_signal * log_temp_fraction
            denominator += log_temp_fraction**2

    exponent = numerator / denominator

    return exponent




def build_temperature_rows(
    indices,
    heat_names,
    group_of,
    stds,
    normalised,
    noise,
    exponent,
    min_temp,
    max_temp,
):
    """Build the per-image temperature results."""
    rows = []

    for i in indices:
        ratio = normalised[i]

        rows.append({
            "index": i + 1,
            "file": heat_names[i],
            "group": group_of[i],
            "std_px": stds[i],
            "signal_px": calculate_signal(stds[i], noise),
            "r": ratio,
            "T_linear_C": float(
                temp_linear(ratio, min_temp, max_temp)
            ),
            "T_density_C": float(
                temp_density(ratio, min_temp, max_temp)
            ),
            "T_power_C": float(
                temp_power(
                    ratio, exponent, min_temp, max_temp
                )
            ),
        })

    return rows
