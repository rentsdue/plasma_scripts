import numpy as np


# Purpose:
#   Build a smooth scalar time series used to identify saturated turbulence.
# How it works:
#   Sums 1/2 k^2 |phi_k|^2 over Fourier modes for each saved frame.
def compute_total_kinetic_energy_series(uk, kx2d, ky2d):
    """Compute a smooth total kinetic-energy scalar per saved timestep.

    Use this, or another smooth energy-like scalar such as zonal energy, for
    saturation-window detection. Do not use intermittent/noisy quantities such
    as particle flux for choosing the saturated averaging window.
    """
    series = []
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2
        series.append(np.sum(e_mode))
    return np.array(series, dtype=np.float64)


# Purpose:
#   Choose the saturated averaging window used by downstream ML targets.
# How it works:
#   Splits a smooth scalar into blocks and finds the stable tail near the final reference level.
def detect_saturation_window(
    series,
    num_blocks=12,
    tolerance=0.10,
    reference_blocks=3,
    min_window_fraction=0.25,
    min_window_points=30,
):
    """Detect a saturated averaging window from a smooth scalar time series.

    Project rule:
        E = smooth scalar per saved frame, preferably total kinetic energy.
        K = 12 blocks.
        ref = mean of the final 3 block means.
        onset = first block after which all later block means remain within
        10% of ref; otherwise fall back to K//2.

    Extra safeguards:
        1. Always returns (window, info) with a consistent metadata dictionary.
        2. Keeps a denominator floor for ref ~ 0.
        3. Enforces a minimum averaging window so a run cannot pass with only a
           few saved frames / correlation times.
    """
    E = np.asarray(series, dtype=np.float64)
    T = len(E)
    K = num_blocks

    if T == 0:
        info = {
            "T": 0,
            "t_start": 0,
            "window_length": 0,
            "window_fraction": 0.0,
            "block_means": np.array([], dtype=np.float64),
            "ref": np.nan,
            "onset_block": None,
            "used_fallback": True,
            "guard_adjusted": False,
            "reason": "empty_series",
        }
        return range(0, 0), info

    if T < K:
        means = np.array([E.mean()], dtype=np.float64)
        ref = means[-1]
        onset = 0
        t_start = T // 2
        used_fallback = True
        reason = "too_few_points_for_blocks"
    else:
        means = np.array([b.mean() for b in np.array_split(E, K)])
        ref = means[-reference_blocks:].mean()
        denom = max(abs(ref), 1e-20)
        onset_found = next(
            (i for i in range(K) if np.all(np.abs(means[i:] - ref) / denom < tolerance)),
            None,
        )
        used_fallback = onset_found is None
        onset = K // 2 if used_fallback else onset_found
        reason = "fallback_no_stable_block" if used_fallback else "detected"
        t_start = min(onset * (T // K), T - 1)

    min_required = max(int(np.ceil(min_window_fraction * T)), int(min_window_points))
    min_required = min(T, min_required)
    guard_adjusted = False
    if T - t_start < min_required:
        t_start = max(0, T - min_required)
        guard_adjusted = True
        reason = f"{reason}+minimum_window_guard"

    window = range(t_start, T)
    info = {
        "T": T,
        "t_start": t_start,
        "window_length": len(window),
        "window_fraction": len(window) / T,
        "block_means": means,
        "ref": ref,
        "onset_block": onset,
        "used_fallback": used_fallback,
        "guard_adjusted": guard_adjusted,
        "reason": reason,
    }
    return window, info
