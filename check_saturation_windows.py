import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plasma_saturation import (
    compute_total_kinetic_energy_series,
    detect_saturation_window,
)


# -----------------------------------------------------------------------------
# Saturation-window validation utility
# -----------------------------------------------------------------------------
# This script is intentionally diagnostic-only. It checks whether the saturated
# averaging window is detected from a smooth scalar per run, not from a noisy
# intermittent quantity such as particle flux.
#
# Detection scalar used here:
#   E[t] = total kinetic energy at saved timestep index t
#
# Important plotting convention:
#   The x-axis is the saved timestep index: 0, 1, 2, ..., T-1.
#   It is not physical simulation time.
# -----------------------------------------------------------------------------

DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"

REPORT_CSV = "saturation_window_report.csv"
ENERGY_FIG = "saturation_energy_timeseries_all_runs.png"
SUMMARY_FIG = "saturation_window_summary.png"


def list_hwak_files(data_dir):
    return sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".h5") and f.startswith("hwak_C")
    )


def load_run_energy(file_path):
    """Load one run and compute total kinetic energy versus saved timestep index."""
    with h5py.File(file_path, "r") as f:
        c_val = float(f["params/C"][()])
        uk = f["fields/uk"]
        kx2d = f["data/kx"][()]
        ky2d = f["data/ky"][()]
        energy = compute_total_kinetic_energy_series(uk, kx2d, ky2d)
    return c_val, energy


def build_saturation_report(data_dir):
    """Scan all HDF5 runs and collect saturation metadata.

    Returns
    -------
    rows : list[dict]
        One row per run for CSV/reporting.
    run_data : list[dict]
        Includes the energy series for plotting.
    """
    rows = []
    run_data = []

    for file_name in list_hwak_files(data_dir):
        file_path = os.path.join(data_dir, file_name)
        c_val, energy = load_run_energy(file_path)
        window, info = detect_saturation_window(energy)

        rows.append({
            "file": file_name,
            "C": c_val,
            "T": info["T"],
            "t_start": info["t_start"],
            "window_length": info["window_length"],
            "window_fraction": info["window_fraction"],
            "onset_block": info["onset_block"],
            "used_fallback": info["used_fallback"],
            "guard_adjusted": info["guard_adjusted"],
            "reason": info["reason"],
            "ref": info["ref"],
        })

        run_data.append({
            "file": file_name,
            "C": c_val,
            "energy": energy,
            "window": window,
            "info": info,
        })

    rows = sorted(rows, key=lambda r: r["C"])
    run_data = sorted(run_data, key=lambda r: r["C"])
    return rows, run_data


def plot_energy_timeseries(run_data, output_path=ENERGY_FIG):
    """Plot total kinetic energy versus saved timestep index for all runs."""
    n_runs = len(run_data)
    colors = plt.cm.plasma(np.linspace(0, 1, max(n_runs, 1)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for color, run in zip(colors, run_data):
        energy = run["energy"]
        info = run["info"]
        steps = np.arange(len(energy))
        c_val = run["C"]
        label = f"C={c_val:.3g}"

        ax.plot(steps, energy, color=color, lw=1.4, alpha=0.85, label=label)
        ax.axvline(info["t_start"], color=color, ls="--", lw=0.9, alpha=0.65)

    ax.set_xlabel("Saved timestep index")
    ax.set_ylabel("Total kinetic energy scalar")
    ax.set_title("Saturation-window check: total kinetic energy vs saved timestep index")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    print(f"[Success] Saved '{output_path}'.")


def plot_summary(rows, output_path=SUMMARY_FIG):
    """Plot normalized onset/window diagnostics versus C."""
    df = pd.DataFrame(rows).sort_values("C")
    c = df["C"].values
    t_start_frac = df["t_start"].values / df["T"].values
    window_fraction = df["window_fraction"].values

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(c, t_start_frac, marker="o", lw=2, label=r"$t_{start}/T$")
    ax.plot(c, window_fraction, marker="s", lw=2, label="window fraction")

    fallback = df["used_fallback"].values.astype(bool)
    guarded = df["guard_adjusted"].values.astype(bool)
    if np.any(fallback):
        ax.scatter(c[fallback], t_start_frac[fallback], marker="x", s=90,
                   color="red", label="fallback used")
    if np.any(guarded):
        ax.scatter(c[guarded], window_fraction[guarded], marker="D", s=60,
                   facecolors="none", edgecolors="black", label="minimum-window guard")

    ax.set_xscale("log")
    ax.set_xlabel(r"Adiabaticity parameter $C$")
    ax.set_ylabel("Fraction of saved timesteps")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Saturation-window onset and averaging-window fraction")
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    print(f"[Success] Saved '{output_path}'.")


def main():
    rows, run_data = build_saturation_report(DATA_DIR)

    df = pd.DataFrame(rows).sort_values("C")
    df.to_csv(REPORT_CSV, index=False)
    print(f"[Success] Saved '{REPORT_CSV}'.")

    print("\nSaturation-window report:")
    print(df[[
        "C", "T", "t_start", "window_length", "window_fraction",
        "onset_block", "used_fallback", "guard_adjusted", "reason",
    ]].to_string(index=False))

    plot_energy_timeseries(run_data, ENERGY_FIG)
    plot_summary(rows, SUMMARY_FIG)


if __name__ == "__main__":
    main()