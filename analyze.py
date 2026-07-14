"""Quick-look plots for completed Tokam2D simulations or scan HDF5 files.

Run from the project root and provide the simulation output folder:

    python results/analyze.py results/test_1
    python results/analyze.py results/my_run

Or from inside the results folder:

    python analyze.py test_1

The input folder must contain ``simulation_fields.h5``.

This version also supports the scan files used by the ML scripts:

    python analyze.py /path/to/scan_IIIA_512
    python analyze.py /path/to/hwak_C0.5.h5

If a directory contains ``hwak_C*.h5`` files, all matching files are analyzed.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCAN_DATA_DIR = Path("/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512")
DEFAULT_SCAN_OUTPUT_DIR = "scan_analysis_outputs"


def find_project_root(start: Path) -> Path:
    """Find the Tokam2D project root from this script's location."""
    for candidate in [start, *start.parents]:
        if (candidate / "diagnostics" / "simulation_diag_handler.py").exists():
            return candidate
    raise RuntimeError(
        "Could not find Tokam2D project root. Expected to find "
        "diagnostics/simulation_diag_handler.py in this folder or a parent."
    )


try:
    PROJECT_ROOT = find_project_root(SCRIPT_DIR)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from diagnostics.simulation_diag_handler import Simulation, set_plot_defaults  # noqa: E402
except Exception as exc:  # pragma: no cover - used when only scan-HDF5 mode is needed
    PROJECT_ROOT = None
    Simulation = None

    def set_plot_defaults():
        """Fallback plot defaults when Tokam2D diagnostics are unavailable."""
        plt.rcParams.update({"figure.dpi": 120})

    TOKAM2D_IMPORT_ERROR = exc
else:
    TOKAM2D_IMPORT_ERROR = None


def require_python_environment() -> None:
    """Require an active Python environment before running diagnostics."""
    venv = os.environ.get("VIRTUAL_ENV")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    if not venv and not conda_env:
        raise RuntimeError(
            "No active Python environment detected.\n"
            "Activate the Tokam2D environment first, for example:\n"
            "  source tokam_env/bin/activate"
        )

    active_name = Path(venv).name if venv else conda_env
    if active_name != "tokam_env":
        print(
            f"Warning: active Python environment appears to be '{active_name}', "
            "not 'tokam_env'. Continuing anyway."
        )


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n\n"
            "Provide a completed Tokam2D output folder, e.g.\n"
            "  python results/analyze.py results/test_1"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create quick-look plots for a Tokam2D folder or hwak_C*.h5 scan files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Either a Tokam2D output folder containing simulation_fields.h5, "
            "a directory containing hwak_C*.h5 scan files, or one hwak_C*.h5 file."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_SCAN_OUTPUT_DIR),
        help="Output directory for scan-HDF5 plots and summary CSV.",
    )
    return parser.parse_args()


def resolve_input_path(input_path: Path) -> Path:
    """Resolve user input, falling back to the default scan data directory."""
    expanded = input_path.expanduser()
    if expanded.exists():
        return expanded.resolve()

    if not expanded.is_absolute():
        default_candidate = DEFAULT_SCAN_DATA_DIR / expanded
        if default_candidate.exists():
            return default_candidate.resolve()

    return expanded.resolve()


def select_field(h5_file: h5py.File) -> tuple[str, str]:
    preferred = [
        ("density", "Density"),
        ("n", "Density (n)"),
        ("potential", "Potential"),
        ("phi", "Potential (phi)"),
    ]
    for key, label in preferred:
        if key in h5_file and len(h5_file[key].shape) == 3:
            return key, label

    fields_3d = [key for key in h5_file.keys() if len(getattr(h5_file[key], "shape", ())) == 3]
    if not fields_3d:
        raise KeyError("No 3D time-dependent field found in simulation_fields.h5.")
    return fields_3d[0], fields_3d[0]


def field_colormap(field: str) -> str:
    if field in ["density", "n"]:
        return "inferno"
    return "Spectral_r"


def field_label(field: str) -> str:
    """Return a readable plot label for a simulation field."""
    labels = {
        "density": "Density",
        "potential": "Potential",
        "vorticity": "Vorticity",
        "flux": "Radial flux",
        "VEx": "Radial ExB velocity",
        "VEy": "Poloidal ExB velocity",
    }
    return labels.get(field, field)


def scan_field_title_label(field: str) -> str:
    """Return the requested scan-plot title label with field symbols."""
    labels = {
        "density": "Density (n)",
        "potential": "Potential (φ)",
        "vorticity": "Vorticity (ω)",
    }
    return labels.get(field, field_label(field))


def read_scalar(sim: Simulation, key: str):
    value = sim[key]
    if value is None:
        raise KeyError(key)
    try:
        value = value[()]
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def parameter_summary(sim: Simulation) -> str:
    """Build a compact parameter summary shown between titles."""
    try:
        eq = read_scalar(sim, "eq")
    except Exception:
        eq = None

    def scalar_text(key: str, fmt: str = ".3g"):
        try:
            value = read_scalar(sim, key)
        except Exception:
            return None
        try:
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.size > 0:
                value = arr.flat[0]
        except Exception:
            pass
        try:
            return f"{key}={format(float(value), fmt)}"
        except Exception:
            return f"{key}={value}"

    keys_by_model = {
        "HW": ["C", "kappa", "g", "Dn", "Dphi"],
        "mHW": ["C", "kappa", "g", "Dn", "Dphi"],
        "BHW": ["C", "kappa", "g", "Dn", "Dphi"],
        "SOL": ["sigma_nn", "sigma_nphi", "sigma_phiphi", "sigma_phin", "g", "Dn", "Dphi"],
    }
    keys = keys_by_model.get(eq, ["Dn", "Dphi"])
    parts = [text for text in (scalar_text(key) for key in keys) if text is not None]
    return ", ".join(parts)


def simulation_title(sim: Simulation) -> str:
    """Return a readable model title for plot headers."""
    try:
        eq = read_scalar(sim, "eq")
    except Exception:
        eq = None

    title_map = {
        "HW": "Hasegawa-Wakatani",
        "mHW": "Modified Hasegawa-Wakatani",
        "BHW": "Flux-Balanced Hasegawa-Wakatani",
        "SOL": "SOL",
    }
    return title_map.get(eq, "Tokam2D")


def time_unit_label(sim: Simulation) -> str:
    """Return a compact time-unit label for final-state plot titles."""
    try:
        eq = read_scalar(sim, "eq")
    except Exception:
        eq = None
    if eq in ["HW", "mHW", "BHW"]:
        return r"$[L/c_0]$"
    return r"$[\omega_{c0}^{-1}]$"


def save_particle_flux_csv(sim: Simulation, sim_folder: Path) -> Path | None:
    """Save domain-averaged radial particle flux versus saved time-step index."""
    output_csv = sim_folder / "particle_flux_vs_time_step.csv"
    rows = []

    for time_step, time_value in enumerate(np.asarray(sim.time)):
        try:
            flux = np.asarray(sim.get_data_slice("flux", it=time_step))
        except Exception as exc:
            print(f"Could not compute particle flux at time_step={time_step}: {exc}")
            return None
        rows.append((time_step, float(time_value), float(np.mean(flux))))

    np.savetxt(
        output_csv,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="time_step,time,particle_flux",
        comments="",
        fmt=["%d", "%.18e", "%.18e"],
    )
    print(f"Saved particle flux CSV: {output_csv}")
    return output_csv


def normalize_frame(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize data to [0, 1] for plotting and colorbar display."""
    scale = vmax - vmin
    if scale == 0:
        return np.zeros_like(data, dtype=float)
    normalized = (data - vmin) / scale
    return np.clip(normalized, 0.0, 1.0)


def draw_final_state_plot(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    field_key: str,
    field_label: str,
    time_value: float,
    output_image: Path,
    model_title: str,
    time_unit: str,
    parameter_text: str,
) -> None:
    """Save a final-state plot using the same explicit layout as GIF frames."""
    x_span = float(x[-1] - x[0]) if len(x) > 1 else 1.0
    y_span = float(y[-1] - y[0]) if len(y) > 1 else 1.0
    domain_aspect = y_span / x_span if x_span > 0 else 1.0

    fig_width = 5.2
    fig_height = max(4.4, min(6.2, fig_width * max(domain_aspect, 0.8)))

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    if vmin == vmax:
        vmax = vmin + 1.0

    data = normalize_frame(data, vmin, vmax)

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=140)

    # Match generate_graphics.py: explicit axes keep colorbar labels from being
    # clipped and force the colorbar height to match the plotted field box.
    plot_left = 0.12
    plot_bottom = 0.13
    plot_height = 0.72
    plot_width = plot_height / domain_aspect if domain_aspect > 0 else plot_height
    max_plot_width = 0.72
    if plot_width > max_plot_width:
        plot_width = max_plot_width
        plot_height = plot_width * domain_aspect
        plot_bottom = 0.13 + (0.72 - plot_height) / 2.0

    if data.shape == (len(x), len(y)) and data.shape != (len(y), len(x)):
        data = data.T

    ax = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
    cax = fig.add_axes([plot_left + plot_width + 0.035, plot_bottom, 0.025, plot_height])

    mesh = ax.pcolormesh(
        x,
        y,
        data,
        cmap=field_colormap(field_key),
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.set_label(f"Normalized {field_label}", fontsize=9)

    fig.suptitle(model_title, fontsize=12, fontweight="bold", y=0.985)
    if parameter_text:
        fig.text(0.5, 0.935, parameter_text, ha="center", va="center", fontsize=7)
    ax.set_title(f"{field_label} at time = {time_value:.2f} {time_unit}", fontsize=9, pad=3)
    ax.set_xlabel(r"x [$\rho_0$]", fontsize=9)
    ax.set_ylabel(r"y [$\rho_0$]", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_final_state_plots(
    sim: Simulation,
    sim_folder: Path,
    x: np.ndarray,
    y: np.ndarray,
    time_value: float,
    model_title: str,
    time_unit: str,
    parameter_text: str,
) -> list[Path]:
    """Save final-state plots for density, vorticity, and potential."""
    output_paths: list[Path] = []
    final_state_fields = ["density", "vorticity", "potential"]

    for field in final_state_fields:
        try:
            data = np.asarray(sim.get_data_slice(field, it=-1))
        except Exception as exc:
            print(f"Skipping final-state plot for '{field}': {exc}")
            continue

        output_image = sim_folder / f"endstate_{field}.png"
        draw_final_state_plot(
            x,
            y,
            data,
            field,
            field_label(field),
            time_value,
            output_image,
            model_title,
            time_unit,
            parameter_text,
        )
        print(f"Saved final-state plot: {output_image}")
        output_paths.append(output_image)

    return output_paths


# Purpose:
#   Find the ML scan files in a directory.
# How it works:
#   Selects all files matching hwak_C*.h5 and sorts them by the C value stored
#   inside the file when possible, falling back to filename order otherwise.
def list_scan_h5_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("hwak_C*.h5"))

    def c_sort_key(path: Path):
        try:
            with h5py.File(path, "r") as f:
                return float(f["params/C"][()])
        except Exception:
            return path.name

    return sorted(files, key=c_sort_key)


# Purpose:
#   Infer the real-space grid shape from the scan-file Fourier array.
# How it works:
#   The scan files store real-FFT data with shape [time, channel, nx, ny//2+1],
#   so the full real-space y length is 2*(stored_y_length - 1).
def infer_scan_real_shape(uk_dataset) -> tuple[int, int]:
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


# Purpose:
#   Reconstruct final density, potential, and vorticity fields from one scan HDF5 file.
# How it works:
#   Reads the final Fourier snapshot, applies irfft2 to density and potential,
#   and computes vorticity as omega_k = -k^2 phi_k before transforming to real space.
def load_final_scan_fields(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        c_val = float(f["params/C"][()]) if "params/C" in f else np.nan
        uk = f["fields/uk"]
        kx2d = f["data/kx"][()]
        ky2d = f["data/ky"][()]
        nx, ny = infer_scan_real_shape(uk)
        final_idx = uk.shape[0] - 1

        phi_k = uk[final_idx, 0, :, :]
        if "fields/nk" in f:
            n_k = f["fields/nk"][final_idx, 0, :, :]
        else:
            n_k = uk[final_idx, 1, :, :]

        k2 = kx2d**2 + ky2d**2
        omega_k = -k2 * phi_k

        phi = np.fft.irfft2(phi_k, s=(nx, ny), norm="forward")
        density = np.fft.irfft2(n_k, s=(nx, ny), norm="forward")
        vorticity = np.fft.irfft2(omega_k, s=(nx, ny), norm="forward")

    return {
        "file": h5_path.name,
        "C": c_val,
        "T": final_idx + 1,
        "nx": nx,
        "ny": ny,
        "fields": {
            "density": density,
            "potential": phi,
            "vorticity": vorticity,
        },
    }


# Purpose:
#   Save one final-state field image from a scan HDF5 file.
# How it works:
#   Uses imshow with robust percentile color limits so all scan-file plots are
#   directly readable even when amplitudes vary strongly across C.
def save_scan_field_plot(data: np.ndarray, field: str, c_val: float, output_image: Path) -> None:
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    if vmin == vmax:
        vmax = vmin + 1.0

    normalized_data = normalize_frame(data, vmin, vmax)
    nx, ny = normalized_data.shape

    fig = plt.figure(figsize=(5.6, 5.6), dpi=140)
    plot_left = 0.12
    plot_bottom = 0.12
    plot_width = 0.72
    plot_height = 0.72
    ax = fig.add_axes([plot_left, plot_bottom, plot_width, plot_height])
    cax = fig.add_axes([plot_left + plot_width + 0.04, plot_bottom, 0.025, plot_height])

    im = ax.imshow(
        normalized_data.T,
        origin="lower",
        cmap=field_colormap(field),
        vmin=0.0,
        vmax=1.0,
        extent=[0, nx, 0, ny],
        aspect="equal",
    )
    ax.set_title(f"{scan_field_title_label(field)} for C = {c_val:.4g}", fontsize=12, pad=10)
    ax.set_xlabel(r"x [$\rho_0$]", fontsize=10)
    ax.set_ylabel(r"y [$\rho_0$]", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(color="0.6", linewidth=0.8, alpha=0.55)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.linspace(0.0, 1.0, 6))
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(f"Normalized {field_label(field)}", fontsize=10)

    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# Purpose:
#   Analyze one direct hwak_C*.h5 scan file.
# How it works:
#   Reconstructs final density, potential, and vorticity fields, saves plots,
#   and returns scalar RMS values for the scan summary CSV.
def analyze_one_scan_file(h5_path: Path, output_dir: Path) -> dict:
    data = load_final_scan_fields(h5_path)
    c_val = data["C"]
    safe_c = f"C_{c_val:.6g}".replace(".", "p").replace("-", "m")
    file_stem = h5_path.stem
    row = {
        "file": data["file"],
        "C": c_val,
        "T": data["T"],
        "nx": data["nx"],
        "ny": data["ny"],
    }

    for field, field_data in data["fields"].items():
        output_image = output_dir / f"{safe_c}_{file_stem}_{field}.png"
        save_scan_field_plot(field_data, field, c_val, output_image)
        row[f"{field}_rms"] = float(np.sqrt(np.mean(field_data**2)))
        row[f"{field}_min"] = float(np.min(field_data))
        row[f"{field}_max"] = float(np.max(field_data))
        print(f"Saved scan plot: {output_image}")

    return row


# Purpose:
#   Analyze all 17 scan-style simulation files in a directory.
# How it works:
#   Loops through hwak_C*.h5 files, analyzes each final state, and writes a CSV
#   containing C, grid size, number of snapshots, and basic field amplitudes.
def analyze_scan_directory(input_path: Path, output_dir: Path) -> None:
    files = [input_path] if input_path.is_file() else list_scan_h5_files(input_path)
    if not files:
        raise FileNotFoundError(f"No hwak_C*.h5 files found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} scan HDF5 file(s).")
    print(f"Writing scan analysis outputs to: {output_dir}")

    rows = []
    for h5_path in files:
        print(f"\nAnalyzing scan file: {h5_path}")
        rows.append(analyze_one_scan_file(h5_path, output_dir))

    summary_path = output_dir / "scan_analysis_summary.csv"
    columns = list(rows[0].keys())
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in columns) + "\n")
    print(f"\nSaved scan summary CSV: {summary_path}")


# Purpose:
#   Preserve the original single-folder Tokam2D analysis path.
# How it works:
#   Loads simulation_fields.h5 through Tokam2D diagnostics, saves particle flux,
#   and creates final-state plots for density, vorticity, and potential.
def analyze_tokam2d_folder(sim_folder: Path) -> None:
    require_python_environment()
    if Simulation is None:
        raise RuntimeError(
            "Tokam2D diagnostics could not be imported, so folder mode is unavailable. "
            f"Original import error: {TOKAM2D_IMPORT_ERROR}"
        )

    fields_path = sim_folder / "simulation_fields.h5"
    metadata_path = sim_folder / "metadata.h5"

    require_file(fields_path, "simulation fields file")

    sim = None
    model_title = "Tokam2D"
    time_unit = ""
    parameter_text = ""
    try:
        sim = Simulation(str(sim_folder))
        model_title = simulation_title(sim)
        time_unit = time_unit_label(sim)
        parameter_text = parameter_summary(sim)
    except Exception as exc:
        print(f"Could not read Simulation metadata for plot titles/CSV: {exc}")

    print(f"Opening Tokam2D simulation folder: {sim_folder}")
    print(f"Reading field data from: {fields_path}")

    with h5py.File(fields_path, "r") as f:
        print("\nVariables available in simulation_fields.h5:")
        for key in f.keys():
            print(f" -> {key}: shape {getattr(f[key], 'shape', 'scalar')}")

        for key in ["x", "y", "time"]:
            if key not in f:
                raise KeyError(f"Required dataset '{key}' not found in {fields_path}")

        x = f["x"][()]
        y = f["y"][()]
        time = f["time"][()]

    if parameter_text:
        print(f"Parameter summary: {parameter_text}")

    if sim is not None:
        save_particle_flux_csv(sim, sim_folder)
        output_images = save_final_state_plots(
            sim,
            sim_folder,
            x,
            y,
            float(time[-1]),
            model_title,
            time_unit,
            parameter_text,
        )
    else:
        output_images = []
        print("Skipping final-state plots because Simulation data could not be loaded.")

    if output_images:
        print("\nAnalysis complete. Heatmaps saved to:")
        for output_image in output_images:
            print(f" -> {output_image}")
    else:
        print("\nAnalysis complete, but no final-state heatmaps were saved.")

    if metadata_path.exists():
        with h5py.File(metadata_path, "r") as f:
            if "simulation_duration" in f:
                print(f"The simulation took {float(f['simulation_duration'][()]):.2f} seconds to run.")
            else:
                print("metadata.h5 exists, but has no 'simulation_duration' dataset.")
    else:
        print("metadata.h5 not found; skipping runtime summary.")


def main() -> None:
    args = parse_args()
    set_plot_defaults()
    input_path = resolve_input_path(args.input_path)
    output_dir = args.output_dir.expanduser().resolve()

    if input_path.is_file() and input_path.name.startswith("hwak_C") and input_path.suffix == ".h5":
        analyze_scan_directory(input_path, output_dir)
        return

    if input_path.is_dir() and (input_path / "simulation_fields.h5").exists():
        analyze_tokam2d_folder(input_path)
        return

    if input_path.is_dir() and list_scan_h5_files(input_path):
        analyze_scan_directory(input_path, output_dir)
        return

    raise FileNotFoundError(
        "Input path is not a recognized analysis target. Expected one of:\n"
        "  1. Tokam2D folder containing simulation_fields.h5\n"
        "  2. Directory containing hwak_C*.h5 scan files\n"
        "  3. Single hwak_C*.h5 scan file\n"
        f"Received: {input_path}"
    )


if __name__ == "__main__":
    main()
