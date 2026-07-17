import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from plasma_saturation import detect_saturation_window

# -----------------------------------------------------------------------------
# Step 3 Training: 2D Statistical Map Regression vs C
# -----------------------------------------------------------------------------
# TARGET_TYPE options:
#   "rms"      : train RMS map only
#   "flux"     : train local flux-magnitude map only
#   "spectrum" : train 2D potential spectral-power map only
#   "all"      : convenience mode; trains rms, flux, spectrum PER FAMILY
#
# Important: "all" is intentionally per-family. It does NOT stack
# [rms | flux | spectrum] into one target vector and does NOT fit one shared PCA.
# Instead, each family gets its own PCA/POD basis, FFNN coefficient model,
# interpolation baseline, and score.
#
# Spectrum scoring uses only informative low modes to avoid the high-|k| dead zone:
#   fold +kx/-kx -> |kx|, then score the ~40x40 box |kx|=1..40 and ky=0..39.
# -----------------------------------------------------------------------------

DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
TARGET_TYPE = "all"  # "rms", "flux", "spectrum", or "all"
N_POD_MODES = 4
EPOCHS = 1200
LR = 0.01
EPS = 1e-20
LOW_KX_MODES = 40
LOW_KY_MODES = 40


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Purpose:
#   Recover real-space grid dimensions from a real-FFT Fourier dataset.
# How it works:
#   Uses the stored x length and reconstructs y from the Hermitian half-spectrum.
def infer_real_shape(uk_dataset):
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


# Purpose:
#   Build a smooth scalar time series used to identify saturated turbulence.
# How it works:
#   Sums 1/2 k^2 |phi_k|^2 over Fourier modes for each saved frame.
def compute_total_kinetic_energy_series(uk, kx2d, ky2d):
    series = []
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2
        series.append(np.sum(e_mode))
    return np.array(series, dtype=np.float64)


# Purpose:
#   Build deterministic 2D statistical targets from saturated snapshots.
# How it works:
#   Averages RMS potential, flux, and spectral power over the saturated window.
def extract_step4_maps(h5_file):
    c_val = h5_file["params/C"][()]
    uk = h5_file["fields/uk"]
    kx2d = h5_file["data/kx"][()]
    ky2d = h5_file["data/ky"][()]
    nx, ny = infer_real_shape(uk)
    T = uk.shape[0]
    ke_series = compute_total_kinetic_energy_series(uk, kx2d, ky2d)
    window, sat_info = detect_saturation_window(ke_series)
    t_start = sat_info["t_start"]
    block_means = sat_info["block_means"]
    n_t = len(window)

    phi_sq_accum = np.zeros((nx, ny), dtype=np.float64)
    flux_accum = np.zeros((nx, ny), dtype=np.float64)
    spectrum_accum = np.zeros((nx, ny), dtype=np.float64)

    for t in window:
        phi_k = uk[t, 0, :, :]
        n_k = h5_file["fields/nk"][t, 0, :, :] if "fields/nk" in h5_file else uk[t, 1, :, :]
        grady_phi_k = 1j * ky2d * phi_k
        phi = np.fft.irfft2(phi_k, s=(nx, ny), norm="forward")
        density = np.fft.irfft2(n_k, s=(nx, ny), norm="forward")
        grady_phi = np.fft.irfft2(grady_phi_k, s=(nx, ny), norm="forward")
        phi_sq_accum += phi**2
        flux_accum += -density * grady_phi
        spectrum_accum += np.abs(np.fft.fft2(phi, norm="forward")) ** 2

    rms_map = np.sqrt(phi_sq_accum / n_t)
    flux_map = flux_accum / n_t
    spectrum_map = spectrum_accum / n_t
    flux_scale = 1e-2 * np.median(np.abs(flux_map))  # robust per-run scale for signed flux
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        flux_scale = EPS
    return {
        "C": c_val,
        "shape": (nx, ny),
        "rms": np.log10(rms_map + EPS),
        "flux": np.arcsinh(flux_map / flux_scale),
        "spectrum": np.log10(spectrum_map + EPS),
    }


# Purpose:
#   Build Step 3 deterministic 2D map regression targets.
# How it works:
#   Flattens saturated 2D maps for PCA/POD compression and FFNN coefficient prediction.
class Step4MapDataset(Dataset):
    def __init__(self, data_dir, target_type="spectrum"):
        if target_type not in {"rms", "flux", "spectrum"}:
            raise ValueError("target_type must be 'rms', 'flux', or 'spectrum'. Use main loop for 'all'.")
        self.data_dir = data_dir
        self.target_type = target_type
        self.file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".h5") and f.startswith("hwak_C"))
        self.raw_c_values, self.inputs, self.targets = [], [], []
        self.grid_shape = None
        self._process_files()

    def _process_files(self):
        print(f"Extracting Step 3 target='{self.target_type}' maps...")
        for file_name in self.file_list:
            with h5py.File(os.path.join(self.data_dir, file_name), "r") as f:
                maps = extract_step4_maps(f)
            self.grid_shape = maps["shape"]
            c_val = maps["C"]
            self.raw_c_values.append(c_val)
            self.inputs.append([np.log10(c_val)])
            self.targets.append(maps[self.target_type].reshape(-1))
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def unflatten(self, flat):
        return flat.reshape(self.grid_shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


# Purpose:
#   Predict POD/PCA coefficients from log10(C).
# How it works:
#   Uses a small fully connected network to output low-dimensional modal coefficients.
class PodCoefficientFFNN(nn.Module):
    def __init__(self, input_dim=1, num_modes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, num_modes),
        )
    def forward(self, x):
        return self.network(x)


# Purpose:
#   Convert a 2D log spectrum into a folded |kx| representation.
# How it works:
#   Adds +kx and -kx power in linear space, then converts back to log10 power.
def fold_kx_log_spectrum(log_map, n_kx=LOW_KX_MODES):
    """Fold +kx and -kx in linear power, then return log10 folded spectrum."""
    n_kx = min(n_kx, (log_map.shape[0] - 1) // 2)
    linear = 10.0 ** log_map
    folded = linear[1:n_kx + 1, :] + linear[-n_kx:, :][::-1, :]
    return np.log10(folded + EPS)


# Purpose:
#   Score Step 3 predictions on physically informative target regions.
# How it works:
#   Uses median log error for maps and focuses spectra on the folded low-mode box.
def informative_error(pred_flat, true_flat, dataset):
    pred_map = dataset.unflatten(pred_flat)
    true_map = dataset.unflatten(true_flat)
    if dataset.target_type != "spectrum":
        return float(np.median(np.abs(pred_map - true_map)))
    pred_fold = fold_kx_log_spectrum(pred_map, LOW_KX_MODES)
    true_fold = fold_kx_log_spectrum(true_map, LOW_KX_MODES)
    n_ky = min(LOW_KY_MODES, pred_fold.shape[1])  # ky=0..39 by default
    # Score only the informative low-mode box: folded |kx|=1..40 and ky=0..39.
    return float(np.median(np.abs(pred_fold[:, :n_ky] - true_fold[:, :n_ky])))


def error_unit(target_type):
    return "dex" if target_type in {"rms", "spectrum"} else "transformed units"


def robust_percentile_limits(*arrays, percentiles=(2, 98), fallback=(0.0, 1.0)):
    values = np.concatenate([np.ravel(np.asarray(arr)) for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    vmin, vmax = np.nanpercentile(values, percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        center = float(np.nanmedian(values)) if values.size else 0.0
        spread = float(np.nanstd(values)) if values.size else 0.0
        if not np.isfinite(spread) or spread <= 0:
            spread = 1.0
        return center - spread, center + spread
    return float(vmin), float(vmax)


def map_for_display(dataset, flat):
    img = dataset.unflatten(flat)
    if dataset.target_type == "spectrum":
        return fold_kx_log_spectrum(img, LOW_KX_MODES)[:, :LOW_KY_MODES]
    return img


def select_hydrodynamic_row(df, max_c=1e-2, preferred_c=0.008975):
    hydro = df[df["C_Value"] < max_c]
    candidates = hydro if len(hydro) else df
    idx = (np.log10(candidates["C_Value"].astype(float)) - np.log10(preferred_c)).abs().idxmin()
    return candidates.loc[idx]


# Purpose:
#   Show one held-out Step 3 prediction beside its simulation target.
# How it works:
#   Plots true map, POD-FFNN map, and absolute error map for a representative C.
def plot_validation_panel(dataset, row, output_name):
    true_map = map_for_display(dataset, row["True_Target"])
    pred_map = map_for_display(dataset, row["NN_Target"])
    if dataset.target_type == "spectrum":
        title = r"Folded low-mode spectrum $P_\phi(|k_x|,k_y)$"
        xlabel, ylabel = r"$|k_x|$ mode", r"$k_y$ mode"
    else:
        title = f"{dataset.target_type} spatial map"
        xlabel, ylabel = "x coordinate", "y coordinate"
    error_map = np.abs(pred_map - true_map)
    vmin, vmax = robust_percentile_limits(true_map, pred_map)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, img, ttl, cmap in [
        (axs[0], true_map, "Simulation", "inferno"),
        (axs[1], pred_map, "ML prediction", "inferno"),
        (axs[2], error_map, f"Prediction error [{error_unit(dataset.target_type)}]", "viridis"),
    ]:
        # Transpose because imshow places array columns on the x-axis and rows
        # on the y-axis. The map arrays are stored as (x or kx, y or ky), so
        # img.T displays x/kx horizontally and y/ky vertically.
        im = ax.imshow(img.T, origin="lower", cmap=cmap, vmin=vmin if cmap == "inferno" else None, vmax=vmax if cmap == "inferno" else None)
        ax.set_title(ttl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"{title} for held-out C={row['C_Value']:.4g}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


# Purpose:
#   Inspect Step 3 spectral prediction quality across ky modes and C.
# How it works:
#   Reduces folded 2D spectra to mean ky profiles and plots simulation, prediction, and error.
def plot_spectrum_mode_sweep(dataset, df, output_name):
    if dataset.target_type != "spectrum":
        return
    modes = np.arange(0, LOW_KY_MODES)
    c_values = df["C_Value"].values
    true_profiles, pred_profiles = [], []
    for _, row in df.iterrows():
        true_fold = fold_kx_log_spectrum(dataset.unflatten(row["True_Target"]), LOW_KX_MODES)
        pred_fold = fold_kx_log_spectrum(dataset.unflatten(row["NN_Target"]), LOW_KX_MODES)
        true_profiles.append(np.mean(true_fold[:, :LOW_KY_MODES], axis=0))
        pred_profiles.append(np.mean(pred_fold[:, :LOW_KY_MODES], axis=0))
    true_profiles, pred_profiles = np.array(true_profiles), np.array(pred_profiles)
    vmin, vmax = robust_percentile_limits(true_profiles, pred_profiles)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    im0 = axs[0].pcolormesh(modes, c_values, true_profiles, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_yscale("log"); axs[0].set_title(r"Simulation: $\log_{10}P_\phi(k_y;C)$")
    axs[0].set_xlabel(r"Poloidal mode $k_y$"); axs[0].set_ylabel(r"Adiabaticity $C$")
    plt.colorbar(im0, ax=axs[0], label=r"Spectral power")
    im1 = axs[1].pcolormesh(modes, c_values, pred_profiles, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    axs[1].set_yscale("log"); axs[1].set_title(r"ML prediction: $\log_{10}P_\phi(k_y;C)$")
    axs[1].set_xlabel(r"Poloidal mode $k_y$"); axs[1].set_ylabel(r"Adiabaticity $C$")
    plt.colorbar(im1, ax=axs[1], label=r"Spectral power")
    profile_err = np.median(np.abs(pred_profiles - true_profiles), axis=1)
    axs[2].plot(c_values, profile_err, marker="o")
    axs[2].set_xscale("log"); axs[2].set_title("Low-mode profile error")
    axs[2].set_xlabel(r"Adiabaticity $C$"); axs[2].set_ylabel("Median error [dex]")
    axs[2].grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout(); plt.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


# Purpose:
#   Compare POD-FFNN error against the interpolation baseline.
# How it works:
#   Plots held-out median error versus C for both model families.
def plot_error_comparison(df, target_type, output_name):
    """Plot held-out POD-FFNN error against the non-ML interpolation baseline."""
    c_values = df["C_Value"].values
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(c_values, df["NN_DEX"].values, marker="o", lw=2.2, label="POD-FFNN prediction")
    ax.plot(c_values, df["Base_DEX"].values, marker="s", lw=2.0, ls="--", label="Linear interpolation baseline")
    if "Floor_DEX" in df:
        ax.plot(c_values, df["Floor_DEX"].values, marker="^", lw=1.8, ls=":", label="POD reconstruction check")
    ax.set_xscale("log")
    ax.set_xlabel(r"Adiabaticity $C$")
    ax.set_ylabel(f"Median absolute error [{error_unit(target_type)}]")
    ax.set_title(f"Step 4 {target_type} map error: ML vs interpolation")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


# Purpose:
#   Perform leave-one-C-out validation for a Step 3 target family.
# How it works:
#   Fits POD on training C values, trains an FFNN on coefficients, and compares against interpolation.
def run_loocv(dataset, device):
    results = []
    for test_idx in range(len(dataset)):
        test_c = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(len(dataset)) if i != test_idx]
        train_y, true_y = dataset.targets[train_indices], dataset.targets[test_idx]
        n_modes = min(N_POD_MODES, len(train_indices), train_y.shape[1])
        pca = PCA(n_components=n_modes)
        train_coeffs = pca.fit_transform(train_y)
        coeff_mean, coeff_std = train_coeffs.mean(axis=0), train_coeffs.std(axis=0) + 1e-8
        train_coeffs_scaled = (train_coeffs - coeff_mean) / coeff_std
        train_x = torch.tensor(dataset.inputs[train_indices], dtype=torch.float32, device=device)
        train_target = torch.tensor(train_coeffs_scaled, dtype=torch.float32, device=device)
        test_x = torch.tensor(dataset.inputs[test_idx], dtype=torch.float32, device=device).unsqueeze(0)
        baseline = interp1d(dataset.inputs[train_indices].squeeze(), train_y, axis=0, kind="linear", fill_value="extrapolate")
        base_pred_y = baseline(dataset.inputs[test_idx].squeeze())
        model = PodCoefficientFFNN(num_modes=n_modes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(EPOCHS):
            pred = model(train_x)
            loss = criterion(pred, train_target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            scaled_pred_coeffs = model(test_x).cpu().numpy().squeeze()
        pred_coeffs = scaled_pred_coeffs * coeff_std + coeff_mean
        nn_pred_y = pca.inverse_transform(pred_coeffs.reshape(1, -1)).squeeze()
        floor_y = pca.inverse_transform(pca.transform(true_y.reshape(1, -1))).squeeze()
        nn_error = informative_error(nn_pred_y, true_y, dataset)
        base_error = informative_error(base_pred_y, true_y, dataset)
        floor_error = informative_error(floor_y, true_y, dataset)
        results.append({"C_Value": test_c, "NN_DEX": nn_error, "Base_DEX": base_error,
                        "Floor_DEX": floor_error, "True_Target": true_y, "NN_Target": nn_pred_y,
                        "Base_Target": base_pred_y, "Floor_Target": floor_y})
        unit = error_unit(dataset.target_type)
        print(f" -> {dataset.target_type} C={test_c:.3e} | NN={nn_error:.4f} {unit} | baseline={base_error:.4f} {unit} | POD-recon={floor_error:.4f} {unit}")
    return pd.DataFrame(results).sort_values("C_Value")


def plot_hydrodynamic_holdout_grid(results_by_target, output_name="step3_hydrodynamic_holdout_grid.png"):
    target_order = [target for target in ["flux", "spectrum", "rms"] if target in results_by_target]
    if not target_order:
        return
    n_rows, n_cols = len(target_order), 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.2 * n_rows), squeeze=False)
    col_titles = ["Simulation", "POD-FFNN", "POD reconstruction", "|NN − truth|", "|POD recon − truth|"]
    for row_idx, target in enumerate(target_order):
        dataset, df = results_by_target[target]
        row = select_hydrodynamic_row(df)
        true_map = map_for_display(dataset, row["True_Target"])
        nn_map = map_for_display(dataset, row["NN_Target"])
        floor_map = map_for_display(dataset, row["Floor_Target"])
        nn_err_map = np.abs(nn_map - true_map)
        floor_err_map = np.abs(floor_map - true_map)
        field_vmin, field_vmax = robust_percentile_limits(true_map, nn_map, floor_map)
        err_vmin, err_vmax = robust_percentile_limits(nn_err_map, floor_err_map, percentiles=(0, 98), fallback=(0.0, 1.0))
        images = [true_map, nn_map, floor_map, nn_err_map, floor_err_map]
        cmaps = ["inferno", "inferno", "inferno", "viridis", "viridis"]
        titles = [
            col_titles[0],
            f"{col_titles[1]}\nerr {row['NN_DEX']:.3f}",
            f"{col_titles[2]}\nerr {row['Floor_DEX']:.3f}",
            col_titles[3],
            col_titles[4],
        ]
        for col_idx, (img, cmap, title) in enumerate(zip(images, cmaps, titles)):
            ax = axs[row_idx, col_idx]
            is_error = col_idx >= 3
            im = ax.imshow(
                img.T,
                origin="lower",
                cmap=cmap,
                vmin=err_vmin if is_error else field_vmin,
                vmax=err_vmax if is_error else field_vmax,
            )
            ax.set_title(title, fontweight="bold" if col_idx in {1, 2} else None)
            ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"{target} @ C={row['C_Value']:.4g}\n[{error_unit(target)}]", fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        "Step 3 hydrodynamic holdout comparison (C << 1)\n"
        "POD reconstruction is an L2 basis check; median-scored errors are not a strict floor.",
        fontweight="bold",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


# Purpose:
#   Train and evaluate one Step 3 target family.
# How it works:
#   Builds the dataset, runs LOOCV, prints errors, and saves validation figures.
def train_one_family(target_type, device):
    dataset = Step4MapDataset(DATA_DIR, target_type=target_type)
    if target_type == "spectrum":
        print(
            f"[Info] Spectrum score uses folded low-mode box: "
            f"|kx|=1..{LOW_KX_MODES}, ky=0..{LOW_KY_MODES - 1}."
        )
    df = run_loocv(dataset, device)
    print("\n" + "=" * 82)
    print(f"STEP 4 REPORT — TARGET: {target_type}".center(82))
    print("=" * 82)
    for _, row in df.iterrows():
        unit = error_unit(target_type)
        print(f"C={row['C_Value']:<10.4g} | NN={row['NN_DEX']:<8.4f} {unit} | baseline={row['Base_DEX']:<8.4f} {unit} | POD-recon={row['Floor_DEX']:<8.4f} {unit}")
    sample_row = select_hydrodynamic_row(df)
    plot_error_comparison(df, target_type, f"step3_{target_type}_error_comparison.png")
    plot_validation_panel(dataset, sample_row, f"step3_{target_type}_2d_map_validation.png")
    if target_type == "spectrum":
        plot_spectrum_mode_sweep(dataset, df, "step3_spectrum_mode_space_sweep.png")
    return dataset, df


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_list = ["rms", "flux", "spectrum"] if TARGET_TYPE == "all" else [TARGET_TYPE]
    all_results = {}
    for target in target_list:
        all_results[target] = train_one_family(target, device)
    plot_hydrodynamic_holdout_grid(all_results)