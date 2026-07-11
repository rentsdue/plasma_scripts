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

# -----------------------------------------------------------------------------
# Step 4 Training: 2D Statistical Map Regression vs C
# -----------------------------------------------------------------------------
# Supports selectable targets:
#   TARGET_TYPE = "rms"      -> log10 RMS potential fluctuation map
#   TARGET_TYPE = "flux"     -> log10 local particle-flux magnitude map
#   TARGET_TYPE = "spectrum" -> log10 2D potential spectral power map
#   TARGET_TYPE = "all"      -> 3-channel target [rms, flux, spectrum]
#
# Main model:
#   log10(C) -> FFNN -> POD/PCA coefficients -> inverse PCA -> 2D map
#
# Baseline:
#   linear interpolation in log10(C) directly on flattened target maps
# -----------------------------------------------------------------------------

DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
TARGET_TYPE = "all"  # Choose: "rms", "flux", "spectrum", or "all"
N_POD_MODES = 4
EPOCHS = 1200
LR = 0.01
EPS = 1e-20
SATURATION_START_FRACTION = 0.5


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_real_shape(uk_dataset):
    """Infer real-space grid shape from rfft2-style storage."""
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


def extract_step4_maps(h5_file, start_fraction=SATURATION_START_FRACTION):
    """Extract log-scaled Step 4 maps from one simulation file."""
    c_val = h5_file["params/C"][()]
    kappa = h5_file["params/kappa"][()] if "params/kappa" in h5_file else 1.0

    uk = h5_file["fields/uk"]
    ky2d = h5_file["data/ky"][()]
    nx, ny = infer_real_shape(uk)

    T = uk.shape[0]
    t0 = int(start_fraction * T)
    window = range(t0, T)
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
        flux_accum += -kappa * density * grady_phi
        spectrum_accum += np.abs(np.fft.fft2(phi, norm="forward")) ** 2

    rms_map = np.sqrt(phi_sq_accum / n_t)
    flux_map = flux_accum / n_t
    spectrum_map = spectrum_accum / n_t

    return {
        "C": c_val,
        "shape": (nx, ny),
        "rms": np.log10(rms_map + EPS),
        "flux": np.log10(np.abs(flux_map) + EPS),
        "spectrum": np.log10(spectrum_map + EPS),
    }


class Step4MapDataset(Dataset):
    def __init__(self, data_dir, target_type="spectrum"):
        if target_type not in {"rms", "flux", "spectrum", "all"}:
            raise ValueError("target_type must be 'rms', 'flux', 'spectrum', or 'all'")

        self.data_dir = data_dir
        self.target_type = target_type
        self.file_list = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".h5") and f.startswith("hwak_C")
        )

        self.raw_c_values = []
        self.inputs = []
        self.targets = []
        self.target_maps = {
            "rms": [],
            "flux": [],
            "spectrum": [],
        }
        self.grid_shape = None

        self._process_files()

    def _select_target(self, maps):
        if self.target_type == "all":
            # Shape: [3, nx, ny]
            return np.stack([maps["rms"], maps["flux"], maps["spectrum"]], axis=0)

        # Shape: [1, nx, ny]
        return maps[self.target_type][None, :, :]

    def _process_files(self):
        print(f"Extracting Step 4 target='{self.target_type}' maps...")

        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, "r") as f:
                maps = extract_step4_maps(f)

            self.grid_shape = maps["shape"]
            c_val = maps["C"]
            selected = self._select_target(maps)

            self.raw_c_values.append(c_val)
            self.inputs.append([np.log10(c_val)])
            self.targets.append(selected.reshape(-1))

            for key in self.target_maps:
                self.target_maps[key].append(maps[key])

        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        for key in self.target_maps:
            self.target_maps[key] = np.array(self.target_maps[key], dtype=np.float32)

    @property
    def n_channels(self):
        return 3 if self.target_type == "all" else 1

    def channel_names(self):
        if self.target_type == "all":
            return ["RMS amplitude", "Local flux", "Spectral power"]
        return [self.target_type]

    def unflatten(self, flat):
        nx, ny = self.grid_shape
        return flat.reshape(self.n_channels, nx, ny)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


class PodCoefficientFFNN(nn.Module):
    def __init__(self, input_dim=1, num_modes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

    def forward(self, x):
        return self.network(x)


def median_dex(pred, true):
    """Median absolute error in log10 space, i.e. dex."""
    return float(np.median(np.abs(pred - true)))


def plot_validation_panel(dataset, row, output_name):
    """Plot true map, predicted map, and absolute dex error map."""
    true_maps = dataset.unflatten(row["True_Target"])
    pred_maps = dataset.unflatten(row["NN_Target"])
    names = dataset.channel_names()

    n_channels = true_maps.shape[0]
    fig, axs = plt.subplots(n_channels, 3, figsize=(13.5, 4.2 * n_channels), squeeze=False)

    for ch in range(n_channels):
        true_img = true_maps[ch]
        pred_img = pred_maps[ch]
        error_img = np.abs(pred_img - true_img)

        vmin, vmax = np.nanpercentile(true_img, [2, 98])

        im0 = axs[ch, 0].imshow(true_img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        axs[ch, 0].set_title(f"True {names[ch]}")
        plt.colorbar(im0, ax=axs[ch, 0], fraction=0.046)

        im1 = axs[ch, 1].imshow(pred_img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        axs[ch, 1].set_title("POD-FFNN prediction")
        plt.colorbar(im1, ax=axs[ch, 1], fraction=0.046)

        im2 = axs[ch, 2].imshow(error_img, cmap="viridis", origin="lower")
        axs[ch, 2].set_title("Absolute error [dex]")
        plt.colorbar(im2, ax=axs[ch, 2], fraction=0.046)

    plt.suptitle(f"Step 4 validation at out-of-sample C={row['C_Value']:.4g}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


def plot_spectrum_mode_sweep(dataset, df, output_name):
    """
    Plot Part-2-style m-mode sweep for spectral targets only.

    This is appropriate for the spectral map because it already lives in Fourier/mode space.
    For RMS and flux real-space maps, this plot is skipped unless TARGET_TYPE='all', where
    the spectral channel is available as channel 2.
    """
    if dataset.target_type not in {"spectrum", "all"}:
        print("[Info] Skipping m-mode sweep because target is not spectral.")
        return

    spectrum_channel = 0 if dataset.target_type == "spectrum" else 2
    mode_max = min(65, dataset.grid_shape[1])
    modes = np.arange(1, mode_max)
    c_values = df["C_Value"].values

    true_profiles = []
    pred_profiles = []

    for _, row in df.iterrows():
        true_map = dataset.unflatten(row["True_Target"])[spectrum_channel]
        pred_map = dataset.unflatten(row["NN_Target"])[spectrum_channel]

        # Collapse 2D spectrum into 1D ky/m_y profile by averaging over kx.
        true_profiles.append(np.mean(true_map, axis=0)[1:mode_max])
        pred_profiles.append(np.mean(pred_map, axis=0)[1:mode_max])

    true_profiles = np.array(true_profiles)
    pred_profiles = np.array(pred_profiles)

    vmin, vmax = np.nanpercentile(true_profiles, [2, 98])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    im0 = axs[0].pcolormesh(modes, c_values, true_profiles, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    axs[0].set_yscale("log")
    axs[0].set_title(r"True mode sweep: $\log_{10} P_\phi(m_y)$")
    axs[0].set_xlabel("Mode number m")
    axs[0].set_ylabel("C")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(modes, c_values, pred_profiles, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
    axs[1].set_yscale("log")
    axs[1].set_title("Predicted mode sweep")
    axs[1].set_xlabel("Mode number m")
    axs[1].set_ylabel("C")
    plt.colorbar(im1, ax=axs[1])

    profile_err = np.median(np.abs(pred_profiles - true_profiles), axis=1)
    axs[2].plot(c_values, profile_err, marker="o", color="#1f77b4")
    axs[2].set_xscale("log")
    axs[2].set_title("Median mode-profile error")
    axs[2].set_xlabel("C")
    axs[2].set_ylabel("Median absolute error [dex]")
    axs[2].grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_name, dpi=200)
    print(f"[Success] Saved '{output_name}'.")


def run_loocv(dataset, device):
    results = []
    num_points = len(dataset)

    print(f"\nRunning Step 4 LOOCV for target='{dataset.target_type}' over {num_points} C values...")

    for test_idx in range(num_points):
        test_c = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]

        train_y = dataset.targets[train_indices]
        true_y = dataset.targets[test_idx]

        # PCA modes cannot exceed number of training samples or output dimension.
        n_modes = min(N_POD_MODES, len(train_indices), train_y.shape[1])

        pca = PCA(n_components=n_modes)
        train_coeffs = pca.fit_transform(train_y)

        coeff_mean = train_coeffs.mean(axis=0)
        coeff_std = train_coeffs.std(axis=0) + 1e-8
        train_coeffs_scaled = (train_coeffs - coeff_mean) / coeff_std

        train_x = torch.tensor(dataset.inputs[train_indices], dtype=torch.float32, device=device)
        train_target = torch.tensor(train_coeffs_scaled, dtype=torch.float32, device=device)
        test_x = torch.tensor(dataset.inputs[test_idx], dtype=torch.float32, device=device).unsqueeze(0)

        # Linear interpolation baseline directly in flattened log-map space.
        baseline = interp1d(
            dataset.inputs[train_indices].squeeze(),
            train_y,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        base_pred_y = baseline(dataset.inputs[test_idx].squeeze())

        model = PodCoefficientFFNN(num_modes=n_modes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(EPOCHS):
            pred = model(train_x)
            loss = criterion(pred, train_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scaled_pred_coeffs = model(test_x).cpu().numpy().squeeze()

        pred_coeffs = scaled_pred_coeffs * coeff_std + coeff_mean
        nn_pred_y = pca.inverse_transform(pred_coeffs.reshape(1, -1)).squeeze()

        nn_error = median_dex(nn_pred_y, true_y)
        base_error = median_dex(base_pred_y, true_y)

        results.append({
            "C_Value": test_c,
            "NN_DEX": nn_error,
            "Base_DEX": base_error,
            "True_Target": true_y,
            "NN_Target": nn_pred_y,
            "Base_Target": base_pred_y,
        })

        print(f" -> C={test_c:.3e} | NN={nn_error:.4f} dex | baseline={base_error:.4f} dex")

    return pd.DataFrame(results).sort_values("C_Value")


def print_report(df, target_type):
    print("\n" + "=" * 82)
    print(f"STEP 4 2D MAP REGRESSION REPORT — TARGET: {target_type}".center(82))
    print("=" * 82)
    print(f"{'C':<12} | {'NN median dex':<16} | {'Baseline median dex':<20}")
    print("-" * 82)
    for _, row in df.iterrows():
        print(f"{row['C_Value']:<12.4g} | {row['NN_DEX']:<16.4f} | {row['Base_DEX']:<20.4f}")
    print("=" * 82)


if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Step4MapDataset(DATA_DIR, target_type=TARGET_TYPE)
    df = run_loocv(dataset, device)
    print_report(df, TARGET_TYPE)

    # Representative 2D validation panel.
    sample_row = df.iloc[len(df) // 2]
    validation_output = f"step3_{TARGET_TYPE}_2d_map_validation.png"
    plot_validation_panel(dataset, sample_row, validation_output)

    # Part-2-style mode-space sweep only when a spectral channel is present.
    sweep_output = f"step3_{TARGET_TYPE}_mode_space_sweep.png"
    plot_spectrum_mode_sweep(dataset, df, sweep_output)
