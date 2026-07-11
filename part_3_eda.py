import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Step 4 EDA: 2D Statistical Maps vs C
# -----------------------------------------------------------------------------
# Extracts three time-averaged saturated 2D map families:
#   1. RMS potential fluctuation map: log10(sqrt(<phi^2>_t))
#   2. Local particle-flux magnitude map: log10(|<-kappa n d_y phi>_t|)
#   3. 2D potential spectral power map: log10(<|phi_hat(kx, ky)|^2>_t)
#
# This file is EDA-only: it checks smoothness and PCA/POD compressibility.
# Training is handled in part_3_train.py.
# -----------------------------------------------------------------------------

DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
EPS = 1e-20
SATURATION_START_FRACTION = 0.5


def infer_real_shape(uk_dataset):
    """
    Infer real-space grid shape from rfft2-style storage.

    Expected uk shape:
        [time, channel, nx, ny_rfft]

    For real FFT storage:
        ny_rfft = ny // 2 + 1
        ny = 2 * (ny_rfft - 1)
    """
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


def extract_step4_maps(h5_file, start_fraction=SATURATION_START_FRACTION):
    """
    Extract Step 4 time-averaged 2D targets from one HDF5 simulation file.

    Returns log-scaled maps:
        rms      = log10(sqrt(<phi^2>_t) + EPS)
        flux     = log10(|<-kappa n d_y phi>_t| + EPS)
        spectrum = log10(<|phi_hat(kx, ky)|^2>_t + EPS)

    Also returns signed flux before abs/log as flux_signed for physical inspection.
    """
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
        # Channel convention used in your earlier scripts:
        #   uk[:, 0, :, :] = potential phi_hat
        #   uk[:, 1, :, :] = density n_hat
        phi_k = uk[t, 0, :, :]

        if "fields/nk" in h5_file:
            n_k = h5_file["fields/nk"][t, 0, :, :]
        else:
            n_k = uk[t, 1, :, :]

        # Spectral derivative: d_y phi = F^-1[i ky phi_hat]
        grady_phi_k = 1j * ky2d * phi_k

        phi = np.fft.irfft2(phi_k, s=(nx, ny), norm="forward")
        density = np.fft.irfft2(n_k, s=(nx, ny), norm="forward")
        grady_phi = np.fft.irfft2(grady_phi_k, s=(nx, ny), norm="forward")

        # Spatial maps
        phi_sq_accum += phi**2
        flux_accum += -kappa * density * grady_phi

        # Full square 2D spectral power map from reconstructed real-space phi
        phi_full_k = np.fft.fft2(phi, norm="forward")
        spectrum_accum += np.abs(phi_full_k) ** 2

    rms_map = np.sqrt(phi_sq_accum / n_t)
    flux_signed_map = flux_accum / n_t
    spectrum_map = spectrum_accum / n_t

    return {
        "C": c_val,
        "shape": (nx, ny),
        "rms": np.log10(rms_map + EPS),
        "flux": np.log10(np.abs(flux_signed_map) + EPS),
        "flux_signed": flux_signed_map,
        "spectrum": np.log10(spectrum_map + EPS),
    }


class Saturated2DMapEDADataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".h5") and f.startswith("hwak_C")
        )

        self.raw_c_values = []
        self.inputs = []
        self.maps = {
            "rms": [],
            "flux": [],
            "spectrum": [],
        }
        self.flat_targets = []
        self.grid_shape = None

        self._process_files()

    def _process_files(self):
        print("Extracting Step 4 maps: RMS amplitude, local flux, and 2D spectral power...")

        for i, file_name in enumerate(self.file_list):
            file_path = os.path.join(self.data_dir, file_name)

            with h5py.File(file_path, "r") as f:
                if i == 0:
                    print("\nHDF5 diagnostic for first file:")
                    print("  file:", file_name)
                    print("  root groups:", list(f.keys()))
                    print("  fields:", list(f["fields"].keys()) if "fields" in f else "missing")
                    print("  uk shape:", f["fields/uk"].shape)

                extracted = extract_step4_maps(f)

            self.grid_shape = extracted["shape"]
            c_val = extracted["C"]

            self.raw_c_values.append(c_val)
            self.inputs.append([np.log10(c_val)])

            for target in self.maps:
                self.maps[target].append(extracted[target])

            # Combined vector for checking whether all Step 4 maps are POD-compressible
            combined_vector = np.concatenate([
                extracted["rms"].ravel(),
                extracted["flux"].ravel(),
                extracted["spectrum"].ravel(),
            ])
            self.flat_targets.append(combined_vector)

        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.flat_targets = np.array(self.flat_targets, dtype=np.float32)

        for target in self.maps:
            self.maps[target] = np.array(self.maps[target], dtype=np.float32)


def plot_projection_smoothness(dataset):
    """
    Plot simple 1D projections of the three Step 4 map families.

    RMS and flux are real-space maps, so their natural projection is over y:
        profile(x) = mean_y map(x, y)

    Spectrum is already in Fourier/mode space, so we show the ky=0 zonal cut.
    """
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(dataset.raw_c_values)))

    nx, ny = dataset.grid_shape
    x_axis = np.arange(nx)
    mode_axis = np.arange(1, min(65, ny))

    for idx, c_val in enumerate(dataset.raw_c_values):
        label = f"C={c_val:.3e}"

        rms_profile = np.mean(dataset.maps["rms"][idx], axis=1)
        flux_profile = np.mean(dataset.maps["flux"][idx], axis=1)

        # For spectral map, ky=0 column gives a zonal spectral cut over kx/mx.
        zonal_spectral_slice = dataset.maps["spectrum"][idx][1:len(mode_axis) + 1, 0]

        axs[0].plot(x_axis, rms_profile, color=colors[idx], lw=1.2, alpha=0.75)
        axs[1].plot(x_axis, flux_profile, color=colors[idx], lw=1.2, alpha=0.75)
        axs[2].plot(mode_axis, zonal_spectral_slice, color=colors[idx], lw=1.2, alpha=0.75, label=label)

    axs[0].set_title(r"Average fluctuation amplitude: $\log_{10}(RMS_\phi)$")
    axs[0].set_xlabel(r"Radial position index $x$")
    axs[0].set_ylabel(r"Amplitude $\log_{10}(RMS_\phi)$")

    axs[1].set_title(r"Average particle-flux magnitude: $\log_{10}|\Gamma_n|$")
    axs[1].set_xlabel(r"Radial position index $x$")
    axs[1].set_ylabel(r"Flux magnitude $\log_{10}|\Gamma_n|$")

    axs[2].set_title(r"Zonal spectral-power cut: $\log_{10}P_\phi(m_x,k_y=0)$")
    axs[2].set_xlabel(r"Radial mode number $m_x$")
    axs[2].set_ylabel(r"Spectral power $\log_{10}P_\phi$")
    axs[2].legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    for ax in axs:
        ax.grid(True, alpha=0.25, linestyle=":")

    plt.tight_layout()
    output = "step3_eda_2d_fields_smoothness.png"
    plt.savefig(output, dpi=200)
    print(f"[Success] Saved '{output}'.")


def plot_example_maps(dataset):
    """Save a representative visual check of the three extracted 2D maps."""
    idx = len(dataset.raw_c_values) // 2
    c_val = dataset.raw_c_values[idx]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    targets = ["rms", "flux", "spectrum"]
    titles = [
        r"Fluctuation amplitude: $\log_{10}(RMS_\phi)$",
        r"Particle-flux magnitude: $\log_{10}|\Gamma_n|$",
        r"2D spectral power: $\log_{10}P_\phi(k_x,k_y)$",
    ]

    for ax, target, title in zip(axs, targets, titles):
        img = dataset.maps[target][idx]
        vmin, vmax = np.nanpercentile(img, [2, 98])
        im = ax.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Representative 2D Statistical Maps at C={c_val:.4g}", fontweight="bold")
    plt.tight_layout()
    output = "step3_eda_example_2d_maps.png"
    plt.savefig(output, dpi=200)
    print(f"[Success] Saved '{output}'.")


if __name__ == "__main__":
    dataset = Saturated2DMapEDADataset(DATA_DIR)

    print("\n" + "=" * 70)
    print("PCA EXPLAINED VARIANCE ON COMBINED STEP 4 MAPS".center(70))
    print("=" * 70)

    max_modes = min(8, len(dataset.raw_c_values))
    pca = PCA(n_components=max_modes)
    pca.fit(dataset.flat_targets)

    cumulative = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        cumulative += 100 * ratio
        print(f"Mode {i:2d}: {100 * ratio:7.3f}% | cumulative: {cumulative:7.3f}%")

    print("=" * 70)

    plot_projection_smoothness(dataset)
    plot_example_maps(dataset)
