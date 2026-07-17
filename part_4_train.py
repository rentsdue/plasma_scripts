import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from plasma_saturation import detect_saturation_window

# -----------------------------------------------------------------------------
# Step 5 / Part 4: Conditional VAE Snapshot Generation
# -----------------------------------------------------------------------------
# Goal:
#   Learn p([n(x,y), phi(x,y)] | C), i.e. generate statistically plausible
#   saturated 2D density and potential snapshots conditioned on adiabaticity C.
#
# This is intentionally generative, not direct regression:
#   log10(C) + random latent z -> generated [n, phi] snapshot
#
# Extraction follows the syntax used in earlier parts:
#   uk[t, 0, :, :] = phi_hat
#   uk[t, 1, :, :] = n_hat
#   real fields are reconstructed with np.fft.irfft2(..., norm="forward")
# -----------------------------------------------------------------------------

DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"

SEED = 42
DOWNSAMPLE_TO = 128          # Keep 128 for first baseline. Use None only after architecture changes.
MAX_SNAPSHOTS_PER_C = None   # e.g. 64 for quicker tests; None uses all saturated saved snapshots.

BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
LATENT_DIM = 64
BETA_KL = 1e-3
KL_WARMUP_EPOCHS = 20
N_FIELDS = 3  # [density n, potential phi, vorticity omega]
PREFERRED_HELDOUT_C_VALUES = [0.1, 0.5, 3.0]
N_HELDOUT_C_VALUES = 3

MODEL_OUT = "step4_cvae_model.pt"
LOSS_FIG = "step4_cvae_loss_curve.png"
RECON_FIG = "step4_reconstruction_examples.png"
GEN_FIG = "step4_generated_snapshots.png"

FIELD_NAMES = ["n", "phi", "omega"]
FIELD_LABELS = [r"$n$", r"$\phi$", r"$\omega$"]


# Purpose:
#   Make the Step 4 generative experiment reproducible.
# How it works:
#   Seeds Python, NumPy, and PyTorch random number generators, and asks cuDNN to
#   use deterministic kernels where possible.
def seed_everything(seed=SEED):
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
#   Infer the real-space grid size from TOKAM2D's real-FFT Fourier layout.
# How it works:
#   The x dimension is stored directly, while the y dimension is reconstructed
#   from the Hermitian half-spectrum length used by irfft2.
def infer_real_shape(uk_dataset):
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


# Purpose:
#   Compute a smooth saturation diagnostic for every saved snapshot.
# How it works:
#   Uses total kinetic energy K(t)=1/2 sum_k k^2 |phi_k|^2 from the Fourier
#   potential field. This is smoother than flux and is used to choose the
#   saturated training window.
def compute_total_kinetic_energy_series(uk, kx2d, ky2d):
    series = []
    ky_weights = np.ones(uk.shape[-1], dtype=np.float64)
    if uk.shape[-1] > 2:
        ky_weights[1:-1] = 2.0
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2 * ky_weights[None, :]
        series.append(np.sum(e_mode))
    return np.array(series, dtype=np.float64)


# Purpose:
#   Reduce snapshot resolution before VAE training.
# How it works:
#   Converts a [channels, nx, ny] NumPy image to a temporary PyTorch tensor and
#   applies bilinear interpolation to produce a smaller square image.
def downsample_image_np(image, size):
    """Downsample [channels, nx, ny] numpy image to [channels, size, size]."""
    if size is None:
        return image.astype(np.float32)
    with torch.no_grad():
        x = torch.tensor(image[None, ...], dtype=torch.float32)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0).numpy().astype(np.float32)


def truncate_rfft2_array(arr, out_nx, out_ny):
    """Low-pass truncate a real-FFT-layout array to an [out_nx, out_ny//2+1] spectrum."""
    out_nyh = out_ny // 2 + 1
    out = np.zeros((out_nx, out_nyh), dtype=arr.dtype)
    pos_rows = out_nx // 2 + 1
    neg_rows = out_nx - pos_rows
    cols = min(out_nyh, arr.shape[1])
    out[:pos_rows, :cols] = arr[:pos_rows, :cols]
    if neg_rows > 0:
        out[-neg_rows:, :cols] = arr[-neg_rows:, :cols]
    return out


def reconstruct_fields_from_fourier(phi_k, n_k, kx2d, ky2d, real_shape):
    """Reconstruct [n, phi, omega] from Fourier data at the requested real-space shape."""
    nx, ny = real_shape
    omega_k = -(kx2d**2 + ky2d**2) * phi_k
    phi = np.fft.irfft2(phi_k, s=(nx, ny), norm="forward")
    phi = phi - np.mean(phi)  # Remove gauge-dependent spatially constant potential offset.
    density = np.fft.irfft2(n_k, s=(nx, ny), norm="forward")
    omega = np.fft.irfft2(omega_k, s=(nx, ny), norm="forward")
    return np.stack([density, phi, omega], axis=0).astype(np.float32)


def reconstruct_model_resolution_fields(phi_k, n_k, kx2d, ky2d, native_shape, target_size):
    """Use Fourier truncation, not real-space interpolation, to set model resolution."""
    if target_size is None:
        return reconstruct_fields_from_fourier(phi_k, n_k, kx2d, ky2d, native_shape)

    phi_k_trunc = truncate_rfft2_array(phi_k, target_size, target_size)
    n_k_trunc = truncate_rfft2_array(n_k, target_size, target_size)
    kx_trunc = truncate_rfft2_array(kx2d, target_size, target_size)
    ky_trunc = truncate_rfft2_array(ky2d, target_size, target_size)
    return reconstruct_fields_from_fourier(
        phi_k_trunc,
        n_k_trunc,
        kx_trunc,
        ky_trunc,
        (target_size, target_size),
    )


# Purpose:
#   Build a Gaussian baseline with the same per-channel Fourier amplitudes as a
#   real snapshot.
# How it works:
#   Randomizes phases using the FFT of real white noise, preserving Hermitian
#   symmetry, then transforms back to real space and restores the channel mean.
def spectrum_matched_gaussian_baseline(image, rng):
    baseline = np.zeros_like(image, dtype=np.float32)
    for ch in range(image.shape[0]):
        field = image[ch].astype(np.float64)
        field_mean = np.mean(field)
        centered = field - field_mean
        target_amp = np.abs(np.fft.fft2(centered, norm="forward"))
        noise = rng.normal(size=centered.shape)
        noise_phase = np.exp(1j * np.angle(np.fft.fft2(noise, norm="forward")))
        gaussian = np.fft.ifft2(target_amp * noise_phase, norm="forward").real + field_mean
        baseline[ch] = gaussian.astype(np.float32)
    return baseline


# Purpose:
#   Build the Step 4 residual-correction dataset.
# How it works:
#   For each C-run, it detects the saturated window, transforms density and
#   potential from Fourier space to real space, optionally downsamples, and
#   stores actual fields, spectrum-matched Gaussian baselines, and residuals
#   conditioned on log10(C).
# Architecture role:
#   Provides samples of Δx = x_actual - x_Gaussian for the conditional VAE.
class SaturatedSnapshotDataset(Dataset):
    """Dataset of residuals beyond a spectrum-matched Gaussian baseline."""

    def __init__(self, data_dir, downsample_to=DOWNSAMPLE_TO, max_snapshots_per_c=MAX_SNAPSHOTS_PER_C):
        self.data_dir = data_dir
        self.downsample_to = downsample_to
        self.max_snapshots_per_c = max_snapshots_per_c
        self.file_list = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".h5") and f.startswith("hwak_C")
        )

        self.conditions = []      # log10(C)
        self.actual_images = []   # x_actual: [3, H, W]
        self.baselines = []       # x_Gaussian: [3, H, W]
        self.residuals = []       # Δx = x_actual - x_Gaussian
        self.residual_scales = [] # per-sample/channel Gaussian RMS used to scale Δx
        self.raw_c_values = []
        self.image_shape = None
        self.channel_mean = None
        self.channel_std = None

        self._process_files()

    def _process_files(self):
        print("Extracting saturated [n, φ, ω] residuals relative to Gaussian baselines...")
        rng = np.random.default_rng(SEED)

        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, "r") as f:
                c_val = f["params/C"][()]
                log_c = np.log10(c_val)
                uk = f["fields/uk"]
                kx2d = f["data/kx"][()]
                ky2d = f["data/ky"][()]
                nx, ny = infer_real_shape(uk)

                T = uk.shape[0]
                ke_series = compute_total_kinetic_energy_series(uk, kx2d, ky2d)
                window, sat_info = detect_saturation_window(ke_series)
                t_start = sat_info["t_start"]
                block_means = sat_info["block_means"]
                all_indices = list(window)
                if self.max_snapshots_per_c is not None and len(all_indices) > self.max_snapshots_per_c:
                    rng = np.random.default_rng(SEED)
                    all_indices = sorted(rng.choice(all_indices, size=self.max_snapshots_per_c, replace=False).tolist())

                for t in all_indices:
                    phi_k = uk[t, 0, :, :]
                    n_k = f["fields/nk"][t, 0, :, :] if "fields/nk" in f else uk[t, 1, :, :]
                    image = reconstruct_model_resolution_fields(
                        phi_k,
                        n_k,
                        kx2d,
                        ky2d,
                        native_shape=(nx, ny),
                        target_size=self.downsample_to,
                    )
                    baseline = spectrum_matched_gaussian_baseline(image, rng)
                    residual = image - baseline
                    residual_scale = np.sqrt(np.mean(baseline**2, axis=(1, 2), keepdims=True)).astype(np.float32) + 1e-8
                    scaled_residual = residual / residual_scale

                    self.conditions.append([log_c])
                    self.actual_images.append(image)
                    self.baselines.append(baseline)
                    self.residuals.append(scaled_residual)
                    self.residual_scales.append(residual_scale)
                    self.raw_c_values.append(c_val)

        self.conditions = np.array(self.conditions, dtype=np.float32)
        self.actual_images = np.array(self.actual_images, dtype=np.float32)
        self.baselines = np.array(self.baselines, dtype=np.float32)
        self.residuals = np.array(self.residuals, dtype=np.float32)
        self.residual_scales = np.array(self.residual_scales, dtype=np.float32)
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.image_shape = self.actual_images.shape[-2:]
        print(f"Loaded {len(self.actual_images)} residual samples with image shape {self.actual_images.shape[1:]}")

    def recompute_normalization_from_indices(self, indices):
        """Residuals are sample-wise scaled by Gaussian RMS; no global C-mixing normalization."""
        self.channel_mean = np.zeros((1, N_FIELDS, 1, 1), dtype=np.float32)
        self.channel_std = np.ones((1, N_FIELDS, 1, 1), dtype=np.float32)
        print("Residuals scaled sample-wise by Gaussian baseline RMS; no global per-channel normalization applied.")

    def denormalize_residual(self, x):
        """Return dimensionless residual tensor with shape [..., N_FIELDS, H, W]."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    def denormalize(self, x):
        """Backward-compatible alias for residual denormalization."""
        return self.denormalize_residual(x)

    def unique_c_values(self):
        return np.unique(self.raw_c_values)

    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        return torch.tensor(self.residuals[idx]), torch.tensor(self.conditions[idx])


# Purpose:
#   Conditional variational autoencoder for non-Gaussian residuals.
# How it works:
#   The encoder receives residual fields plus a log10(C)-channel. The decoder
#   receives a random latent vector z plus log10(C), then generates a residual
#   correction to add constructively to a spectrum-matched Gaussian baseline.
class ConditionalVAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=LATENT_DIM):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for this baseline architecture.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.reduced_size = image_size // 16
        self.enc_flat_dim = 256 * self.reduced_size * self.reduced_size

        # Encoder receives [density, phi, omega, C_channel]
        self.encoder = nn.Sequential(
            nn.Conv2d(N_FIELDS + 1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)

        # Decoder receives [z, log10(C)]
        self.fc_decode = nn.Linear(latent_dim + 1, self.enc_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, N_FIELDS, 4, stride=2, padding=1),
        )

    def encode(self, x, c):
        b, _, h, w = x.shape
        c_channel = c.view(b, 1, 1, 1).expand(b, 1, h, w)
        x_cond = torch.cat([x, c_channel], dim=1)
        features = self.encoder(x_cond)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        x = self.fc_decode(zc)
        x = x.view(-1, 256, self.reduced_size, self.reduced_size)
        return self.decoder(x)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


# Purpose:
#   Combine reconstruction quality with latent-space regularization.
# How it works:
#   Uses MSE for image reconstruction plus a beta-weighted KL divergence that
#   keeps the latent distribution close to a standard normal prior.
def vae_loss(recon, x, mu, logvar, beta=BETA_KL):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def robust_symmetric_limits(*arrays, percentile=99.0):
    values = np.concatenate([np.ravel(np.asarray(arr)) for arr in arrays])
    limit = np.nanpercentile(np.abs(values), percentile)
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def imshow_xy(ax, img, **kwargs):
    """Display stored (x, y) fields with x horizontal and y vertical."""
    return ax.imshow(np.asarray(img).T, origin="lower", **kwargs)


def make_balanced_c_sampler(dataset, indices):
    """Return a sampler that gives each C value equal expected training weight."""
    subset_c = dataset.raw_c_values[np.array(indices, dtype=int)]
    unique_c, counts = np.unique(subset_c, return_counts=True)
    count_by_c = {float(c): int(count) for c, count in zip(unique_c, counts)}
    weights = np.array([1.0 / count_by_c[float(c)] for c in subset_c], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(indices),
        replacement=True,
    )


# Purpose:
#   Visualize VAE training progress.
# How it works:
#   Plots total, reconstruction, and KL losses across epochs on a log scale.
def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train total loss")
    plt.plot(history["train_recon"], label="train residual reconstruction")
    plt.plot(history["val_recon"], label="held-out C residual reconstruction")
    plt.plot(history["weighted_kl"], label=r"weighted KL, $\beta D_{KL}$")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional VAE Training Loss")
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_FIG, dpi=200)
    print(f"[Success] Saved '{LOSS_FIG}'.")


# Purpose:
#   Show residual-correction examples from the validation/held-out set.
# How it works:
#   Passes selected residuals through the encoder-decoder, adds the predicted
#   residual to the Gaussian baseline, and compares against the actual field.
def plot_reconstructions(model, dataset, device, indices=None, n_examples=3):
    model.eval()
    if indices is None:
        n_examples = min(n_examples, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)
    else:
        indices = np.array(indices, dtype=int)
        n_examples = min(n_examples, len(indices))
        indices = indices[np.linspace(0, len(indices) - 1, n_examples, dtype=int)]
    n_panels = 5
    fig, axs = plt.subplots(N_FIELDS, n_panels, figsize=(4.0 * n_panels, 9.0))
    if N_FIELDS == 1:
        axs = axs[None, :]

    with torch.no_grad():
        idx = indices[len(indices) // 2]
        x, c = dataset[idx]
        x_b = x.unsqueeze(0).to(device)
        c_b = c.unsqueeze(0).to(device)
        recon, _, _ = model(x_b, c_b)
        true_residual_np = dataset.denormalize_residual(x_b)[0]
        pred_residual_np = dataset.denormalize_residual(recon)[0]
        baseline_np = dataset.baselines[idx]
        residual_scale_np = dataset.residual_scales[idx]
        actual_np = dataset.actual_images[idx]
        true_residual_physical_np = true_residual_np * residual_scale_np
        pred_residual_physical_np = pred_residual_np * residual_scale_np
        corrected_np = baseline_np + pred_residual_physical_np
        err_np = corrected_np - actual_np
        c_val = 10 ** float(c.item())

        for row, label in enumerate(FIELD_LABELS):
            vmin, vmax = robust_symmetric_limits(baseline_np[row], corrected_np[row], actual_np[row])
            evmin, evmax = robust_symmetric_limits(err_np[row])
            rvmin, rvmax = robust_symmetric_limits(true_residual_physical_np[row], pred_residual_physical_np[row])
            panels = [baseline_np[row], pred_residual_physical_np[row], corrected_np[row], actual_np[row], err_np[row]]
            titles = [fr"Gaussian {label}", fr"VAE correction {label}", fr"Corrected {label}", fr"Actual {label}", fr"Error {label}"]
            limits = [(vmin, vmax), (rvmin, rvmax), (vmin, vmax), (vmin, vmax), (evmin, evmax)]
            if axs.shape[1] != len(panels):
                raise RuntimeError("plot_reconstructions axes do not match requested panel count.")
            for col, (img, title, (lo, hi)) in enumerate(zip(panels, titles, limits)):
                im = imshow_xy(axs[row, col], img, cmap="RdBu_r", vmin=lo, vmax=hi)
                axs[row, col].set_title(title)
                axs[row, col].set_xlabel("x index")
                axs[row, col].set_ylabel("y index")
                plt.colorbar(im, ax=axs[row, col], fraction=0.046)
    plt.suptitle(fr"Residual VAE correction for held-out $C={c_val:.3g}$", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RECON_FIG, dpi=200)
    print(f"[Success] Saved '{RECON_FIG}'.")


# Purpose:
#   Generate new plasma snapshots at selected C values.
# How it works:
#   Draws random latent vectors z and decodes them together with log10(C), so
#   multiple statistically plausible samples can be generated for the same C.
def plot_generated_samples(model, dataset, device, c_values=None, actual_indices=None, samples_per_c=1):
    model.eval()
    if c_values is None:
        c_values = dataset.unique_c_values()
    c_values = np.array(c_values, dtype=np.float32)
    if len(c_values) > 3:
        c_values = np.array([c_values[0], c_values[len(c_values)//2], c_values[-1]])

    samples_per_c = 1
    if actual_indices is None:
        actual_indices = np.arange(len(dataset), dtype=int)
    else:
        actual_indices = np.array(actual_indices, dtype=int)

    with torch.no_grad():
        saved_paths = []
        for c_val in c_values:
            matching_indices = actual_indices[np.isclose(dataset.raw_c_values[actual_indices], c_val, rtol=0.0, atol=1e-6)]
            if len(matching_indices) == 0:
                nearest_local_idx = np.argmin(np.abs(np.log10(dataset.raw_c_values[actual_indices]) - np.log10(c_val)))
                actual_idx = int(actual_indices[nearest_local_idx])
            else:
                actual_idx = int(matching_indices[len(matching_indices) // 2])

            c_log = torch.full((samples_per_c, 1), np.log10(c_val), dtype=torch.float32, device=device)
            z = torch.randn(samples_per_c, LATENT_DIM, device=device)
            gen = model.decode(z, c_log)
            pred_residual_np = dataset.denormalize_residual(gen)[0]

            baseline_np = dataset.baselines[actual_idx]
            residual_scale_np = dataset.residual_scales[actual_idx]
            actual_np = dataset.actual_images[actual_idx]
            pred_residual_physical_np = pred_residual_np * residual_scale_np
            corrected_np = baseline_np + pred_residual_physical_np

            fig, axs = plt.subplots(4, N_FIELDS, figsize=(4 * N_FIELDS, 12.0))

            for ch, label in enumerate(FIELD_LABELS):
                vmin, vmax = robust_symmetric_limits(baseline_np[ch], corrected_np[ch], actual_np[ch])
                rvmin, rvmax = robust_symmetric_limits(pred_residual_physical_np[ch])
                panels = [
                    (0, baseline_np[ch], fr"Gaussian {label}: $C={c_val:.3g}$", vmin, vmax),
                    (1, pred_residual_physical_np[ch], fr"VAE correction {label}: $C={c_val:.3g}$", rvmin, rvmax),
                    (2, corrected_np[ch], fr"Corrected {label}: $C={c_val:.3g}$", vmin, vmax),
                    (3, actual_np[ch], fr"Actual {label}: $C={c_val:.3g}$", vmin, vmax),
                ]
                for row, img, title, lo, hi in panels:
                    im = imshow_xy(axs[row, ch], img, cmap="RdBu_r", vmin=lo, vmax=hi)
                    axs[row, ch].set_title(title)
                    axs[row, ch].set_xlabel("x index")
                    axs[row, ch].set_ylabel("y index")
                    plt.colorbar(im, ax=axs[row, ch], fraction=0.046)

            fig.suptitle(fr"Gaussian baseline + VAE residual correction at $C={c_val:.3g}$", fontweight="bold")
            fig.tight_layout()
            c_tag = f"{float(c_val):.4g}".replace(".", "p").replace("-", "m")
            output_path = GEN_FIG.replace(".png", f"_C_{c_tag}.png")
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            saved_paths.append(output_path)

    print(f"[Success] Saved generated-vs-actual holdout comparisons: {', '.join(saved_paths)}")


# Purpose:
#   Resolve preferred held-out C targets to values that actually exist.
# How it works:
#   Finds the nearest available simulation C for each preferred target, removes
#   duplicates, and fills any missing slots with spread-out available C values.
#   This keeps the low/mid/high validation idea without assuming exact C values
#   are present in the current scan.
def resolve_heldout_c_values(available_c_values, preferred_c_values, n_heldout=3):
    available = np.sort(np.asarray(available_c_values, dtype=np.float32))
    preferred = np.asarray(preferred_c_values, dtype=np.float32)
    if available.size < 2:
        raise ValueError("Need at least two distinct C values to create a train/validation C split.")

    n_heldout = int(min(max(n_heldout, 1), available.size - 1))
    selected = []

    for target in preferred:
        nearest = available[np.argmin(np.abs(np.log10(available) - np.log10(target)))]
        if not any(np.isclose(nearest, c, rtol=0.0, atol=1e-6) for c in selected):
            selected.append(float(nearest))
        if len(selected) == n_heldout:
            break

    if len(selected) < n_heldout:
        fill_positions = np.linspace(0, available.size - 1, n_heldout, dtype=int)
        for idx in fill_positions:
            candidate = float(available[idx])
            if not any(np.isclose(candidate, c, rtol=0.0, atol=1e-6) for c in selected):
                selected.append(candidate)
            if len(selected) == n_heldout:
                break

    selected = np.array(sorted(selected), dtype=np.float32)
    print("\nHeld-out C target resolution:")
    print(f"  Preferred held-out C values: {np.array2string(preferred, precision=4)}")
    print(f"  Available C values         : {np.array2string(available, precision=4)}")
    print(f"  Resolved held-out C values : {np.array2string(selected, precision=4)}")
    return selected


# Purpose:
#   Create a grouped held-out-C split for Step 4 validation.
# How it works:
#   All snapshots whose raw C value is in the resolved held-out C values are assigned to
#   validation; every other C value is used for training. This tests whether the
#   conditional VAE generalizes to unseen physical parameters.
def make_heldout_c_split(dataset, heldout_c_values, atol=1e-6):
    heldout_c_values = np.array(heldout_c_values, dtype=np.float32)
    is_heldout = np.zeros(len(dataset), dtype=bool)
    for c_val in heldout_c_values:
        is_heldout |= np.isclose(dataset.raw_c_values, c_val, rtol=0.0, atol=atol)

    train_indices = np.where(~is_heldout)[0].tolist()
    val_indices = np.where(is_heldout)[0].tolist()
    if not train_indices or not val_indices:
        raise ValueError(
            "Held-out C split failed. Check that HELDOUT_C_VALUES exist in the dataset "
            "and that at least one C remains for training."
        )
    return train_indices, val_indices


# Purpose:
#   Print the grouped split so the validation design is transparent.
# How it works:
#   Reports training/held-out C values and the number of snapshots in each set.
def print_split_summary(dataset, train_indices, val_indices):
    train_c = np.unique(dataset.raw_c_values[train_indices])
    val_c = np.unique(dataset.raw_c_values[val_indices])
    print("\nHeld-out-C validation split:")
    print(f"  Training C values : {np.array2string(train_c, precision=4)}")
    print(f"  Held-out C values : {np.array2string(val_c, precision=4)}")
    print(f"  Training snapshots: {len(train_indices)}")
    print(f"  Validation snapshots: {len(val_indices)}\n")


# Purpose:
#   Run the full conditional VAE training workflow.
# How it works:
#   Loads saturated snapshots, holds out full C values for validation, trains the
#   VAE on the remaining C values, saves the model, and produces loss,
#   reconstruction, and generation figures.
def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SaturatedSnapshotDataset(DATA_DIR)
    image_size = dataset.image_shape[0]
    resolved_heldout_c_values = resolve_heldout_c_values(
        dataset.unique_c_values(),
        PREFERRED_HELDOUT_C_VALUES,
        n_heldout=N_HELDOUT_C_VALUES,
    )
    train_indices, val_indices = make_heldout_c_split(dataset, resolved_heldout_c_values)
    dataset.recompute_normalization_from_indices(train_indices)
    print_split_summary(dataset, train_indices, val_indices)
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    train_sampler = make_balanced_c_sampler(dataset, train_indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConditionalVAE(image_size=image_size, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss": [], "train_recon": [], "val_recon": [], "weighted_kl": []}

    for epoch in range(1, EPOCHS + 1):
        beta = BETA_KL * min(1.0, epoch / KL_WARMUP_EPOCHS)
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        n_batches = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            recon, mu, logvar = model(x, c)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), recon_loss.item(), kl_loss.item()])
            n_batches += 1
        means = totals / max(n_batches, 1)
        model.eval()
        val_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for x, c in val_loader:
                x, c = x.to(device), c.to(device)
                recon, _, _ = model(x, c)
                val_total += F.mse_loss(recon, x, reduction="mean").item()
                val_batches += 1
        val_recon = val_total / max(val_batches, 1)

        history["train_loss"].append(means[0])
        history["train_recon"].append(means[1])
        history["val_recon"].append(val_recon)
        history["weighted_kl"].append(beta * means[2])

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:04d}/{EPOCHS} | beta={beta:.2e} | "
                f"loss={means[0]:.4e} | train recon={means[1]:.4e} | "
                f"val recon={val_recon:.4e} | weighted KL={beta * means[2]:.4e}"
            )

    torch.save({
        "model_state_dict": model.state_dict(),
        "channel_mean": dataset.channel_mean,
        "channel_std": dataset.channel_std,
        "latent_dim": LATENT_DIM,
        "n_fields": N_FIELDS,
        "field_names": FIELD_NAMES,
        "image_shape": dataset.image_shape,
        "downsample_to": DOWNSAMPLE_TO,
        "preferred_heldout_c_values": PREFERRED_HELDOUT_C_VALUES,
        "resolved_heldout_c_values": resolved_heldout_c_values.tolist(),
    }, MODEL_OUT)
    print(f"[Success] Saved '{MODEL_OUT}'.")

    plot_loss_curve(history)
    plot_reconstructions(model, dataset, device, indices=val_indices)
    plot_generated_samples(model, dataset, device, c_values=resolved_heldout_c_values, actual_indices=val_indices)


if __name__ == "__main__":
    train()