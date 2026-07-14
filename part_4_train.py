import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
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
BETA_KL = 1e-4
PREFERRED_HELDOUT_C_VALUES = [0.1, 0.5, 3.0]
N_HELDOUT_C_VALUES = 3

MODEL_OUT = "part4_cvae_model.pt"
LOSS_FIG = "part4_cvae_loss_curve.png"
RECON_FIG = "part4_reconstruction_examples.png"
GEN_FIG = "part4_generated_snapshots.png"


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
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2
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


# Purpose:
#   Build the Step 4 dataset of saturated real-space plasma snapshots.
# How it works:
#   For each C-run, it detects the saturated window, transforms density and
#   potential from Fourier space to real space, optionally downsamples, and
#   stores [density, potential] images conditioned on log10(C).
# Architecture role:
#   Provides samples from p([n, phi] | C) for the conditional VAE.
class SaturatedSnapshotDataset(Dataset):
    """Dataset of saturated real-space snapshots [density, potential] conditioned on log10(C)."""

    def __init__(self, data_dir, downsample_to=DOWNSAMPLE_TO, max_snapshots_per_c=MAX_SNAPSHOTS_PER_C):
        self.data_dir = data_dir
        self.downsample_to = downsample_to
        self.max_snapshots_per_c = max_snapshots_per_c
        self.file_list = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".h5") and f.startswith("hwak_C")
        )

        self.conditions = []  # log10(C)
        self.images = []      # [2, H, W], channels = [density, potential]
        self.raw_c_values = []
        self.image_shape = None
        self.channel_mean = None
        self.channel_std = None

        self._process_files()

    def _process_files(self):
        print("Extracting saturated [n, phi] snapshots for conditional VAE...")

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

                    phi = np.fft.irfft2(phi_k, s=(nx, ny), norm="forward")
                    density = np.fft.irfft2(n_k, s=(nx, ny), norm="forward")

                    image = np.stack([density, phi], axis=0)
                    image = downsample_image_np(image, self.downsample_to)

                    self.conditions.append([log_c])
                    self.images.append(image)
                    self.raw_c_values.append(c_val)

        self.conditions = np.array(self.conditions, dtype=np.float32)
        self.images = np.array(self.images, dtype=np.float32)
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.image_shape = self.images.shape[-2:]
        print(f"Loaded {len(self.images)} snapshots with image shape {self.images.shape[1:]}")

    def recompute_normalization_from_indices(self, indices):
        """Normalize all images using training indices only to avoid validation leakage."""
        train_images = self.images[np.array(indices, dtype=int)]
        self.channel_mean = train_images.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self.channel_std = (train_images.std(axis=(0, 2, 3), keepdims=True) + 1e-8).astype(np.float32)
        self.images = (self.images - self.channel_mean) / self.channel_std
        print("Channel normalization fitted on training C values only:")
        print(f"  density mean/std = {self.channel_mean.ravel()[0]:.4e} / {self.channel_std.ravel()[0]:.4e}")
        print(f"  phi     mean/std = {self.channel_mean.ravel()[1]:.4e} / {self.channel_std.ravel()[1]:.4e}")

    def denormalize(self, x):
        """Denormalize torch or numpy tensor with shape [..., 2, H, W]. Returns numpy."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        mean = self.channel_mean.reshape(1, 2, 1, 1)
        std = self.channel_std.reshape(1, 2, 1, 1)
        return x * std + mean

    def unique_c_values(self):
        return np.unique(self.raw_c_values)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.conditions[idx])


# Purpose:
#   Conditional variational autoencoder for saturated plasma snapshots.
# How it works:
#   The encoder receives [density, potential, log10(C)-channel] and maps each
#   image to a Gaussian latent distribution. The decoder receives a random
#   latent vector z plus log10(C), then generates a plausible [density, phi]
#   snapshot for that condition.
class ConditionalVAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=LATENT_DIM):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for this baseline architecture.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.reduced_size = image_size // 16
        self.enc_flat_dim = 256 * self.reduced_size * self.reduced_size

        # Encoder receives [density, phi, C_channel]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
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
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),
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


# Purpose:
#   Visualize VAE training progress.
# How it works:
#   Plots total, reconstruction, and KL losses across epochs on a log scale.
def plot_loss_curve(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="total loss")
    plt.plot(history["recon"], label="reconstruction loss")
    plt.plot(history["kl"], label="KL loss")
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
#   Show reconstruction examples from the validation/held-out set.
# How it works:
#   Passes selected true snapshots through the encoder-decoder and plots true
#   vs reconstructed density/potential fields.
def plot_reconstructions(model, dataset, device, indices=None, n_examples=3):
    model.eval()
    if indices is None:
        n_examples = min(n_examples, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)
    else:
        indices = np.array(indices, dtype=int)
        n_examples = min(n_examples, len(indices))
        indices = indices[np.linspace(0, len(indices) - 1, n_examples, dtype=int)]
    fig, axs = plt.subplots(n_examples, 4, figsize=(12, 3.2 * n_examples))
    if n_examples == 1:
        axs = axs[None, :]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            x, c = dataset[idx]
            x_b = x.unsqueeze(0).to(device)
            c_b = c.unsqueeze(0).to(device)
            recon, _, _ = model(x_b, c_b)
            true_np = dataset.denormalize(x_b)[0]
            recon_np = dataset.denormalize(recon)[0]
            c_val = 10 ** float(c.item())

            panels = [true_np[0], recon_np[0], true_np[1], recon_np[1]]
            titles = ["True density n", "Reconstructed n", "True potential phi", "Reconstructed phi"]
            for col, (img, title) in enumerate(zip(panels, titles)):
                im = axs[row, col].imshow(img, origin="lower", cmap="RdBu_r")
                axs[row, col].set_title(f"{title}\nC={c_val:.3g}")
                plt.colorbar(im, ax=axs[row, col], fraction=0.046)
    plt.suptitle("Conditional VAE Reconstruction Examples", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RECON_FIG, dpi=200)
    print(f"[Success] Saved '{RECON_FIG}'.")


# Purpose:
#   Generate new plasma snapshots at selected C values.
# How it works:
#   Draws random latent vectors z and decodes them together with log10(C), so
#   multiple statistically plausible samples can be generated for the same C.
def plot_generated_samples(model, dataset, device, c_values=None, samples_per_c=2):
    model.eval()
    if c_values is None:
        c_values = dataset.unique_c_values()
    c_values = np.array(c_values, dtype=np.float32)
    if len(c_values) > 3:
        c_values = np.array([c_values[0], c_values[len(c_values)//2], c_values[-1]])

    fig, axs = plt.subplots(len(c_values), samples_per_c * 2, figsize=(4 * samples_per_c * 2, 3.5 * len(c_values)))
    if len(c_values) == 1:
        axs = axs[None, :]

    with torch.no_grad():
        for row, c_val in enumerate(c_values):
            c_log = torch.full((samples_per_c, 1), np.log10(c_val), dtype=torch.float32, device=device)
            z = torch.randn(samples_per_c, LATENT_DIM, device=device)
            gen = model.decode(z, c_log)
            gen_np = dataset.denormalize(gen)
            for j in range(samples_per_c):
                im0 = axs[row, 2*j].imshow(gen_np[j, 0], origin="lower", cmap="RdBu_r")
                axs[row, 2*j].set_title(f"Generated n\nC={c_val:.3g}")
                plt.colorbar(im0, ax=axs[row, 2*j], fraction=0.046)
                im1 = axs[row, 2*j+1].imshow(gen_np[j, 1], origin="lower", cmap="RdBu_r")
                axs[row, 2*j+1].set_title(f"Generated phi\nC={c_val:.3g}")
                plt.colorbar(im1, ax=axs[row, 2*j+1], fraction=0.046)
    plt.suptitle("Conditional VAE Generated Plasma Snapshots", fontweight="bold")
    plt.tight_layout()
    plt.savefig(GEN_FIG, dpi=200)
    print(f"[Success] Saved '{GEN_FIG}'.")


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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConditionalVAE(image_size=image_size, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"loss": [], "recon": [], "kl": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        n_batches = 0
        for x, c in train_loader:
            x, c = x.to(device), c.to(device)
            recon, mu, logvar = model(x, c)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), recon_loss.item(), kl_loss.item()])
            n_batches += 1
        means = totals / max(n_batches, 1)
        history["loss"].append(means[0])
        history["recon"].append(means[1])
        history["kl"].append(means[2])

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:04d}/{EPOCHS} | loss={means[0]:.4e} | recon={means[1]:.4e} | KL={means[2]:.4e}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "channel_mean": dataset.channel_mean,
        "channel_std": dataset.channel_std,
        "latent_dim": LATENT_DIM,
        "image_shape": dataset.image_shape,
        "downsample_to": DOWNSAMPLE_TO,
        "preferred_heldout_c_values": PREFERRED_HELDOUT_C_VALUES,
        "resolved_heldout_c_values": resolved_heldout_c_values.tolist(),
    }, MODEL_OUT)
    print(f"[Success] Saved '{MODEL_OUT}'.")

    plot_loss_curve(history)
    plot_reconstructions(model, dataset, device, indices=val_indices)
    plot_generated_samples(model, dataset, device, c_values=resolved_heldout_c_values)


if __name__ == "__main__":
    train()