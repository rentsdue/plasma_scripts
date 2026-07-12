import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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
TRAIN_FRACTION = 0.9

MODEL_OUT = "part4_cvae_model.pt"
LOSS_FIG = "part4_cvae_loss_curve.png"
RECON_FIG = "part4_reconstruction_examples.png"
GEN_FIG = "part4_generated_snapshots.png"


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_real_shape(uk_dataset):
    nx = uk_dataset.shape[2]
    ny = 2 * (uk_dataset.shape[3] - 1)
    return nx, ny


def compute_total_kinetic_energy_series(uk, kx2d, ky2d):
    series = []
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2
        series.append(np.sum(e_mode))
    return np.array(series, dtype=np.float64)


def downsample_image_np(image, size):
    """Downsample [channels, nx, ny] numpy image to [channels, size, size]."""
    if size is None:
        return image.astype(np.float32)
    with torch.no_grad():
        x = torch.tensor(image[None, ...], dtype=torch.float32)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.squeeze(0).numpy().astype(np.float32)


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
        self._compute_normalization()

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

    def _compute_normalization(self):
        self.channel_mean = self.images.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self.channel_std = (self.images.std(axis=(0, 2, 3), keepdims=True) + 1e-8).astype(np.float32)
        self.images = (self.images - self.channel_mean) / self.channel_std
        print("Channel normalization:")
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


def vae_loss(recon, x, mu, logvar, beta=BETA_KL):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


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


def plot_reconstructions(model, dataset, device, n_examples=3):
    model.eval()
    n_examples = min(n_examples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)
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


def plot_generated_samples(model, dataset, device, samples_per_c=2):
    model.eval()
    c_values = dataset.unique_c_values()
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


def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SaturatedSnapshotDataset(DATA_DIR)
    image_size = dataset.image_shape[0]
    train_len = int(TRAIN_FRACTION * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
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
    }, MODEL_OUT)
    print(f"[Success] Saved '{MODEL_OUT}'.")

    plot_loss_curve(history)
    plot_reconstructions(model, dataset, device)
    plot_generated_samples(model, dataset, device)


if __name__ == "__main__":
    train()