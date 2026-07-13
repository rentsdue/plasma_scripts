import os
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


# -----------------------------------------------------------------------------
# 1. Dataset Extraction: Sliced Log-Power Targets
# -----------------------------------------------------------------------------
# Purpose:
#   Build Step 1 training targets for zonal-flow log-power profiles.
# How it works:
#   Time-averages saturated zonal Fourier power and stores log10 profiles versus log10(C).
class SaturatedLogPowerDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        self.log_profiles = []  
        
        self._process_files()
        
    def _process_files(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                c_val = f['params/C'][()]
                x_input = np.log10(c_val)
                
                uk = f['fields/uk']
                kx2d = f['data/kx'][()]
                ky2d = f['data/ky'][()]
                
                T = uk.shape[0]
                ke_series = compute_total_kinetic_energy_series(uk, kx2d, ky2d)
                window, sat_info = detect_saturation_window(ke_series)
                t_start = sat_info["t_start"]
                block_means = sat_info["block_means"]
                
                profiles_t = []
                for t in window:
                    potential_fft = uk[t, 0, :, :]
                    kx1d = kx2d[:, 0]
                    U_zf_fourier = 1j * kx1d * potential_fft[:, 0]
                    profiles_t.append(np.abs(U_zf_fourier))
                
                # Compute time-mean power, drop m=0 baseline, keep Hermitian half
                P_bar = np.mean(np.square(profiles_t), axis=0)
                y = np.log10(P_bar[1:171] + 1e-20)
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.log_profiles.append(y)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.log_profiles = np.array(self.log_profiles, dtype=np.float32)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return torch.tensor(self.inputs[idx]), torch.tensor(self.log_profiles[idx])

# Purpose:
#   Predict POD/PCA coefficients from log10(C).
# How it works:
#   Uses a small fully connected network to output low-dimensional modal coefficients.
class PodCoefficientFFNN(nn.Module):
    def __init__(self, input_dim=1, num_modes=4):
        super(PodCoefficientFFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
    def forward(self, x): return self.network(x)

# -----------------------------------------------------------------------------
# 3. Main Processing Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SaturatedLogPowerDataset(DATA_DIR)
    N_POD_MODES = 4
    num_points = len(dataset)
    results = []
    
    print("Executing out-of-sample training folds across log-power spectrum...")
    for test_idx in range(num_points):
        test_c_val = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]
        point_type = "Endpoint" if (test_c_val == min(dataset.raw_c_values) or test_c_val == max(dataset.raw_c_values)) else "Interior"
        
        train_profiles_raw = dataset.log_profiles[train_indices]
        true_y = dataset.log_profiles[test_idx]
        
        # Fit POD on training split log-power targets directly
        pca = PCA(n_components=N_POD_MODES)
        train_coefficients = pca.fit_transform(train_profiles_raw)
        
        coeff_mean, coeff_std = np.mean(train_coefficients, axis=0), np.std(train_coefficients, axis=0) + 1e-8
        train_coeffs_scaled = (train_coefficients - coeff_mean) / coeff_std
        
        train_inputs = torch.tensor(dataset.inputs[train_indices]).to(device)
        train_targets = torch.tensor(train_coeffs_scaled, dtype=torch.float32).to(device)
        test_input = torch.tensor(dataset.inputs[test_idx]).unsqueeze(0).to(device)
        
        # Baseline Model: Linear Interp across training log-power spectrums
        interp_func = interp1d(
            dataset.inputs[train_indices].squeeze(), train_profiles_raw, 
            axis=0, kind='linear', fill_value="extrapolate"
        )
        base_pred_y = interp_func(dataset.inputs[test_idx].squeeze())
        
        # Train Neural Network
        model = PodCoefficientFFNN(num_modes=N_POD_MODES).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(1200):
            outputs = model(train_inputs)
            loss = criterion(outputs, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            nn_pred_coeffs_scaled = model(test_input).cpu().numpy().squeeze()
            
        # Reconstruct prediction in log-power space
        nn_pred_coeffs = (nn_pred_coeffs_scaled * coeff_std) + coeff_mean
        nn_pred_y = pca.inverse_transform(nn_pred_coeffs.reshape(1, -1)).squeeze()
        
        # METRIC FIX: Calculate Median Absolute Deviation (DEX) over modes
        nn_dex = np.median(np.abs(nn_pred_y - true_y))
        base_dex = np.median(np.abs(base_pred_y - true_y))
        
        results.append({
            "C_Value": test_c_val, "Type": point_type,
            "NN_DEX": nn_dex, "Base_DEX": base_dex,
            "True_Profile": true_y, "NN_Profile": nn_pred_y
        })

    # -----------------------------------------------------------------------------
    # 4. Generate Standardized Graphics and Report
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values(by="C_Value")
    
    print("\n" + "="*85)
    print(f"{'STEP 1 LOG-POWER PROFILE MEDIAN ABSOLUTE ERROR (DEX) REVIEW':^85}")
    print("="*85)
    print(f"{'C-Value':<8} | {'Type':<8} | {'NN Error (DEX Decades)':<25} | {'Base Error (DEX Decades)':<24}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(f"{row['C_Value']:<8.3f} | {row['Type']:<8} | {row['NN_DEX']:<25.4f} | {row['Base_DEX']:<24.4f}")
    print("="*85)

    log10_c_axis = np.log10(df['C_Value'].values)
    nn_dex_errors = df['NN_DEX'].values
    base_dex_errors = df['Base_DEX'].values
    
    # DISPLAY CROP ADJUSTMENT: Limit horizontal axis visualization to modes 1 to 64
    modes_axis = np.arange(1, 65)
    true_matrix_cropped = np.stack(df['True_Profile'].values)[:, :64]
    nn_matrix_cropped = np.stack(df['NN_Profile'].values)[:, :64]
    
    # VISUALIZATION RANGE CLIPPING: Compute dynamic floor based on maximum value
    vmax = true_matrix_cropped.max()
    vmin = vmax - 8.0  # Captures exactly 8 decades of data dynamics
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # Panel 6a: Cropped True Power Spectrum Heatmap
    im0 = axs[0].pcolormesh(modes_axis, log10_c_axis, true_matrix_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(r"Simulation: $\log_{10} P_{ZF}(m; C)$", fontsize=12)
    axs[0].set_xlabel(r"Radial mode number $m$", fontsize=11)
    axs[0].set_ylabel(r"Adiabaticity $\log_{10}(C)$", fontsize=11)
    fig.colorbar(im0, ax=axs[0], label=r"Zonal-flow power $\log_{10} P_{ZF}$")
    
    # Panel 6b: Cropped Prediction Power Spectrum Heatmap
    im1 = axs[1].pcolormesh(modes_axis, log10_c_axis, nn_matrix_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title(r"ML prediction: $\log_{10} P_{ZF}(m; C)$", fontsize=12)
    axs[1].set_xlabel(r"Radial mode number $m$", fontsize=11)
    axs[1].set_ylabel(r"Adiabaticity $\log_{10}(C)$", fontsize=11)
    fig.colorbar(im1, ax=axs[1], label=r"Zonal-flow power $\log_{10} P_{ZF}$")
    
    # Panel 6c: Performance Curve using Median Deviation (DEX)
    axs[2].plot(log10_c_axis, nn_dex_errors, marker='o', color='#1f77b4', linewidth=2, markersize=6, label='POD-FFNN prediction')
    axs[2].plot(log10_c_axis, base_dex_errors, marker='s', color='#ff7f0e', linestyle='--', linewidth=2, markersize=5, label='Linear interpolation baseline')
    axs[2].set_title(r"Median Spectrum Error vs $C$", fontsize=12)
    axs[2].set_xlabel(r"Adiabaticity $\log_{10}(C)$", fontsize=11)
    axs[2].set_ylabel("Median absolute error [dex]", fontsize=11)
    axs[2].grid(True, which="both", alpha=0.35, linestyle=':')
    axs[2].legend(loc='upper right')
    
    plt.tight_layout()
    output_image_name = "step1_fourier_validation_performance_log.png"
    plt.savefig(output_image_name, dpi=220)
    print(f"\n[Success] Mode-space graphics compiled and saved to '{output_image_name}'")
