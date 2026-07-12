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

# Guarantee cross-system mathematical reproducibility
torch.manual_seed(0)
np.random.seed(0)


def compute_total_kinetic_energy_series(uk, kx2d, ky2d):
    series = []
    for t in range(uk.shape[0]):
        phi_k = uk[t, 0, :, :]
        e_mode = 0.5 * (kx2d**2 + ky2d**2) * np.abs(phi_k)**2
        series.append(np.sum(e_mode))
    return np.array(series, dtype=np.float64)


def detect_saturation_window(series, num_blocks=12, tolerance=0.10, reference_blocks=3):
    """Detect the saturated window using block-mean convergence.

    This follows the project rule:
        E = scalar time series, K = 12 blocks, ref = mean of final 3 blocks,
        onset = first block after which all block means stay within 10% of ref.

    A tiny denominator floor is retained to avoid division-by-zero if ref ~ 0.
    """
    E = np.asarray(series, dtype=np.float64)
    K = num_blocks
    if len(E) < K:
        t_start = len(E) // 2
        return range(t_start, len(E)), t_start, E
    means = np.array([b.mean() for b in np.array_split(E, K)])
    ref = means[-reference_blocks:].mean()
    denom = max(abs(ref), 1e-20)
    onset = next(
        (i for i in range(K) if np.all(np.abs(means[i:] - ref) / denom < tolerance)),
        K // 2,
    )
    t_start = min(onset * (len(E) // K), len(E) - 1)
    window = range(t_start, len(E))
    return window, t_start, means

# -----------------------------------------------------------------------------
# 1. Dataset Extraction: Multi-Scale Folded Array Building
# -----------------------------------------------------------------------------
class SaturatedFoldedSpectraDataset(Dataset):
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
                window, t_start, block_means = detect_saturation_window(ke_series)
                
                E_kx_t, E_ky_t, E_zonal_t = [], [], []
                for t in window:
                    phi_k = uk[t, 0, :, :]
                    E_mode = 0.5 * (kx2d**2 + ky2d**2) * np.square(np.abs(phi_k))
                    
                    E_kx_t.append(np.sum(E_mode, axis=1))
                    E_ky_t.append(np.sum(E_mode, axis=0))
                    E_zonal_t.append(E_mode[:, 0])
                
                E_kx_bar = np.mean(E_kx_t, axis=0)
                E_ky_bar = np.mean(E_ky_t, axis=0)
                E_zonal_bar = np.mean(E_zonal_t, axis=0)
                
                # Fold wrap-around vectors to capture complete energy domains cleanly
                E_kx_fold = E_kx_bar[1:170] + E_kx_bar[171:340][::-1]          # 169
                E_ky_fold = E_ky_bar[1:171]                                    # 170
                E_zonal_fold = E_zonal_bar[1:170] + E_zonal_bar[171:340][::-1]  # 169
                
                y_combined = np.concatenate([
                    np.log10(E_kx_fold + 1e-20),
                    np.log10(E_ky_fold + 1e-20),
                    np.log10(E_zonal_fold + 1e-20)
                ])
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.log_profiles.append(y_combined)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.log_profiles = np.array(self.log_profiles, dtype=np.float32)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return torch.tensor(self.inputs[idx]), torch.tensor(self.log_profiles[idx])

class PodSpectraFFNN(nn.Module):
    def __init__(self, input_dim=1, num_modes=4):
        super(PodSpectraFFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
    def forward(self, x): return self.network(x)

# -----------------------------------------------------------------------------
# 2. Main Execution Engine: LOOCV Loop with Sub-Slice Analysis
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SaturatedFoldedSpectraDataset(DATA_DIR)
    N_POD_MODES = 4
    num_points = len(dataset)
    results = []
    
    print("Executing out-of-sample training folds across folded multi-spectrum matrices...")
    for test_idx in range(num_points):
        test_c_val = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]
        point_type = "Endpoint" if (test_c_val == min(dataset.raw_c_values) or test_c_val == max(dataset.raw_c_values)) else "Interior"
        
        train_profiles_raw = dataset.log_profiles[train_indices]
        true_y = dataset.log_profiles[test_idx]
        
        # Proper Orthogonal Decomposition over joint training slice
        pca = PCA(n_components=N_POD_MODES)
        train_coefficients = pca.fit_transform(train_profiles_raw)
        
        coeff_mean, coeff_std = np.mean(train_coefficients, axis=0), np.std(train_coefficients, axis=0) + 1e-8
        train_coeffs_scaled = (train_coefficients - coeff_mean) / coeff_std
        
        train_inputs = torch.tensor(dataset.inputs[train_indices]).to(device)
        train_targets = torch.tensor(train_coeffs_scaled, dtype=torch.float32).to(device)
        test_input = torch.tensor(dataset.inputs[test_idx]).unsqueeze(0).to(device)
        
        # Baseline Validation Model: Linear Spline Multi-Interpolation
        interp_func = interp1d(
            dataset.inputs[train_indices].squeeze(), train_profiles_raw, 
            axis=0, kind='linear', fill_value="extrapolate"
        )
        base_pred_y = interp_func(dataset.inputs[test_idx].squeeze())
        
        # Train Neural Network
        model = PodSpectraFFNN(num_modes=N_POD_MODES).to(device)
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
            
        nn_pred_coeffs = (nn_pred_coeffs_scaled * coeff_std) + coeff_mean
        nn_pred_y = pca.inverse_transform(nn_pred_coeffs.reshape(1, -1)).squeeze()
        
        # --- DE-COUPLED ERROR METRIC EXTRACTION ENGINE ---
        # Splitting the 508 array along corrected mathematical boundaries: [0:169], [169:339], [339:508]
        nn_ex_dex = np.median(np.abs(nn_pred_y[0:169] - true_y[0:169]))
        bs_ex_dex = np.median(np.abs(base_pred_y[0:169] - true_y[0:169]))
        
        nn_ey_dex = np.median(np.abs(nn_pred_y[169:339] - true_y[169:339]))
        bs_ey_dex = np.median(np.abs(base_pred_y[169:339] - true_y[169:339]))
        
        nn_zf_dex = np.median(np.abs(nn_pred_y[339:508] - true_y[339:508]))
        bs_zf_dex = np.median(np.abs(base_pred_y[339:508] - true_y[339:508]))
        
        # Global multi-spectrum variance check
        nn_global = np.median(np.abs(nn_pred_y - true_y))
        bs_global = np.median(np.abs(base_pred_y - true_y))
        
        results.append({
            "C_Value": test_c_val, "Type": point_type,
            "NN_Global": nn_global, "Base_Global": bs_global,
            "NN_Ex": nn_ex_dex, "Base_Ex": bs_ex_dex,
            "NN_Ey": nn_ey_dex, "Base_Ey": bs_ey_dex,
            "NN_Zf": nn_zf_dex, "Base_Zf": bs_zf_dex,
            "True_Profile": true_y, "NN_Profile": nn_pred_y
        })

    # -----------------------------------------------------------------------------
    # 3. Compile Diagnostic Report
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values(by="C_Value")
    
    print("\n" + "="*125)
    print(f"{'STEP 2 DE-COUPLED SPECTRUM ERROR REVIEW (VALUES IN DEX DECADES)':^125}")
    print("="*125)
    
        # Create the header string using single quotes only

    # -----------------------------------------------------------------------------
    # 4. Comparative Plot Generation
    # -----------------------------------------------------------------------------
    log10_c_axis = np.log10(df['C_Value'].values)
    modes_axis = np.arange(1, 65)
    
    # Read specifically from the folded, shifted Zonal boundaries [339 : 339+64]
    true_zonal_cropped = np.stack(df['True_Profile'].values)[:, 339:339+64]
    nn_zonal_cropped = np.stack(df['NN_Profile'].values)[:, 339:339+64]
    
    vmax = true_zonal_cropped.max()
    vmin = vmax - 8.0
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    
    # Panel 7a: True Folded Heatmap
    im0 = axs[0].pcolormesh(modes_axis, log10_c_axis, true_zonal_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(r"Simulation: $\log_{10} E_{ZF}(|m|; C)$", fontsize=11)
    axs[0].set_xlabel(r"Radial mode magnitude $|m|$")
    axs[0].set_ylabel(r"Adiabaticity $\log_{10}(C)$")
    fig.colorbar(im0, ax=axs[0], label=r"Zonal kinetic energy $\log_{10} E_{ZF}$")
    
    # Panel 7b: NN Folded Heatmap
    im1 = axs[1].pcolormesh(modes_axis, log10_c_axis, nn_zonal_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title(r"ML prediction: $\log_{10} E_{ZF}(|m|; C)$", fontsize=11)
    axs[1].set_xlabel(r"Radial mode magnitude $|m|$")
    axs[1].set_ylabel(r"Adiabaticity $\log_{10}(C)$")
    fig.colorbar(im1, ax=axs[1], label=r"Zonal kinetic energy $\log_{10} E_{ZF}$")
    
    # Panel 7c: Comparative Multi-Family Error Panel with Baseline Overlay
    axs[2].plot(log10_c_axis, df['NN_Global'].values, color='black', marker='o', lw=2, label='All spectra: POD-FFNN')
    axs[2].plot(log10_c_axis, df['Base_Global'].values, color='black', linestyle='--', alpha=0.6, label='All spectra: interpolation')
    
    axs[2].plot(log10_c_axis, df['NN_Ex'].values, color='#1f77b4', marker='s', alpha=0.8, label=r'$E(k_x)$: POD-FFNN')
    axs[2].plot(log10_c_axis, df['Base_Ex'].values, color='#1f77b4', linestyle=':', alpha=0.5)
    
    axs[2].plot(log10_c_axis, df['NN_Zf'].values, color='#2ca02c', marker='^', alpha=0.8, label=r'$E_{ZF}$: POD-FFNN')
    axs[2].plot(log10_c_axis, df['Base_Zf'].values, color='#2ca02c', linestyle=':', alpha=0.5)
    
    axs[2].set_title("Prediction Error Compared with Interpolation", fontsize=11)
    axs[2].set_xlabel(r"Adiabaticity $\log_{10}(C)$")
    axs[2].set_ylabel("Median absolute error [dex]")
    axs[2].grid(True, which="both", alpha=0.35, linestyle=':')
    axs[2].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_image_name = "step2_spectra_validation_performance.png"
    plt.savefig(output_image_name, dpi=220)
    print(f"\n[Success] Refined diagnostic graphics compiled to '{output_image_name}'")
