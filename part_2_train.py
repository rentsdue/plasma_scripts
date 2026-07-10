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

# -----------------------------------------------------------------------------
# 1. Dataset Extraction: Concatenated Log-Power Spectra
# -----------------------------------------------------------------------------
class SaturatedSpectraDataset(Dataset):
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
                window = range(T // 2, T)
                
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
                
                log_E_kx = np.log10(E_kx_bar[1:171] + 1e-20)
                log_E_ky = np.log10(E_ky_bar[1:171] + 1e-20)
                log_E_zonal = np.log10(E_zonal_bar[1:171] + 1e-20)
                
                y_combined = np.concatenate([log_E_kx, log_E_ky, log_E_zonal])
                
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
# 2. Main LOOCV Processing Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = SaturatedSpectraDataset(DATA_DIR)
    N_POD_MODES = 4
    num_points = len(dataset)
    results = []
    
    print("Executing out-of-sample training folds across combined k-spectra maps...")
    for test_idx in range(num_points):
        test_c_val = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]
        point_type = "Endpoint" if (test_c_val == min(dataset.raw_c_values) or test_c_val == max(dataset.raw_c_values)) else "Interior"
        
        train_profiles_raw = dataset.log_profiles[train_indices]
        true_y = dataset.log_profiles[test_idx]
        
        # Fit POD dimensionality reductions
        pca = PCA(n_components=N_POD_MODES)
        train_coefficients = pca.fit_transform(train_profiles_raw)
        
        coeff_mean, coeff_std = np.mean(train_coefficients, axis=0), np.std(train_coefficients, axis=0) + 1e-8
        train_coeffs_scaled = (train_coefficients - coeff_mean) / coeff_std
        
        train_inputs = torch.tensor(dataset.inputs[train_indices]).to(device)
        train_targets = torch.tensor(train_coeffs_scaled, dtype=torch.float32).to(device)
        test_input = torch.tensor(dataset.inputs[test_idx]).unsqueeze(0).to(device)
        
        # Baseline Linear Spline Model
        interp_func = interp1d(
            dataset.inputs[train_indices].squeeze(), train_profiles_raw, 
            axis=0, kind='linear', fill_value="extrapolate"
        )
        base_pred_y = interp_func(dataset.inputs[test_idx].squeeze())
        
        # Train Network
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
            
        # Reconstruct full concatenated spectra 
        nn_pred_coeffs = (nn_pred_coeffs_scaled * coeff_std) + coeff_mean
        nn_pred_y = pca.inverse_transform(nn_pred_coeffs.reshape(1, -1)).squeeze()
        
        # Track individual and global absolute decade deviation (DEX)
        nn_dex = np.median(np.abs(nn_pred_y - true_y))
        base_dex = np.median(np.abs(base_pred_y - true_y))
        
        results.append({
            "C_Value": test_c_val, "Type": point_type,
            "NN_DEX": nn_dex, "Base_DEX": base_dex,
            "True_Profile": true_y, "NN_Profile": nn_pred_y
        })

    # -----------------------------------------------------------------------------
    # 3. Compile Logs and Diagnostics
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values(by="C_Value")
    
    print("\n" + "="*85)
    print(f"{'STEP 2 COMBINED K-SPECTRA MEDIAN ABSOLUTE ERROR (DEX) REVIEW':^85}")
    print("="*85)
    print(f"{'C-Value':<8} | {'Type':<8} | {'NN Error (DEX Decades)':<25} | {'Base Error (DEX Decades)':<24}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(f"{row['C_Value']:<8.3f} | {row['Type']:<8} | {row['NN_DEX']:<25.4f} | {row['Base_DEX']:<24.4f}")
    print("="*85)

    # -----------------------------------------------------------------------------
    # 4. Generate Heatmap Set for Zonal Spectrum Component E_zonal(kx)
    # -----------------------------------------------------------------------------
    log10_c_axis = np.log10(df['C_Value'].values)
    modes_axis = np.arange(1, 65)
    
    # Isolate the Zonal Spectrum segment (indices 340 to 340+64) for detailed verification mapping
    true_zonal_cropped = np.stack(df['True_Profile'].values)[:, 340:340+64]
    nn_zonal_cropped = np.stack(df['NN_Profile'].values)[:, 340:340+64]
    
    vmax = true_zonal_cropped.max()
    vmin = vmax - 8.0  # Safe 8-decade scaling structure
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
    
    im0 = axs[0].pcolormesh(modes_axis, log10_c_axis, true_zonal_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title(r"Fig 7a: True Zonal Spectrum Log $E_{\text{zonal}}(k_x)$", fontsize=12)
    axs[0].set_xlabel("Mode Index $m$ (1 to 64)", fontsize=11)
    axs[0].set_ylabel(r"$\log_{10} C$", fontsize=11)
    fig.colorbar(im0, ax=axs[0], label=r"Log Energy Intensity")
    
    im1 = axs[1].pcolormesh(modes_axis, log10_c_axis, nn_zonal_cropped, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title("Fig 7b: LOOCV Mode Prediction", fontsize=12)
    axs[1].set_xlabel("Mode Index $m$ (1 to 64)", fontsize=11)
    axs[1].set_ylabel(r"$\log_{10} C$", fontsize=11)
    fig.colorbar(im1, ax=axs[1], label=r"Log Energy Intensity")
    
    axs[2].plot(log10_c_axis, df['NN_DEX'].values, marker='o', color='#2ca02c', linewidth=2, label='POD-FFNN Model')
    axs[2].set_title("Fig 7c: Unified k-Spectra Global DEX Error", fontsize=12)
    axs[2].set_xlabel(r"$\log_{10} C$", fontsize=11)
    axs[2].set_ylabel("Global Median Absolute Error [Decades]", fontsize=11)
    axs[2].grid(True, which="both", alpha=0.35, linestyle=':')
    axs[2].legend(loc='upper right')
    
    plt.tight_layout()
    output_image_name = "step2_spectra_validation_performance.png"
    plt.savefig(output_image_name, dpi=220)
    print(f"\n[Success] Step 2 structural graphics compiled and saved to '{output_image_name}'")
