import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Dataset with Full Error Bar & Convergence Tracking
# -----------------------------------------------------------------------------
class MultiScalarZFDatasetWithErrorBars(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        self.targets = []              # Pure [log10(ZF_RMS), log10(ZF_Energy), log10(|Gamma_n|)]
        self.true_linear_vals = []     # [ZF_RMS_mean, ZF_Energy_mean, Gamma_n_mean]
        self.time_stds = []            # [ZF_RMS_std, ZF_Energy_std, Gamma_n_std]
        self.flux_converged_flags = [] # Booleans for signal vs noise filter
        
        self._process_files()
        
    def _process_files(self):
        print("Processing files and extracting statistics with time-series error margins...")
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                c_val = f['params/C'][()]
                x_input = np.log10(c_val) # Clean mapping; no artificial floor offset
                kappa = f['params/kappa'][()] if 'params/kappa' in f else 1.0
                
                uk = f['fields/uk']
                kx2d = f['data/kx'][()]
                ky2d = f['data/ky'][()]
                
                T = uk.shape[0]
                window = range(T // 2, T) 
                N_window = len(window)
                
                rms_t, gamma_t, energy_t = [], [], []
                
                for t in window:
                    # INDEX CORRECTION: Index 0 is Potential (phi), Index 1 is Density (n)
                    potential_fft = uk[t, 0, :, :]
                    density_fft   = uk[t, 1, :, :] 
                    
                    gradx_phi = np.fft.irfft2((1j * kx2d) * potential_fft, s=(340, 340))
                    grady_phi = np.fft.irfft2((1j * ky2d) * potential_fft, s=(340, 340))
                    density   = np.fft.irfft2(density_fft, s=(340, 340))
                    
                    U_zf = np.mean(gradx_phi, axis=-1)
                    
                    rms_t.append(np.sqrt(np.mean(U_zf**2)))
                    energy_t.append(0.5 * np.mean(U_zf**2))
                    gamma_t.append(-kappa * np.mean(density * grady_phi))
                
                # Compute statistical means and standard deviations
                zf_rms_avg = np.mean(rms_t)
                zf_energy_avg = np.mean(energy_t)
                gamma_n_avg = np.mean(gamma_t)
                
                zf_rms_std = np.std(rms_t)
                zf_energy_std = np.std(energy_t)
                gamma_n_std = np.std(gamma_t)
                
                # Check Statistical Flux Convergence: |mean| < 2 * std / sqrt(N)
                is_flux_converged = np.abs(gamma_n_avg) >= (2.0 * gamma_n_std / np.sqrt(N_window))
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.targets.append([
                    np.log10(zf_rms_avg),
                    np.log10(zf_energy_avg),
                    np.log10(np.abs(gamma_n_avg))
                ])
                self.true_linear_vals.append([zf_rms_avg, zf_energy_avg, gamma_n_avg])
                self.time_stds.append([zf_rms_std, zf_energy_std, gamma_n_std])
                self.flux_converged_flags.append(is_flux_converged)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.true_linear_vals = np.array(self.true_linear_vals, dtype=np.float32)
        self.time_stds = np.array(self.time_stds, dtype=np.float32)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

# -----------------------------------------------------------------------------
# 2. Neural Network Architecture
# -----------------------------------------------------------------------------
class ScalarFFNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=3):
        super(ScalarFFNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x): return self.network(x)

# -----------------------------------------------------------------------------
# 3. Execution & LOOCV Loop
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    set_seed(42)
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = MultiScalarZFDatasetWithErrorBars(DATA_DIR)
    num_points = len(dataset)
    results = []
    
    print("Beginning Step 0 Multi-Scalar LOOCV Validation Evaluation Loop...")
    for test_idx in range(num_points):
        test_c_val = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]
        point_type = "Endpoint" if (test_c_val == min(dataset.raw_c_values) or test_c_val == max(dataset.raw_c_values)) else "Interior"
        
        # Split Data
        train_inputs = torch.tensor(dataset.inputs[train_indices]).to(device)
        train_targets = torch.tensor(dataset.targets[train_indices]).to(device)
        test_input = torch.tensor(dataset.inputs[test_idx]).unsqueeze(0).to(device)
        
        true_log_values = dataset.targets[test_idx]
        true_linear_values = dataset.true_linear_vals[test_idx]
        is_flux_converged = dataset.flux_converged_flags[test_idx]
        
        # --- Baseline Classical Model: Linear Spline Interpolation in Log-Space ---
        interp_func = interp1d(
            dataset.inputs[train_indices].squeeze(), 
            dataset.targets[train_indices], 
            axis=0, kind='linear', fill_value="extrapolate"
        )
        base_pred_log = interp_func(dataset.inputs[test_idx].squeeze())
        
        # --- Machine Learning Model: Feedforward Neural Network ---
        model = ScalarFFNN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(1500):
            outputs = model(train_inputs)
            loss = criterion(outputs, train_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            nn_pred_log = model(test_input).cpu().numpy().squeeze()
            
        # Reconstruct absolute metrics for physical tracking
        nn_pred_rms, nn_pred_eng = 10**nn_pred_log[0], 10**nn_pred_log[1]
        base_pred_rms, base_pred_eng = 10**base_pred_log[0], 10**base_pred_log[1]
        
        true_rms, true_eng = true_linear_values[0], true_linear_values[1]
        
        # Percentage Errors for Volumetric Metrics
        nn_rms_err = (abs(true_rms - nn_pred_rms) / true_rms) * 100.0
        nn_eng_err = (abs(true_eng - nn_pred_eng) / true_eng) * 100.0
        base_rms_err = (abs(true_rms - base_pred_rms) / true_rms) * 100.0
        base_eng_err = (abs(true_eng - base_pred_eng) / true_eng) * 100.0
        
        # FIX: Score Flux as Absolute Distance in Log10 Space with Convergence Validation
        if is_flux_converged:
            nn_flx_err = abs(nn_pred_log[2] - true_log_values[2])
            base_flx_err = abs(base_pred_log[2] - true_log_values[2])
        else:
            nn_flx_err = np.nan  # Drop from scoring entirely if it is unconverged noise
            base_flx_err = np.nan
            
        results.append({
            "C_Value": test_c_val, "Type": point_type, "Is_Flux_Converged": is_flux_converged,
            "NN_RMS_Err": nn_rms_err, "NN_Eng_Err": nn_eng_err, "NN_Flx_LogErr": nn_flx_err,
            "Base_RMS_Err": base_rms_err, "Base_Eng_Err": base_eng_err, "Base_Flx_LogErr": base_flx_err,
            "True_RMS": true_rms, "True_Eng": true_eng, "True_Flux": true_linear_values[2],
            "NN_Pred_RMS": nn_pred_rms, "Base_Pred_RMS": base_pred_rms,
            "RMS_Std": dataset.time_stds[test_idx][0], "Flux_Std": dataset.time_stds[test_idx][2]
        })

    # -----------------------------------------------------------------------------
    # 4. Reporting Summary Logs
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values(by="C_Value")
    
    print("\n" + "="*95)
    print(f"{'STEP 0 MULTI-SCALAR METRIC TRACKING REPORT':^95}")
    print("="*95)
    print(f"{'C-Value':<8} | {'Type':<8} | {'NN RMS %':<10} | {'Base RMS %':<12} | {'NN Flux LogErr':<15} | {'Base Flux LogErr':<17}")
    print("-" * 95)
    for _, r in df.iterrows():
        flx_nn_str = f"{r['NN_Flx_LogErr']:.4f}" if not np.isnan(r['NN_Flx_LogErr']) else "Noise (Skipped)"
        flx_base_str = f"{r['Base_Flx_LogErr']:.4f}" if not np.isnan(r['Base_Flx_LogErr']) else "Noise (Skipped)"
        print(f"{r['C_Value']:<8.3f} | {r['Type']:<8} | {r['NN_RMS_Err']:<10.2f}% | {r['Base_RMS_Err']:<12.2f}% | {r['NN_Flx_LogErr']:<15} | {r['Base_Flx_LogErr']:<17}")
    print("="*95)
    
    # 5. Save C-Curves with Error Bars
    plt.figure(figsize=(10, 5))
    plt.errorbar(df['C_Value'], df['True_RMS'], yerr=df['RMS_Std'], fmt='o', color='black', label='Simulation mean $\pm$ time variation', capsize=4)
    plt.plot(df['C_Value'], df['NN_Pred_RMS'], 'r-x', label='Neural-network prediction')
    plt.plot(df['C_Value'], df['Base_Pred_RMS'], 'b--.', label='Linear interpolation baseline')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r"Predicting Zonal-Flow Strength: $RMS_{ZF}(C)$")
    plt.xlabel(r"Adiabaticity parameter $C$")
    plt.ylabel(r"Zonal-flow RMS amplitude, $RMS_{ZF}$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("step0_c_curves_with_errorbars.png")
    print("[Success] Output scalar diagnostic plot saved to 'step0_c_curves_with_errorbars.png'")
