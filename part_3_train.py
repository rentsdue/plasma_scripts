import os
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. Global Seeding Engine for Cross-System Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# -----------------------------------------------------------------------------
# 1. 2D Data Extraction Engine (Targeting Pure 2D Spectral Structures)
# -----------------------------------------------------------------------------
class Saturated2DSpectrumDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        # Target matrix shape: [17 snapshots, 1 channel * 340 * 340 features]
        self.flat_targets = [] 
        self.unflattened_spectra = []
        
        self._process_files()
        
    def _process_files(self):
        print("Extracting time-mean 2D Wavenumber Power Spectra...")
        
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                c_val = f['params/C'][()]
                x_input = np.log10(c_val)
                
                uk = f['fields/uk']  # Shape: [Time, Channel, kx, ky]
                T = uk.shape[0]
                window = range(T // 2, T)
                N_t = len(window)
                
                spectrum_2d_accum = np.zeros((340, 340))
                
                for t in window:
                    # Pull half-spectrum complex modes for electrostatic potential
                    phi_k = uk[t, 0, :, :]
                    
                    # Invert back to real physical space using explicit grid constraints
                    phi_space = np.fft.irfft2(phi_k, s=(340, 340), norm="forward")
                    
                    # Transform back via full FFT2 to yield a complete, square 340x340 spectral map
                    phi_full_k = np.fft.fft2(phi_space, norm="forward")
                    spectrum_2d_accum += np.square(np.abs(phi_full_k))
                
                # Compute temporal average and apply micro-floored log10 transformation
                mean_spectrum_2d = spectrum_2d_accum / N_t
                log_spectrum_2d = np.log10(mean_spectrum_2d + 1e-20)
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.flat_targets.append(log_spectrum_2d.flatten())
                self.unflattened_spectra.append(log_spectrum_2d)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.flat_targets = np.array(self.flat_targets, dtype=np.float32)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return torch.tensor(self.inputs[idx]), torch.tensor(self.flat_targets[idx])

# -----------------------------------------------------------------------------
# 2. Hybrid ML Architecture: Fully Connected BOTTLENECK + CNN DECODER (FIXED)
# -----------------------------------------------------------------------------
class SpectralPodCnnDecoder(nn.Module):
    def __init__(self, input_dim=1, num_modes=4):
        super(SpectralPodCnnDecoder, self).__init__()
        
        # 1. Bottleneck Mapping Stage
        self.mlp_bottleneck = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
        
        # 2. Convolutional Spatial Refiner
        self.cnn_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x, pca_model, coeff_mean, coeff_std, device):
        # 1. Forward pass through MLP to isolate normalized POD coordinates
        scaled_coeffs = self.mlp_bottleneck(x)  # Shape: [Batch, num_modes]
        
        # 2. Safely cast PCA parameters and statistics to PyTorch tensors on the GPU/CPU
        c_mean = torch.as_tensor(coeff_mean, dtype=torch.float32, device=device)
        c_std = torch.as_tensor(coeff_std, dtype=torch.float32, device=device)
        
        pca_comps = torch.as_tensor(pca_model.components_, dtype=torch.float32, device=device)  # Shape: [num_modes, 115600]
        pca_mean = torch.as_tensor(pca_model.mean_, dtype=torch.float32, device=device)        # Shape: [115600]
        
        # 3. Differentiable inverse scaling
        unscaled_coeffs = (scaled_coeffs * c_std) + c_mean
        
        # 4. Differentiable Inverse POD Projection via PyTorch Matrix Multiplication
        flat_reconstruction = torch.matmul(unscaled_coeffs, pca_comps) + pca_mean  # Shape: [Batch, 115600]
        
        # 5. Reshape natively to Image Tensor format without disrupting the autograd graph
        image_tensor = flat_reconstruction.view(-1, 1, 340, 340)
        
        # 6. Refine through continuous convolutional filtering
        final_corrected_images = self.cnn_refiner(image_tensor)
        return final_corrected_images

# -----------------------------------------------------------------------------
# 3. Main Execution Engine: Out-of-Sample LOOCV Training Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Saturated2DSpectrumDataset(DATA_DIR)
    N_POD_MODES = 4
    num_points = len(dataset)
    results = []
    
    print("\nRunning Focused 2D Spectral Regression loop over {} data nodes...".format(num_points))
    for test_idx in range(num_points):
        test_c_val = dataset.raw_c_values[test_idx]
        train_indices = [i for i in range(num_points) if i != test_idx]
        
        train_profiles_raw = dataset.flat_targets[train_indices]
        true_flat_y = dataset.flat_targets[test_idx]
        
        # Dimensionality reduction focused purely on the 2D spectrum family
        pca = PCA(n_components=N_POD_MODES)
        train_coefficients = pca.fit_transform(train_profiles_raw)
        
        coeff_mean = np.mean(train_coefficients, axis=0)
        coeff_std = np.std(train_coefficients, axis=0) + 1e-8
        train_coeffs_scaled = (train_coefficients - coeff_mean) / coeff_std
        
        # Format tensors for PyTorch optimization
        train_inputs = torch.tensor(dataset.inputs[train_indices]).to(device)
        train_target_images = torch.tensor(train_profiles_raw, dtype=torch.float32).view(-1, 1, 340, 340).to(device)
        test_input = torch.tensor(dataset.inputs[test_idx]).unsqueeze(0).to(device)
        
        true_spectrum_map = true_flat_y.reshape(340, 340)
        
        # Initialize architecture and optimizer
        model = SpectralPodCnnDecoder(num_modes=N_POD_MODES).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        
        # Optimization Loop
        model.train()
        for epoch in range(1200):
            pred_images = model(train_inputs, pca, coeff_mean, coeff_std, device)
            loss = criterion(pred_images, train_target_images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # [UPDATED EVALUATION BLOCK INSIDE THE LOOCV LOOP]
        model.eval()
        with torch.no_grad():
            nn_pred_image = model(test_input, pca, coeff_mean, coeff_std, device).cpu().squeeze().numpy()
            
        spec_dex = np.median(np.abs(nn_pred_image - true_spectrum_map))
        
        # Collapse 2D spectra into 1D profiles along the poloidal mode axis (k_y)
        # We slice [1:65] to extract modes 1 to 64, bypassing the DC offset at index 0
        true_profile_1d = np.mean(true_spectrum_map, axis=0)[1:65]
        nn_profile_1d = np.mean(nn_pred_image, axis=0)[1:65]
        
        results.append({
            "C_Value": test_c_val,
            "Spec_DEX": spec_dex,
            "True_Img": true_spectrum_map,
            "Pred_Img": nn_pred_image,
            "True_Profile": true_profile_1d, # Saved for the parametric sweep check
            "NN_Profile": nn_profile_1d      # Saved for the parametric sweep check
        })
        print(" -> Completed Fold: Out-of-sample C={:.3e} | Spectral DEX: {:.4f}".format(test_c_val, spec_dex))

    # -----------------------------------------------------------------------------
    # 4. Compile Diagnostic Performance Report
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values(by="C_Value")
    print("\n" + "="*50)
    print("{:^50}".format("STEP 3 2D SPECTRUM REGRESSION REPORT (DEX)")); print("="*50)
    for _, r in df.iterrows():
        print("{:<15.4f} | {:<25.4f}".format(r['C_Value'], r['Spec_DEX']))
    print("="*50)

# -----------------------------------------------------------------------------
    # 5. Restore 2D Validation Panel (Including the Missing Error Map)
    # -----------------------------------------------------------------------------
    sample_row = df.iloc[len(df)//2]  # Pick the median C-value node for out-of-sample view
    c_sample = sample_row['C_Value']
    t_img = sample_row['True_Img']
    p_img = sample_row['Pred_Img']
    
    fig2d, axs2d = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Left Panel: Ground Truth 2D Spectrum
    im0_2d = axs2d[0].imshow(t_img, cmap='inferno', origin='lower')
    axs2d[0].set_title("True $\log_{10}(P_{\phi}(k_x, k_y))$")
    fig2d.colorbar(im0_2d, ax=axs2d[0])
    
    # Center Panel: CNN Decoder Prediction
    im1_2d = axs2d[1].imshow(p_img, cmap='inferno', origin='lower')
    axs2d[1].set_title("Predicted $\log_{10}(P_{\phi}(k_x, k_y))$")
    fig2d.colorbar(im1_2d, ax=axs2d[1])
    
    # Right Panel: Absolute Error Map (Restored!)
    error_map = np.abs(p_img - t_img)
    im2_2d = axs2d[2].imshow(error_map, cmap='viridis', origin='lower')
    axs2d[2].set_title("Absolute Error Map (DEX)")
    fig2d.colorbar(im2_2d, ax=axs2d[2])
    
    plt.suptitle("Step 3 2D Spectral Verification Panel — Out-of-Sample Node C = {:.4f}".format(c_sample), fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_image_2d = "step3_2d_spectrum_validation.png"
    plt.savefig(output_image_2d, dpi=200)
    print("\n[Success] 2D Error Validation panel exported to '{}'.".format(output_image_2d))

    # -----------------------------------------------------------------------------
    # 6. Export Part 1 Style Mode-Space Parametric Sweep Contour Plot
    # -----------------------------------------------------------------------------
    fig1d, axs1d = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    # DISPLAY CROP ADJUSTMENT: Limit horizontal axis visualization to modes 1 to 64
    modes_axis = np.arange(1, 65)
    true_matrix_cropped = np.stack(df['True_Profile'].values)[:, :64]
    nn_matrix_cropped = np.stack(df['NN_Profile'].values)[:, :64]
    c_values = df['C_Value'].values
    
    # Left Panel: Ground Truth Simulation Sweep
    im0_1d = axs1d[0].pcolormesh(modes_axis, c_values, true_matrix_cropped, cmap='inferno', shading='auto')
    axs1d[0].set_yscale('log')
    axs1d[0].set_title("True Spectral Sweep: $\log_{10}(P(k_y))$", fontsize=11, fontweight='bold')
    axs1d[0].set_xlabel("Poloidal Mode ($m$)", fontsize=10)
    axs1d[0].set_ylabel("Parallel Conductivity ($C$)", fontsize=10)
    fig1d.colorbar(im0_1d, ax=axs1d[0], label="Log Power Density")
    
    # Right Panel: Out-of-Sample CNN Prediction Sweep
    im1_1d = axs1d[1].pcolormesh(modes_axis, c_values, nn_matrix_cropped, cmap='inferno', shading='auto')
    axs1d[1].set_yscale('log')
    axs1d[1].set_title("NN Predicted Out-of-Sample Sweep: $\log_{10}(P(k_y))$", fontsize=11, fontweight='bold')
    axs1d[1].set_xlabel("Poloidal Mode ($m$)", fontsize=10)
    fig1d.colorbar(im1_1d, ax=axs1d[1], label="Log Power Density")
    
    plt.suptitle("Step 3 Verification: 1D Poloidal Mode Sweep vs. $C$ (Modes 1–64)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    sweep_output_image = "step3_mode_space_sweep_comparison.png"
    plt.savefig(sweep_output_image, dpi=200)
    print("[Success] Part 1 style parametric sweep plot exported to '{}'.".format(sweep_output_image))
