import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FourierSpectraEDADataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        self.log_profiles = []  # Concatenated [E(kx), E(ky), E_zonal(kx)]
        
        self._process_files()
        
    def _process_files(self):
        print("Extracting time-mean 1D k-spectra (dropping m=0)...")
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
                    phi_k = uk[t, 0, :, :] # Potential (index 0)
                    
                    # Local Kinetic Energy per mode: 0.5 * (kx^2 + ky^2) * |phi|^2
                    # Accounting for norm="forward" storage metrics implicitly via absolute shapes
                    E_mode = 0.5 * (kx2d**2 + ky2d**2) * np.square(np.abs(phi_k))
                    
                    # 1D E(kx) is obtained by summing over the ky dimension
                    E_kx_t.append(np.sum(E_mode, axis=1))
                    
                    # 1D E(ky) is obtained by summing over the kx dimension
                    E_ky_t.append(np.sum(E_mode, axis=0))
                    
                    # Zonal spectrum: ky = 0 column only
                    E_zonal_t.append(E_mode[:, 0])
                
                # Compute time-mean profiles per run
                E_kx_bar = np.mean(E_kx_t, axis=0)
                E_ky_bar = np.mean(E_ky_t, axis=0)
                E_zonal_bar = np.mean(E_zonal_t, axis=0)
                
                # Slice out mode 0 and keep the Hermitian half (m = 1...170)
                # Apply micro-floor to protect high-wavenumber dead modes from -inf
                log_E_kx = np.log10(E_kx_bar[1:171] + 1e-20)
                log_E_ky = np.log10(E_ky_bar[1:171] + 1e-20)
                log_E_zonal = np.log10(E_zonal_bar[1:171] + 1e-20)
                
                # Concatenate into a unified target spectrum row
                y_combined = np.concatenate([log_E_kx, log_E_ky, log_E_zonal])
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.log_profiles.append(y_combined)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.log_profiles = np.array(self.log_profiles, dtype=np.float32)

if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    dataset = FourierSpectraEDADataset(DATA_DIR)
    
    print("\n" + "="*60)
    print(f"{'CHECK 2: PCA EXPLAINED VARIANCE ON COMBINED K-SPECTRA':^60}")
    print("="*60)
    max_modes = min(8, len(dataset.raw_c_values))
    pca = PCA(n_components=max_modes)
    pca.fit(dataset.log_profiles)
    
    cum_var = 0.0
    for idx, ratio in enumerate(pca.explained_variance_ratio_):
        cum_var += ratio * 100
        print(f" -> Mode {idx+1}: {ratio*100:.3f}% Variance | Cumulative: {cum_var:.3f}%")
    print("="*60)

    print("\n[EDA] Plotting Step 2: Multi-Spectrum Distribution (Cropped to m=1...64)...")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(dataset.raw_c_values)))
    modes_axis = np.arange(1, 171)
    
    for idx in range(len(dataset.raw_c_values)):
        lbl = f"C={dataset.raw_c_values[idx]:.3e}"
        # Unpack the three concatenated profiles
        y_kx = dataset.log_profiles[idx, 0:170]
        y_ky = dataset.log_profiles[idx, 170:340]
        y_zf = dataset.log_profiles[idx, 340:510]
        
        axs[0].plot(modes_axis, y_kx, color=colors[idx], alpha=0.7, lw=1.2)
        axs[1].plot(modes_axis, y_ky, color=colors[idx], alpha=0.7, lw=1.2)
        axs[2].plot(modes_axis, y_zf, color=colors[idx], alpha=0.7, lw=1.2, label=lbl)
        
    titles = [
        r"Radial Energy Spectrum: $\log_{10} E(k_x)$",
        r"Poloidal Energy Spectrum: $\log_{10} E(k_y)$",
        r"Zonal-Flow Energy Spectrum: $\log_{10} E_{ZF}(k_x)$"
    ]
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], fontsize=11)
        ax.set_xlabel(r"Mode number $m$")
        ax.set_ylabel(r"Kinetic energy, $\log_{10} E$")
        ax.set_xlim(1, 64) # Crop display to informative range
        ax.grid(True, alpha=0.25, linestyle=':')
        
    axs[2].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, fontsize=8)
    plt.tight_layout()
    
    output_filename = "step2_eda_spectra_smoothness.png"
    plt.savefig(output_filename, dpi=200)
    print(f"[Success] Multi-spectrum plot exported to '{output_filename}'.")
