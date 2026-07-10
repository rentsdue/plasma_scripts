import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class FourierPowerEDADataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        self.log_profiles = []  # Log10 of time-mean power spectrum
        
        self._process_files()
        
    def _process_files(self):
        print("Extracting time-mean log-power spectra (dropping m=0)...")
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                c_val = f['params/C'][()]
                x_input = np.log10(c_val)
                
                uk = f['fields/uk']
                kx2d = f['data/kx'][()]
                
                T = uk.shape[0]
                window = range(T // 2, T)
                
                profiles_t = []
                for t in window:
                    potential_fft = uk[t, 0, :, :]
                    kx1d = kx2d[:, 0]
                    U_zf_fourier = 1j * kx1d * potential_fft[:, 0]
                    profiles_t.append(np.abs(U_zf_fourier))
                
                # 1. Compute time-mean power per mode
                P_bar = np.mean(np.square(profiles_t), axis=0)
                
                # 2. Drop m=0, keep one Hermitian half, and apply a tiny floor to protect dead modes from -inf
                y = np.log10(P_bar[1:171] + 1e-20)
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.log_profiles.append(y)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.log_profiles = np.array(self.log_profiles, dtype=np.float32)

if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    dataset = FourierPowerEDADataset(DATA_DIR)
    
    print("\n" + "="*60)
    print(f"{'CHECK 2: PCA EXPLAINED VARIANCE RATIO ON LOG-POWER':^60}")
    print("="*60)
    
    max_modes = min(8, len(dataset.raw_c_values))
    pca = PCA(n_components=max_modes)
    pca.fit(dataset.log_profiles)
    
    cum_var = 0.0
    for idx, ratio in enumerate(pca.explained_variance_ratio_):
        cum_var += ratio * 100
        print(f" -> Mode {idx+1}: {ratio*100:.3f}% Variance | Cumulative: {cum_var:.3f}%")
    print("="*60)

    print("\n[EDA] Plotting Check 1: Multi-Profile Parametric Smoothness (Cropped to m=1...64)...")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(dataset.raw_c_values)))
    
    # Define physical modes axis corresponding to indices 1 to 170
    modes_axis = np.arange(1, 171)
    
    for idx in range(len(dataset.raw_c_values)):
        lbl = f"C={dataset.raw_c_values[idx]:.3e}"
        plt.plot(modes_axis, dataset.log_profiles[idx], color=colors[idx], alpha=0.8, lw=1.5, label=lbl)
        
    plt.title("Step 1 Log-Power Smoothness Check (Core Wavenumbers)")
    plt.xlabel("Mode Index ($m$)")
    plt.ylabel(r"Zonal Flow Log-Power $\log_{10}(\overline{P}_m)$")
    plt.xlim(1, 64)  # Focus display strictly on informative modes
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
    plt.grid(True, alpha=0.25, linestyle=':')
    plt.tight_layout()
    
    output_filename = "step1_eda_smoothness_check_log.png"
    plt.savefig(output_filename, dpi=200)
    print(f"[Success] Smoothness plot exported to '{output_filename}'.")
