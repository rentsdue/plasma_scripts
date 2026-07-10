import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Guarantee cross-system mathematical reproducibility
np.random.seed(0)

class Saturated2DMapEDADataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5') and f.startswith('hwak_C')])
        
        self.raw_c_values = []
        self.inputs = []
        self.flat_targets = []  # Combined long structural vector matrix (Size: 346,800 per run)
        
        # Unflattened arrays kept specifically for 1D structural smoothness projections
        self.spatial_rms_maps = []
        self.spatial_flux_maps = []
        self.spectrum_2d_maps = []
        
        self._process_files()
        
    def _process_files(self):
        print("Extracting time-mean 2D Spatial and Spectral target fields...")
        first_file = True
        
        for file_name in self.file_list:
            file_path = os.path.join(self.data_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                
                # -----------------------------------------------------------------------------
                # COMPREHENSIVE DEBUG BLOCK (Runs once on the first file)
                # -----------------------------------------------------------------------------
                if first_file:
                    print("\n" + "="*50)
                    print("         CRITICAL HDF5 FIELD DIAGNOSTIC        ")
                    print("="*50)
                    print("File being inspected: {}".format(file_name))
                    print("Root-level groups: {}".format(list(f.keys())))
                    
                    if 'fields' in f:
                        print("\nAvailable datasets inside 'fields':")
                        for k in f['fields'].keys():
                            ds = f['fields'][k]
                            print("  -> fields/{:<5} | Shape: {:<18} | Type: {}".format(k, str(ds.shape), ds.dtype))
                    
                    if 'data' in f:
                        print("\nAvailable datasets inside 'data':")
                        for k in f['data'].keys():
                            ds = f['data'][k]
                            sh = str(ds.shape) if hasattr(ds, 'shape') else 'Scalar'
                            dt = ds.dtype if hasattr(ds, 'dtype') else type(ds)
                            print("  -> data/{:<7} | Shape: {:<18} | Type: {}".format(k, sh, dt))
                    print("="*50 + "\n")
                    first_file = False
                # -----------------------------------------------------------------------------

                c_val = f['params/C'][()]
                kappa = f['params/kappa'][()] if 'params/kappa' in f else 1.0
                x_input = np.log10(c_val)
                
                # Inspect your debug printout to confirm if density sits at channel 1 of uk
                uk = f['fields/uk']
                kx2D = f['data/kx'][()]
                ky2D = f['data/ky'][()]
                
                T = uk.shape[0]
                window = range(T // 2, T)
                N_t = len(window)
                
                # Setup target accumulation grids at the proper 340x340 real-space resolution
                spatial_phi_sq_accum = np.zeros((340, 340))
                spatial_flux_accum = np.zeros((340, 340))
                spectrum_2d_accum = np.zeros((340, 340))
                
                for t in window:
                    # Pull the half-spectrum complex modes
                    phi_k = uk[t, 0, :, :]
                    
                    # Dynamically handle if density is channel 1 or a standalone key
                    if 'fields/nk' in f:
                        n_k = f['fields/nk'][t, 0, :, :]
                    else:
                        n_k = uk[t, 1, :, :] # Fallback to channel 1 of the state vector
                    
                    # Compute analytical poloidal gradient in half-Fourier space
                    grady_phi_k = 1j * ky2D * phi_k
                    
                    # INVERSION FIX: Use real-FFT inversion with explicit grid constraints
                    phi_space = np.fft.irfft2(phi_k, s=(340, 340), norm="forward")
                    n_space = np.fft.irfft2(n_k, s=(340, 340), norm="forward")
                    grady_phi_space = np.fft.irfft2(grady_phi_k, s=(340, 340), norm="forward")
                    
                    # Accumulate real-space physical distributions
                    spatial_phi_sq_accum += np.square(phi_space)
                    spatial_flux_accum += (-kappa * n_space * grady_phi_space)
                    
                    # SPECTRAL UNIFORMITY FIX: Take full FFT of real-space potential 
                    # This yields a full 340x340 spectral map matching your image channels
                    phi_full_k = np.fft.fft2(phi_space, norm="forward")
                    spectrum_2d_accum += np.square(np.abs(phi_full_k))
                
                # Compute temporal averages across the window
                mean_space_rms = np.sqrt(spatial_phi_sq_accum / N_t)
                mean_space_flux = spatial_flux_accum / N_t
                mean_spectrum_2d = spectrum_2d_accum / N_t
                
                # Apply micro-floored log10 scaling filters
                log_space_rms = np.log10(mean_space_rms + 1e-20)
                log_space_flux = np.log10(np.abs(mean_space_flux) + 1e-20)
                log_spectrum_2d = np.log10(mean_spectrum_2d + 1e-20)
                
                # Verify final array dimensions match perfectly
                if len(self.raw_c_values) == 0:
                    print("Processed First Grid Maps Successfully:")
                    print("  -> Spatial RMS shape: {}".format(log_space_rms.shape))
                    print("  -> Local Flux shape : {}".format(log_space_flux.shape))
                    print("  -> 2D Spectrum shape: {}".format(log_spectrum_2d.shape))
                
                # Save the raw unflattened 2D maps for the 1D projection checks
                self.spatial_rms_maps.append(log_space_rms)
                self.spatial_flux_maps.append(log_space_flux)
                self.spectrum_2d_maps.append(log_spectrum_2d)

                # Flatten and store into unified 346,800 feature signature rows
                combined_vector = np.concatenate([
                    log_space_rms.flatten(),
                    log_space_flux.flatten(),
                    log_spectrum_2d.flatten()
                ])
                
                self.raw_c_values.append(c_val)
                self.inputs.append([x_input])
                self.flat_targets.append(combined_vector)
                
        self.raw_c_values = np.array(self.raw_c_values, dtype=np.float32)
        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.flat_targets = np.array(self.flat_targets, dtype=np.float32)

if __name__ == "__main__":
    DATA_DIR = "/zhisongqu_data/ameir/guillon_dns_triad/scan_IIIA_512"
    dataset = Saturated2DMapEDADataset(DATA_DIR)
    
    # -----------------------------------------------------------------------------
    # EDA CHECK 1: POD/PCA Eigenvalue Spectrum on 2D Snapshot Flattened Matrix
    # -----------------------------------------------------------------------------
    print("\n" + "="*65)
    print("{:^65}".format("CHECK 1: PCA EXPLAINED VARIANCE ON FULL CONCATENATED 2D MAPS"))
    print("="*65)
    max_modes = min(8, len(dataset.raw_c_values))
    pca = PCA(n_components=max_modes)
    pca.fit(dataset.flat_targets)
    
    cum_var = 0.0
    for idx, ratio in enumerate(pca.explained_variance_ratio_):
        cum_var += ratio * 100
        print(f" -> Mode {idx+1:2d}: {ratio*100:6.3f}% Variance | Cumulative: {cum_var:6.3f}%")
    print("="*65)

    # -----------------------------------------------------------------------------
    # EDA CHECK 2: Generate 1D Structural Projections for Smoothness Check
    # -----------------------------------------------------------------------------
    print("\n[EDA] Compiling 1D spatial/spectral projections across the parametric sweep...")
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(dataset.raw_c_values)))
    
    # Fixed coordinate index grid axes
    spatial_x_axis = np.arange(340)
    spectral_kx_axis = np.arange(1, 65) # Cropped to energetic core modes
    
    for idx in range(len(dataset.raw_c_values)):
        lbl = f"C={dataset.raw_c_values[idx]:.3e}"
        
        # Projection 1: Spatial RMS Potential - Average over poloidal axis (y) to get radial profile (x)
        # Taking the mean across columns (axis 1)
        spatial_rms_profile = np.mean(dataset.spatial_rms_maps[idx], axis=1)
        
        # Projection 2: Spatial Local Flux - Average over poloidal axis (y) to get radial profile (x)
        spatial_flux_profile = np.mean(dataset.spatial_flux_maps[idx], axis=1)
        
        # Projection 3: 2D Spectrum Slice - Extract the central k_y = 0 line (the pure zonal column)
        # Slice from mode 1 to 64 to avoid singularity and capture the primary cascade
        zonal_spectral_slice = dataset.spectrum_2d_maps[idx][1:65, 0]
        
        # Plot Profiles
        axs[0].plot(spatial_x_axis, spatial_rms_profile, color=colors[idx], alpha=0.7, lw=1.2)
        axs[1].plot(spatial_x_axis, spatial_flux_profile, color=colors[idx], alpha=0.7, lw=1.2)
        axs[2].plot(spectral_kx_axis, zonal_spectral_slice, color=colors[idx], alpha=0.7, lw=1.2, label=lbl)
        
    # Configure Layout Aesthetics
    axs[0].set_title(r"Poloidal Mean Spatial $\log_{10}(\text{RMS}_{\phi})$", fontsize=11)
    axs[0].set_xlabel("Radial Coordinate Grid Box ($x$)")
    axs[0].set_ylabel("Log Intensity Scaling")
    axs[0].grid(True, alpha=0.25, linestyle=':')
    
    axs[1].set_title(r"Poloidal Mean Spatial $\log_{10}(|\Gamma_n|)$", fontsize=11)
    axs[1].set_xlabel("Radial Coordinate Grid Box ($x$)")
    axs[1].set_ylabel("Log Flux Scaling")
    axs[1].grid(True, alpha=0.25, linestyle=':')
    
    axs[2].set_title(r"2D Spectral Cut $\log_{10}(P_{\phi})$ at $k_y=0$", fontsize=11)
    axs[2].set_xlabel("Wavenumber Mode Index ($k_x$)")
    axs[2].set_ylabel("Log Power Scaling")
    axs[2].set_xlim(1, 64)
    axs[2].grid(True, alpha=0.25, linestyle=':')
    
    axs[2].legend(bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, fontsize=8)
    plt.tight_layout()
    
    output_filename = "step3_eda_2d_fields_smoothness.png"
    plt.savefig(output_filename, dpi=200)
    print(f"[Success] 2D Field projection diagnostic plot exported to '{output_filename}'.")
