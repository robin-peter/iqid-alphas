import os
import numpy as np
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize

def process_dose_kernels(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir):
    # Load and convert dose kernels
    dose_kernel = load_txt_kernel(kernel_file, dim, num_alpha_decays)
    mgy_kernel = mev_to_mgy(dose_kernel, vox_vol_m, dens_kgm)
    
    # Radial averaging and padding of kernels
    avg_kernel = radial_avg_kernel(mgy_kernel, mode="whole", bin_size=0.5)
    padded_kernel = pad_kernel_to_vsize(avg_kernel, vox_xy, slice_z)
    
    # Save the processed dose kernels
    save_processed_kernels(padded_kernel, output_dir)

def save_processed_kernels(kernel, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'processed_kernel.npy'), kernel)

def main(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir):
    process_dose_kernels(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir)

if __name__ == "__main__":
    kernel_file = "path/to/your/kernel_file.txt"
    dim = 128
    num_alpha_decays = 1e6
    vox_vol_m = 1e-9
    dens_kgm = 1e3
    vox_xy = 1
    slice_z = 12
    output_dir = "path/to/save/processed_kernels"
    main(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir)
