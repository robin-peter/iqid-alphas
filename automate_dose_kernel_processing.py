import os
import json
import logging
import numpy as np
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize

# Configure logging
logging.basicConfig(filename='automate_dose_kernel_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s'))

def process_dose_kernels(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir):
    try:
        # Load and convert dose kernels
        dose_kernel = load_txt_kernel(kernel_file, dim, num_alpha_decays)
        mgy_kernel = mev_to_mgy(dose_kernel, vox_vol_m, dens_kgm)
        
        # Radial averaging and padding of kernels
        avg_kernel = radial_avg_kernel(mgy_kernel, mode="whole", bin_size=0.5)
        padded_kernel = pad_kernel_to_vsize(avg_kernel, vox_xy, slice_z)
        
        # Save the processed dose kernels
        save_processed_kernels(padded_kernel, output_dir)
    except Exception as e:
        logging.error("Failed to process dose kernels: %s", str(e))
        raise

def save_processed_kernels(kernel, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'processed_kernel.npy'), kernel)
    except Exception as e:
        logging.error("Failed to save processed kernels: %s", str(e))
        raise

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        kernel_file = config['automate_dose_kernel_processing']['kernel_file']
        dim = config['automate_dose_kernel_processing']['dim']
        num_alpha_decays = config['automate_dose_kernel_processing']['num_alpha_decays']
        vox_vol_m = config['automate_dose_kernel_processing']['vox_vol_m']
        dens_kgm = config['automate_dose_kernel_processing']['dens_kgm']
        vox_xy = config['automate_dose_kernel_processing']['vox_xy']
        slice_z = config['automate_dose_kernel_processing']['slice_z']
        output_dir = config['automate_dose_kernel_processing']['output_dir']

        process_dose_kernels(kernel_file, dim, num_alpha_decays, vox_vol_m, dens_kgm, vox_xy, slice_z, output_dir)
    except Exception as e:
        logging.error("Failed to complete main dose kernel processing: %s", str(e))
        raise

if __name__ == "__main__":
    main()
