import os
import numpy as np
import unittest
from automate_dose_kernel_processing import process_dose_kernels, save_processed_kernels
from iqid.dpk import load_txt_kernel, mev_to_mgy, radial_avg_kernel, pad_kernel_to_vsize

class TestAutomateDoseKernelProcessing(unittest.TestCase):

    def setUp(self):
        self.kernel_file = "path/to/your/kernel_file.txt"
        self.dim = 128
        self.num_alpha_decays = 1e6
        self.vox_vol_m = 1e-9
        self.dens_kgm = 1e3
        self.vox_xy = 1
        self.slice_z = 12
        self.output_dir = "path/to/save/processed_kernels"

    def test_process_dose_kernels(self):
        process_dose_kernels(self.kernel_file, self.dim, self.num_alpha_decays, self.vox_vol_m, self.dens_kgm, self.vox_xy, self.slice_z, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'processed_kernel.npy')))

    def test_save_processed_kernels(self):
        kernel = np.zeros((128, 128, 128))
        save_processed_kernels(kernel, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'processed_kernel.npy')))

    def test_load_txt_kernel(self):
        kernel = load_txt_kernel(self.kernel_file, self.dim, self.num_alpha_decays)
        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.shape, (self.dim, self.dim, self.dim))

    def test_mev_to_mgy(self):
        kernel = np.zeros((128, 128, 128))
        mgy_kernel = mev_to_mgy(kernel, self.vox_vol_m, self.dens_kgm)
        self.assertIsNotNone(mgy_kernel)

    def test_radial_avg_kernel(self):
        kernel = np.zeros((128, 128, 128))
        avg_kernel = radial_avg_kernel(kernel, mode="whole", bin_size=0.5)
        self.assertIsNotNone(avg_kernel)

    def test_pad_kernel_to_vsize(self):
        kernel = np.zeros((128, 128, 128))
        padded_kernel = pad_kernel_to_vsize(kernel, self.vox_xy, self.slice_z)
        self.assertIsNotNone(padded_kernel)

if __name__ == '__main__':
    unittest.main()
