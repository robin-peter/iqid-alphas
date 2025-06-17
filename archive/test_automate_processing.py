import os
import numpy as np
import unittest
from automate_processing import load_and_process_listmode_data, filter_correct_analyze_data, generate_spatial_images, generate_temporal_information, detect_contours_extract_ROIs, save_processed_data

class TestAutomateProcessing(unittest.TestCase):

    def setUp(self):
        self.file_name = "path/to/your/listmode_data.dat"
        self.output_dir = "path/to/save/processed_data"
        self.cluster_data, self.time_ms, self.cluster_area, self.xC_global, self.yC_global, self.frame_num = load_and_process_listmode_data(self.file_name)

    def test_load_and_process_listmode_data(self):
        self.assertIsNotNone(self.cluster_data)
        self.assertIsNotNone(self.time_ms)
        self.assertIsNotNone(self.cluster_area)
        self.assertIsNotNone(self.xC_global)
        self.assertIsNotNone(self.yC_global)
        self.assertIsNotNone(self.frame_num)

    def test_filter_correct_analyze_data(self):
        binfac = 1
        ROI_area_thresh = 100
        t_binsize = 1000
        t_half = 3600
        cluster_data = filter_correct_analyze_data(self.cluster_data, binfac, ROI_area_thresh, t_binsize, t_half)
        self.assertIsNotNone(cluster_data)

    def test_generate_spatial_images(self):
        subpx = 1
        cluster_image = generate_spatial_images(self.cluster_data, subpx)
        self.assertIsNotNone(cluster_image)

    def test_generate_temporal_information(self):
        event_fx = 0.1
        xlim = (0, None)
        ylim = (0, None)
        temporal_image = generate_temporal_information(self.cluster_data, event_fx, xlim, ylim)
        self.assertIsNotNone(temporal_image)

    def test_detect_contours_extract_ROIs(self):
        im = np.zeros((100, 100))
        gauss = 15
        thresh = 0
        contours, ROIs = detect_contours_extract_ROIs(self.cluster_data, im, gauss, thresh)
        self.assertIsNotNone(contours)
        self.assertIsNotNone(ROIs)

    def test_save_processed_data(self):
        save_processed_data(self.cluster_data, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'xC.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'yC.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'f.npy')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'time_ms.npy')))

if __name__ == '__main__':
    unittest.main()
