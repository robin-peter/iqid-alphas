import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import shutil
import numpy as np # For creating dummy image data if not mocking fully

# Adjust import path
try:
    from iqid_alphas.pipelines.simple import SimplePipeline
    from iqid_alphas.core.raw_image_splitter import RawImageSplitter # For type hinting if needed by mock
    from iqid_alphas.core.alignment import ImageAligner # For type hinting if needed by mock
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from iqid_alphas.pipelines.simple import SimplePipeline
    from iqid_alphas.core.raw_image_splitter import RawImageSplitter
    from iqid_alphas.core.alignment import ImageAligner


class TestSimplePipeline(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_base_dir = Path("temp_test_simple_pipeline_data")
        self.sample_base_dir = self.test_base_dir / "sample_dirs"
        self.output_base_dir = self.test_base_dir / "output_dirs"

        if self.test_base_dir.exists():
            shutil.rmtree(self.test_base_dir)

        self.sample_base_dir.mkdir(parents=True, exist_ok=True)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy image file that can be "read" by a mocked skimage.io.imread
        self.dummy_slice_content = np.array([[1,2],[3,4]], dtype=np.uint8)


    def tearDown(self):
        """Clean up test environment."""
        if self.test_base_dir.exists():
            shutil.rmtree(self.test_base_dir)

    @patch('skimage.io.imsave') # Mock actual saving of images
    @patch('skimage.io.imread') # Mock actual reading of images
    @patch('iqid_alphas.pipelines.simple.ImageAligner')
    @patch('iqid_alphas.pipelines.simple.RawImageSplitter')
    def test_process_iqid_stack_raw_stage(self, MockRawSplitter, MockAligner, mock_imread, mock_imsave):
        """Test processing a sample from the 'raw' stage."""
        # Setup mocks
        mock_raw_splitter_instance = MockRawSplitter.return_value
        mock_aligner_instance = MockAligner.return_value

        # Mock RawImageSplitter().split_image()
        dummy_segmented_paths_str = [
            str(self.output_base_dir / "sample_raw_out" / "sample_raw_temp_segmented" / "slice_0.tif"),
            str(self.output_base_dir / "sample_raw_out" / "sample_raw_temp_segmented" / "slice_1.tif")
        ]
        mock_raw_splitter_instance.split_image.return_value = dummy_segmented_paths_str

        # Mock ImageAligner().align_images()
        # Let's say the "aligned" data is just a slight modification or the same
        mock_aligner_instance.align_images.return_value = (self.dummy_slice_content + 5, {'shift': [1,1]})

        # Mock skimage.io.imread to return consistent dummy data for all reads
        mock_imread.return_value = self.dummy_slice_content

        # Create sample directory structure for 'raw' stage
        sample_dir = self.sample_base_dir / "sample_raw"
        raw_data_dir = sample_dir / "Raw"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy raw file (content doesn't matter as split_image is mocked)
        with open(raw_data_dir / "raw_image.tif", "w") as f:
            f.write("dummy raw tiff content")

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_raw_out"

        results = pipeline.process_iqid_stack(str(sample_dir), str(output_dir))

        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['processed_stage'], 'raw')
        self.assertTrue(results['message'].startswith("Successfully processed"))
        self.assertIsNotNone(results['output_location'])

        mock_raw_splitter_instance.split_image.assert_called_once_with(
            str(raw_data_dir / "raw_image.tif"),
            str(output_dir / f"{sample_dir.name}_temp_segmented")
        )

        # Check alignment calls: one for the first slice (imread), then one align_images call for the second slice.
        self.assertGreaterEqual(mock_imread.call_count, 2) # At least 2 reads (ref + 1 moving)
        mock_aligner_instance.align_images.assert_called_once() # Only one actual alignment call for 2 slices

        # Check that aligned files were "saved"
        # First slice is saved directly, then subsequent ones are aligned and saved.
        expected_aligned_dir = output_dir / f"{sample_dir.name}_aligned_output"
        self.assertEqual(mock_imsave.call_count, len(dummy_segmented_paths_str))
        # Use unittest.mock.ANY for the numpy array data to avoid direct array comparison issues
        mock_imsave.assert_any_call(str(expected_aligned_dir / "aligned_0.tif"), unittest.mock.ANY, plugin='tifffile', check_contrast=False)
        mock_imsave.assert_any_call(str(expected_aligned_dir / "aligned_1.tif"), unittest.mock.ANY, plugin='tifffile', check_contrast=False)

        # Verify the actual content of the saved files by checking the arguments mock_imsave was called with.
        # This is more robust than relying on the order of assert_any_call for content.
        # Call 1: aligned_0.tif
        args_call_0, _ = mock_imsave.call_args_list[0]
        self.assertEqual(args_call_0[0], str(expected_aligned_dir / "aligned_0.tif"))
        np.testing.assert_array_equal(args_call_0[1], self.dummy_slice_content)

        # Call 2: aligned_1.tif
        args_call_1, _ = mock_imsave.call_args_list[1]
        self.assertEqual(args_call_1[0], str(expected_aligned_dir / "aligned_1.tif"))
        np.testing.assert_array_equal(args_call_1[1], self.dummy_slice_content + 5)


        # Check if temp directory was created and then removed (tricky to check removal without more advanced mocks or checks)
        # For now, we assume shutil.rmtree works if no error.
        self.assertFalse((output_dir / f"{sample_dir.name}_temp_segmented").exists(), "Temporary segmented directory should have been removed.")


    @patch('skimage.io.imsave')
    @patch('skimage.io.imread')
    @patch('iqid_alphas.pipelines.simple.ImageAligner')
    @patch('iqid_alphas.pipelines.simple.RawImageSplitter') # Still need to mock this even if not called
    def test_process_iqid_stack_segmented_stage(self, MockRawSplitter, MockAligner, mock_imread, mock_imsave):
        """Test processing a sample from the 'segmented' stage."""
        mock_raw_splitter_instance = MockRawSplitter.return_value
        mock_aligner_instance = MockAligner.return_value
        mock_aligner_instance.align_images.return_value = (self.dummy_slice_content + 10, {'shift': [1,1]})
        mock_imread.return_value = self.dummy_slice_content

        sample_dir = self.sample_base_dir / "sample_segmented"
        segmented_data_dir = sample_dir / "1_segmented"
        segmented_data_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy segmented files
        num_segmented_slices = 3
        for i in range(num_segmented_slices):
            with open(segmented_data_dir / f"segment_{i}.tif", "w") as f:
                f.write(f"dummy segmented tiff {i}")

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_segmented_out"

        results = pipeline.process_iqid_stack(str(sample_dir), str(output_dir))

        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['processed_stage'], 'segmented')
        self.assertTrue(results['message'].startswith("Successfully processed"))

        mock_raw_splitter_instance.split_image.assert_not_called()

        self.assertGreaterEqual(mock_imread.call_count, num_segmented_slices)
        self.assertEqual(mock_aligner_instance.align_images.call_count, num_segmented_slices - 1)

        expected_aligned_dir = output_dir / f"{sample_dir.name}_aligned_output"
        self.assertEqual(mock_imsave.call_count, num_segmented_slices)
        # Use unittest.mock.ANY for the numpy array data
        mock_imsave.assert_any_call(str(expected_aligned_dir / "aligned_0.tif"), unittest.mock.ANY, plugin='tifffile', check_contrast=False)
        mock_imsave.assert_any_call(str(expected_aligned_dir / "aligned_1.tif"), unittest.mock.ANY, plugin='tifffile', check_contrast=False)
        mock_imsave.assert_any_call(str(expected_aligned_dir / "aligned_2.tif"), unittest.mock.ANY, plugin='tifffile', check_contrast=False)

        # Verify actual content for specific calls
        call_args_list = mock_imsave.call_args_list
        # aligned_0.tif
        args_call_0, _ = call_args_list[0]
        self.assertEqual(args_call_0[0], str(expected_aligned_dir / "aligned_0.tif"))
        np.testing.assert_array_equal(args_call_0[1], self.dummy_slice_content)

        # aligned_1.tif
        args_call_1, _ = call_args_list[1]
        self.assertEqual(args_call_1[0], str(expected_aligned_dir / "aligned_1.tif"))
        np.testing.assert_array_equal(args_call_1[1], self.dummy_slice_content + 10)

        # aligned_2.tif
        args_call_2, _ = call_args_list[2]
        self.assertEqual(args_call_2[0], str(expected_aligned_dir / "aligned_2.tif"))
        np.testing.assert_array_equal(args_call_2[1], self.dummy_slice_content + 10)


    @patch('skimage.io.imsave') # Mock imsave as it might be called by other parts
    @patch('iqid_alphas.pipelines.simple.ImageAligner')
    @patch('iqid_alphas.pipelines.simple.RawImageSplitter')
    def test_process_iqid_stack_aligned_stage(self, MockRawSplitter, MockAligner, mock_imsave):
        """Test processing a sample that is already 'aligned'."""
        mock_raw_splitter_instance = MockRawSplitter.return_value
        mock_aligner_instance = MockAligner.return_value

        sample_dir = self.sample_base_dir / "sample_aligned"
        aligned_data_dir = sample_dir / "2_aligned"
        aligned_data_dir.mkdir(parents=True, exist_ok=True)
        with open(aligned_data_dir / "already_aligned_0.tif", "w") as f:
            f.write("dummy aligned tiff")

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_aligned_out" # Should not be used much

        results = pipeline.process_iqid_stack(str(sample_dir), str(output_dir))

        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['processed_stage'], 'aligned')
        self.assertTrue("already aligned" in results['message'].lower())
        self.assertEqual(results['output_location'], str(aligned_data_dir))


        mock_raw_splitter_instance.split_image.assert_not_called()
        mock_aligner_instance.align_images.assert_not_called()
        # mock_imsave might be called if the pipeline still tries to write something,
        # but ideally, for an already aligned sample, it shouldn't process/write new files.
        # Based on current code, it returns early, so imsave isn't called by this path.
        mock_imsave.assert_not_called()


    def test_process_iqid_stack_no_data(self):
        """Test processing an empty or invalid sample directory."""
        sample_dir = self.sample_base_dir / "sample_empty"
        sample_dir.mkdir(parents=True, exist_ok=True) # Empty directory

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_empty_out"

        with self.assertRaisesRegex(ValueError, "No processable data"):
            pipeline.process_iqid_stack(str(sample_dir), str(output_dir))

    def test_process_iqid_stack_raw_dir_no_files(self):
        """Test raw directory exists but no files."""
        sample_dir = self.sample_base_dir / "sample_raw_empty_subdir"
        (sample_dir / "Raw").mkdir(parents=True, exist_ok=True)

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_raw_empty_subdir_out"
        with self.assertRaisesRegex(ValueError, "Raw directory .* contains no TIFF files"):
            pipeline.process_iqid_stack(str(sample_dir), str(output_dir))

    def test_process_iqid_stack_segmented_dir_no_files(self):
        """Test 1_segmented directory exists but no files."""
        sample_dir = self.sample_base_dir / "sample_seg_empty_subdir"
        (sample_dir / "1_segmented").mkdir(parents=True, exist_ok=True)

        pipeline = SimplePipeline()
        output_dir = self.output_base_dir / "sample_seg_empty_subdir_out"
        with self.assertRaisesRegex(ValueError, "Segmented directory .* contains no TIFF files"):
            pipeline.process_iqid_stack(str(sample_dir), str(output_dir))


if __name__ == '__main__':
    unittest.main()
