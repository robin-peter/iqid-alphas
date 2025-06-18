import unittest
from pathlib import Path
import shutil
import numpy as np
import skimage.io

# Adjust import path if RawImageSplitter is not directly in parent's path
# This assumes that the tests might be run from the repository root.
try:
    from iqid_alphas.core.raw_image_splitter import RawImageSplitter
except ImportError:
    # Fallback for local execution if iqid_alphas is not in PYTHONPATH
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Add root iqid_alphas to path
    from iqid_alphas.core.raw_image_splitter import RawImageSplitter

class TestRawImageSplitter(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("temp_test_splitter_data")
        self.raw_images_dir = self.test_dir / "raw_input"
        self.output_slices_dir = self.test_dir / "output_slices"

        # Clean up directories if they exist from a previous failed run
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_slices_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy raw TIFF image (90x90, 3x3 grid of 30x30 slices)
        self.grid_rows = 3
        self.grid_cols = 3
        self.slice_height = 30
        self.slice_width = 30
        self.raw_height = self.grid_rows * self.slice_height
        self.raw_width = self.grid_cols * self.slice_width

        self.original_image_data = np.arange(
            self.raw_height * self.raw_width, dtype=np.uint16
        ).reshape(self.raw_height, self.raw_width)

        self.sample_raw_tiff_path = self.raw_images_dir / "sample_raw.tif"
        skimage.io.imsave(str(self.sample_raw_tiff_path), self.original_image_data, plugin='tifffile', check_contrast=False)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_split_image_successfully(self):
        """Test splitting a valid raw TIFF image."""
        splitter = RawImageSplitter(grid_rows=self.grid_rows, grid_cols=self.grid_cols)

        # Define a specific output directory for this test run
        test_output_dir = self.output_slices_dir / "test_run1"

        saved_slice_paths = splitter.split_image(str(self.sample_raw_tiff_path), str(test_output_dir))

        self.assertEqual(len(saved_slice_paths), self.grid_rows * self.grid_cols, "Incorrect number of slices created.")

        expected_filenames = []
        for i in range(self.grid_rows * self.grid_cols):
            expected_filenames.append(f"slice_{i}.tif")

        actual_filenames = sorted([Path(p).name for p in saved_slice_paths])
        self.assertListEqual(sorted(expected_filenames), actual_filenames, "Slice filenames are not as expected.")

        # Verify content of a few slices
        # Slice 0 (top-left)
        slice_0_data = skimage.io.imread(str(test_output_dir / "slice_0.tif"))
        self.assertEqual(slice_0_data.shape, (self.slice_height, self.slice_width), "Slice 0 has incorrect dimensions.")
        expected_slice_0_content = self.original_image_data[0:self.slice_height, 0:self.slice_width]
        np.testing.assert_array_equal(slice_0_data, expected_slice_0_content, "Slice 0 content mismatch.")

        # Slice 4 (middle slice for a 3x3 grid)
        slice_4_data = skimage.io.imread(str(test_output_dir / "slice_4.tif"))
        r, c = 1, 1 # 0-indexed row and col for slice_4 (1*3 + 1 = 4)
        self.assertEqual(slice_4_data.shape, (self.slice_height, self.slice_width), "Slice 4 has incorrect dimensions.")
        expected_slice_4_content = self.original_image_data[
            r*self.slice_height:(r+1)*self.slice_height,
            c*self.slice_width:(c+1)*self.slice_width
        ]
        np.testing.assert_array_equal(slice_4_data, expected_slice_4_content, "Slice 4 content mismatch.")

        # Slice 8 (bottom-right for a 3x3 grid)
        slice_8_data = skimage.io.imread(str(test_output_dir / "slice_8.tif"))
        r, c = 2, 2 # 0-indexed row and col for slice_8 (2*3 + 2 = 8)
        self.assertEqual(slice_8_data.shape, (self.slice_height, self.slice_width), "Slice 8 has incorrect dimensions.")
        expected_slice_8_content = self.original_image_data[
            r*self.slice_height:(r+1)*self.slice_height,
            c*self.slice_width:(c+1)*self.slice_width
        ]
        np.testing.assert_array_equal(slice_8_data, expected_slice_8_content, "Slice 8 content mismatch.")

    def test_split_image_file_not_found(self):
        """Test splitting when the raw TIFF file does not exist."""
        splitter = RawImageSplitter()
        with self.assertRaises(FileNotFoundError):
            splitter.split_image("non_existent_raw_file.tif", str(self.output_slices_dir))

    def test_split_image_indivisible_dimensions(self):
        """Test splitting when image dimensions are not divisible by grid."""
        indivisible_raw_data = np.arange(91 * 90, dtype=np.uint16).reshape(91, 90)
        indivisible_raw_path = self.raw_images_dir / "indivisible_raw.tif"
        skimage.io.imsave(str(indivisible_raw_path), indivisible_raw_data, plugin='tifffile', check_contrast=False)

        splitter = RawImageSplitter(grid_rows=3, grid_cols=3)
        with self.assertRaisesRegex(ValueError, "not perfectly divisible by grid dimensions"):
            splitter.split_image(str(indivisible_raw_path), str(self.output_slices_dir))

    def test_init_invalid_grid_dimensions(self):
        """Test RawImageSplitter initialization with invalid grid dimensions."""
        with self.assertRaisesRegex(ValueError, "Grid dimensions must be positive integers."):
            RawImageSplitter(grid_rows=0, grid_cols=3)
        with self.assertRaisesRegex(ValueError, "Grid dimensions must be positive integers."):
            RawImageSplitter(grid_rows=3, grid_cols=-1)

    def test_split_image_non_2d_image(self):
        """Test splitting when input image is not effectively 2D."""
        # Create a 3D image that cannot be squeezed to 2D in the expected way
        # e.g. a true multi-channel image or z-stack not fitting the grid-on-plane model
        problematic_3d_data = np.arange(30 * 30 * 3, dtype=np.uint16).reshape(30, 30, 3)
        problematic_3d_path = self.raw_images_dir / "problematic_3d.tif"
        skimage.io.imsave(str(problematic_3d_path), problematic_3d_data, plugin='tifffile', check_contrast=False)

        splitter = RawImageSplitter(grid_rows=1, grid_cols=1) # Grid doesn't matter here
        with self.assertRaisesRegex(ValueError, "Expected a 2D image or a 3D image reducible to 2D"):
            splitter.split_image(str(problematic_3d_path), str(self.output_slices_dir))


if __name__ == '__main__':
    unittest.main()
