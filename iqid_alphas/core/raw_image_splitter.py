from pathlib import Path
import skimage.io
import numpy as np
import math

class RawImageSplitter:
    """
    Responsible for splitting raw multi-slice TIFFs based on a grid layout.
    """

    def __init__(self, grid_rows: int = 3, grid_cols: int = 3):
        """
        Initialize the splitter with grid dimensions.

        Parameters:
        ----------
        grid_rows : int, optional
            Number of rows in the grid of slices (default is 3).
        grid_cols : int, optional
            Number of columns in the grid of slices (default is 3).
        """
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("Grid dimensions must be positive integers.")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    def split_image(self, raw_tiff_path: str, output_dir: str) -> list[str]:
        """
        Loads a raw TIFF image, splits it into slices based on the grid,
        and saves each slice to the output directory.

        Parameters:
        ----------
        raw_tiff_path : str
            Path to the raw multi-slice TIFF file.
        output_dir : str
            Directory where the individual slice TIFF files will be saved.

        Returns:
        -------
        list[str]
            A list of absolute paths to the saved slice files.

        Raises:
        ------
        FileNotFoundError
            If the raw_tiff_path does not exist.
        ValueError
            If the image dimensions are not divisible by the grid dimensions.
        """
        raw_file = Path(raw_tiff_path)
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw TIFF file not found: {raw_tiff_path}")

        image = skimage.io.imread(raw_tiff_path)

        if image.ndim != 2:
            # Attempt to handle common cases like (H, W, 1) or (1, H, W)
            if image.ndim == 3 and image.shape[2] == 1: # (H, W, 1)
                image = image.squeeze(axis=2)
            elif image.ndim == 3 and image.shape[0] == 1: # (1, H, W)
                image = image.squeeze(axis=0)
            else:
                raise ValueError(
                    f"Expected a 2D image or a 3D image reducible to 2D (e.g. HxWx1), "
                    f"but got shape {image.shape}. The document implies a single large 2D image plane "
                    f"containing a grid of slices."
                )

        total_height, total_width = image.shape

        if total_height % self.grid_rows != 0 or total_width % self.grid_cols != 0:
            raise ValueError(
                f"Image dimensions ({total_height}x{total_width}) are not perfectly "
                f"divisible by grid dimensions ({self.grid_rows}x{self.grid_cols})."
            )

        slice_height = total_height // self.grid_rows
        slice_width = total_width // self.grid_cols

        output_paths = []
        output_directory = Path(output_dir)
        output_directory.mkdir(parents=True, exist_ok=True)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                start_row = r * slice_height
                end_row = (r + 1) * slice_height
                start_col = c * slice_width
                end_col = (c + 1) * slice_width

                slice_data = image[start_row:end_row, start_col:end_col]

                slice_index = r * self.grid_cols + c
                slice_filename = f"slice_{slice_index}.tif"
                full_output_path = output_directory / slice_filename

                skimage.io.imsave(str(full_output_path), slice_data, plugin='tifffile', check_contrast=False)
                output_paths.append(str(full_output_path))

        return output_paths

if __name__ == '__main__':
    # Example Usage (Optional: for quick testing if run directly)
    # Create a dummy image
    dummy_rows, dummy_cols = 90, 90
    grid_r, grid_c = 3, 3
    dummy_image_data = np.arange(dummy_rows * dummy_cols, dtype=np.uint16).reshape(dummy_rows, dummy_cols)
    dummy_raw_path = "temp_dummy_raw.tif"
    skimage.io.imsave(dummy_raw_path, dummy_image_data)

    splitter = RawImageSplitter(grid_rows=grid_r, grid_cols=grid_c)
    output_slice_dir = "temp_output_slices"

    try:
        paths = splitter.split_image(dummy_raw_path, output_slice_dir)
        print(f"Successfully saved {len(paths)} slices to {output_slice_dir}:")
        for p in paths:
            print(p)

        # Verify a slice
        loaded_slice_0 = skimage.io.imread(paths[0])
        print(f"Slice 0 shape: {loaded_slice_0.shape}")
        expected_slice_0_data = dummy_image_data[0:dummy_rows//grid_r, 0:dummy_cols//grid_c]
        if np.array_equal(loaded_slice_0, expected_slice_0_data):
            print("Slice 0 content verified.")
        else:
            print("Slice 0 content mismatch.")

    except Exception as e:
        print(f"Error during example usage: {e}")
    finally:
        # Clean up
        import shutil
        if Path(dummy_raw_path).exists():
            Path(dummy_raw_path).unlink()
        if Path(output_slice_dir).exists():
            shutil.rmtree(output_slice_dir)
        print("Cleaned up temporary files.")
