import numpy as np
import skimage.io
from pathlib import Path
import json

# Create dummy raw image
raw_height, raw_width = 90, 90
grid_rows, grid_cols = 3, 3
original_image_data = np.arange(
    raw_height * raw_width, dtype=np.uint16
).reshape(raw_height, raw_width)

# Path for the sample directory itself, not the "Raw" subdirectory yet
sample_root_dir = Path("temp_int_data/ReUpload_SampleProcess")
raw_subdir = sample_root_dir / "Raw" # Create "Raw" subdirectory
raw_subdir.mkdir(parents=True, exist_ok=True)
sample_raw_tiff_path = raw_subdir / "raw_multislice.tif" # Place raw file inside "Raw"
skimage.io.imsave(str(sample_raw_tiff_path), original_image_data, plugin='tifffile', check_contrast=False)
print(f"Created dummy raw TIFF at {sample_raw_tiff_path} within sample directory {sample_root_dir}")

# Create dummy config
config_data = {
    "processing": {"normalize": True},
    "segmentation": {"method": "otsu"},
    "alignment": {"method": "phase_correlation"},
    "visualization": {"save_plots": False}
}
config_path = Path("dummy_cli_config.json")
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)
print(f"Created dummy config at {config_path}")
