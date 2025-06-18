import numpy as np
import skimage.io
from pathlib import Path
import json
import shutil

# --- Configuration for dummy data ---
output_base_dir = Path("temp_reupload_dataset")
if output_base_dir.exists():
    shutil.rmtree(output_base_dir) # Clean slate

# --- Sample 1: Full Workflow (sample_full) ---
sample_full_dir = output_base_dir / "sample_full"
raw_full_dir = sample_full_dir / "Raw"
raw_full_dir.mkdir(parents=True, exist_ok=True)

raw_h, raw_w = 90, 90 # For 3x3 grid of 30x30 slices
raw_image_data = np.arange(raw_h * raw_w, dtype=np.uint16).reshape(raw_h, raw_w)
skimage.io.imsave(str(raw_full_dir / "raw_image.tif"), raw_image_data, plugin='tifffile', check_contrast=False)
print(f"Created: {raw_full_dir / 'raw_image.tif'}")

# --- Sample 2: Starts from Segmented (sample_segmented_start) ---
sample_seg_dir = output_base_dir / "sample_segmented_start"
seg_start_dir = sample_seg_dir / "1_segmented"
seg_start_dir.mkdir(parents=True, exist_ok=True)

slice_dim = 30
for i in range(3): # Create 3 dummy segmented slices
    seg_slice_data = np.full((slice_dim, slice_dim), (i+1) * 10, dtype=np.uint16)
    skimage.io.imsave(str(seg_start_dir / f"slice_{i}.tif"), seg_slice_data, plugin='tifffile', check_contrast=False)
    print(f"Created: {seg_start_dir / f'slice_{i}.tif'}")

# --- Sample 3: Already Aligned (sample_aligned_start) ---
sample_aligned_dir = output_base_dir / "sample_aligned_start"
aligned_start_dir = sample_aligned_dir / "2_aligned"
aligned_start_dir.mkdir(parents=True, exist_ok=True)

for i in range(2): # Create 2 dummy aligned slices
    aligned_slice_data = np.full((slice_dim, slice_dim), (i+1) * 50, dtype=np.uint16)
    skimage.io.imsave(str(aligned_start_dir / f"aligned_{i}.tif"), aligned_slice_data, plugin='tifffile', check_contrast=False)
    print(f"Created: {aligned_start_dir / f'aligned_{i}.tif'}")

# --- Dummy Config File ---
config_data = {
    "comment": "Dummy config for E2E testing",
    "processing": {"normalize": False},
    "segmentation": {"method": "otsu"},
    "alignment": {"method": "phase_correlation"},
    "visualization": {"save_plots": False}
}
config_path = Path("dummy_e2e_config.json")
with open(config_path, 'w') as f:
    json.dump(config_data, f, indent=2)
print(f"Created dummy config: {config_path}")

print("Mock dataset creation complete.")
