import numpy as np
import skimage.io
from pathlib import Path
import json
import shutil

print("--- Setting up mock datasets for final validation ---")

# --- Configuration for ReUpload dummy data ---
reupload_base_dir = Path("temp_reupload_dataset")
if reupload_base_dir.exists():
    shutil.rmtree(reupload_base_dir)
reupload_base_dir.mkdir(parents=True, exist_ok=True)

# Sample 1: Full Workflow (sample_full)
sample_full_dir = reupload_base_dir / "sample_full"
raw_full_dir = sample_full_dir / "Raw"
raw_full_dir.mkdir(parents=True, exist_ok=True)
raw_h, raw_w = 90, 90
raw_image_data = np.arange(raw_h * raw_w, dtype=np.uint16).reshape(raw_h, raw_w)
skimage.io.imsave(str(raw_full_dir / "raw_image.tif"), raw_image_data, plugin='tifffile', check_contrast=False)
print(f"Created: {raw_full_dir / 'raw_image.tif'}")

# Sample 2: Starts from Segmented (sample_segmented_start)
sample_seg_dir = reupload_base_dir / "sample_segmented_start"
seg_start_dir = sample_seg_dir / "1_segmented"
seg_start_dir.mkdir(parents=True, exist_ok=True)
slice_dim = 30
for i in range(3):
    seg_slice_data = np.full((slice_dim, slice_dim), (i+1) * 10, dtype=np.uint16)
    skimage.io.imsave(str(seg_start_dir / f"slice_{i}.tif"), seg_slice_data, plugin='tifffile', check_contrast=False)
print(f"Created segmented slices in: {seg_start_dir}")

# Sample 3: Already Aligned (sample_aligned_start)
sample_aligned_dir = reupload_base_dir / "sample_aligned_start"
aligned_start_dir = sample_aligned_dir / "2_aligned"
aligned_start_dir.mkdir(parents=True, exist_ok=True)
for i in range(2):
    aligned_slice_data = np.full((slice_dim, slice_dim), (i+1) * 50, dtype=np.uint16)
    skimage.io.imsave(str(aligned_start_dir / f"aligned_{i}.tif"), aligned_slice_data, plugin='tifffile', check_contrast=False)
print(f"Created aligned slices in: {aligned_start_dir}")

# --- Configuration for DataPush1 dummy data ---
datapush1_base_dir = Path("temp_datapush1_dataset")
if datapush1_base_dir.exists():
    shutil.rmtree(datapush1_base_dir)
datapush1_base_dir.mkdir(parents=True, exist_ok=True)

# Sample 1: iQID-only (D1M1(P1)_L_iqid_only)
sample1_iqid_dir = datapush1_base_dir / "iQID" / "3D" / "kidney" / "D1M1(P1)_L"
sample1_iqid_dir.mkdir(parents=True, exist_ok=True)
iqid_s1_data = np.full((slice_dim, slice_dim), 100, dtype=np.uint16)
skimage.io.imsave(str(sample1_iqid_dir / "mBq_corr_0.tif"), iqid_s1_data + 0, plugin='tifffile', check_contrast=False)
skimage.io.imsave(str(sample1_iqid_dir / "mBq_corr_1.tif"), iqid_s1_data + 1, plugin='tifffile', check_contrast=False)
print(f"Created: {sample1_iqid_dir / 'mBq_corr_1.tif'}")

# Sample 2: Paired iQID + H&E (D2M2(P1)_R_paired)
sample2_iqid_dir = datapush1_base_dir / "iQID" / "3D" / "tumor" / "D2M2(P1)_R"
sample2_iqid_dir.mkdir(parents=True, exist_ok=True)
iqid_s2_data = np.full((slice_dim, slice_dim), 200, dtype=np.uint16)
skimage.io.imsave(str(sample2_iqid_dir / "mBq_corr_0.tif"), iqid_s2_data + 0, plugin='tifffile', check_contrast=False)
skimage.io.imsave(str(sample2_iqid_dir / "mBq_corr_1.tif"), iqid_s2_data + 1, plugin='tifffile', check_contrast=False)
print(f"Created: {sample2_iqid_dir / 'mBq_corr_1.tif'}")

sample2_he_dir = datapush1_base_dir / "HE" / "3D" / "tumor" / "D2M2_R"
sample2_he_dir.mkdir(parents=True, exist_ok=True)
he_s2_data = np.full((slice_dim, slice_dim, 3), 128, dtype=np.uint8)
skimage.io.imsave(str(sample2_he_dir / "P1R.tif"), he_s2_data, plugin='tifffile', check_contrast=False)
skimage.io.imsave(str(sample2_he_dir / "P2R.tif"), he_s2_data + 10, plugin='tifffile', check_contrast=False)
print(f"Created: {sample2_he_dir / 'P2R.tif'}")

# --- Dummy Config Files ---
reupload_config_data = { "comment": "Dummy config for ReUpload E2E testing"}
dp1_config_data = {
    "comment": "Dummy config for DataPush1 E2E testing",
    "processing": {"normalize": True, "gaussian_blur_sigma": 0.5},
    "segmentation": {
        "method": "otsu", "min_tissue_area": 10, "min_activity_area": 5,
        "tissue_method": "adaptive", "activity_method": "otsu"
    },
    "alignment": {"method": "phase_correlation"},
    "output": {
        "save_aligned_images": True, "create_overlay_plots": False, # Disabled plots for less verbose output
        "generate_combined_report": False, "create_comprehensive_plots": False,
        "generate_report": False
    },
    "visualization": {"save_plots": False}
}

with open("dummy_e2e_config.json", 'w') as f: json.dump(reupload_config_data, f, indent=2)
print("Created: dummy_e2e_config.json")
with open("dummy_dp1_config.json", 'w') as f: json.dump(dp1_config_data, f, indent=2)
print("Created: dummy_dp1_config.json")

print("--- Mock dataset and config creation complete. ---")
