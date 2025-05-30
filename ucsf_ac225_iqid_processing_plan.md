# UCSF Ac-225 Mouse Experiment iQID Dataset Processing Plan

This document outlines the automated processing pipeline for the UCSF Ac-225 mouse experiment iQID dataset. It covers input file structure, configuration management, the logic of the three main processing scripts, and considerations for robustness and scalability.

## 1. Input File Structure Overview

The successful execution of the processing pipeline relies on a specific input file structure. This structure is based on the analysis performed in Step 1 of our plan.

*   **Root Data Directory:** A main directory per experiment or dataset.
    *   Example: `/data/ucsf_ac225_exp1/`
*   **Raw iQID Images:**
    *   Stored in a subdirectory, typically named `raw_iqid_images/` or `acquisitions/`.
    *   Image files are expected in a standard format (e.g., TIFF, DICOM) and may be organized further by timepoint or animal ID.
    *   Example: `/data/ucsf_ac225_exp1/raw_iqid_images/mouse01_day03_slice001.tif`
*   **Quantitative Notes Document:**
    *   A `.docx` file named `quantitative_notes.docx` located in the root data directory or a dedicated `metadata/` subdirectory.
    *   This file contains tables or structured text detailing quantitative correction factors, decay times, scanner calibration factors, etc., necessary for `automate_processing.py`.
    *   Example: `/data/ucsf_ac225_exp1/quantitative_notes.docx`
*   **Dose Kernel File:**
    *   A text file (`.txt`) containing the dose kernel information, typically provided by physics simulations.
    *   Expected to be in the root data directory or a `kernels/` subdirectory.
    *   Example: `/data/ucsf_ac225_exp1/kernels/ac225_water_10keV_kernel.txt`
*   **Output Directories:**
    *   The `config.json` will specify paths for various outputs (processed data, aligned images, dose maps, logs). It's recommended to have a main `output/` directory in the root data directory, with subdirectories for each processing stage.
    *   Example: `/data/ucsf_ac225_exp1/output/processed_data/`, `/data/ucsf_ac225_exp1/output/aligned_images/`

## 2. `config.json` Configuration

Based on Step 2 of our plan, the `config.json` file centralizes all parameters for the processing pipeline.

**Structure and Parameters:**

```json
{
  "experiment_id": "ucsf_ac225_exp1",
  "paths": {
    "root_data_dir": "/data/ucsf_ac225_exp1/",
    "raw_image_dir": "raw_iqid_images/", // Relative to root_data_dir
    "quantitative_notes_doc": "quantitative_notes.docx", // Relative to root_data_dir
    "dose_kernel_file": "kernels/ac225_water_10keV_kernel.txt", // Relative to root_data_dir
    "output_base_dir": "output/", // Relative to root_data_dir
    "processed_data_dir": "output/processed_data/", // Can be absolute or relative
    "aligned_images_dir": "output/aligned_images/",
    "dose_maps_dir": "output/dose_maps/",
    "log_dir": "output/logs/"
  },
  "quantitative_corrections": {
    "decay_correction_enabled": true,
    "half_life_seconds": 864000, // Example for Ac-225 (10 days)
    "scanner_calibration_factor": 1.25,
    "unit_conversion_factor": 0.01 // e.g., counts to Bq/pixel
  },
  "automate_processing_params": {
    "roi_extraction_method": "threshold", // "threshold" or "manual_coordinates"
    "threshold_value": 100, // If method is "threshold"
    "minimum_roi_area_pixels": 50
  },
  "automate_image_alignment_params": {
    "alignment_type": "sequential", // "3d_stack" or "sequential"
    "reference_slice_index": 0, // For sequential if applicable
    "coarse_alignment_factor": 4,
    "padding_value_he": 0, // For H&E staining alignment if used
    "crop_margins": [10, 10, 10, 10] // top, bottom, left, right in pixels
  },
  "automate_dose_kernel_processing_params": {
    "kernel_energy_mev": 0.150, // Energy of particles in kernel file (MeV)
    "mev_to_mgy_conversion_factor": 1.602e-10, // MeV/g to mGy
    "target_voxel_size_mm": [0.5, 0.5, 0.5] // Desired voxel size for dose map
  },
  "logging": {
    "log_level": "INFO", // DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_to_file": true,
    "log_filename_template": "processing_log_{script_name}_{timestamp}.log"
  },
  "parallelization": {
    "enabled": true,
    "max_workers_processing": 4, // For automate_processing
    "max_workers_alignment": 2  // For automate_image_alignment
  }
}
```

**Key Aspects:**
*   **Paths:** Clearly defined input and output paths. Using relative paths from `root_data_dir` can enhance portability.
*   **Quantitative Corrections:** Parameters for decay correction, calibration, and unit conversions.
*   **Processing Parameters:** Script-specific settings (e.g., ROI extraction method for `automate_processing.py`).
*   **Logging:** Configuration for log level and file output.
*   **Parallelization:** Flags and limits for concurrent processing.

## 3. `automate_processing.py` Logic

This script, based on Step 3 of our plan, handles the initial processing of raw iQID images, including metadata extraction, quantitative corrections, and ROI identification.

**Core Logic:**

1.  **Initialization:**
    *   Load `config.json`.
    *   Setup logging as per configuration.
    *   Identify raw image files to be processed from `paths.raw_image_dir`.

2.  **Metadata Extraction:**
    *   For each raw image:
        *   Extract acquisition time, pixel spacing, slice thickness, and other relevant metadata from image headers (e.g., TIFF tags, DICOM headers).
        *   If essential metadata is missing, log a warning/error and potentially skip the file.

3.  **Quantitative Corrections (from `quantitative_notes.docx` and `config.json`):**
    *   **Parse `quantitative_notes.docx`:**
        *   Use a Python library (e.g., `python-docx`) to read tables or structured text from the document.
        *   Extract animal-specific or timepoint-specific correction factors, injection times, measured activities, etc. This requires a predefined structure in the `.docx` file.
    *   **Apply Corrections (for each image/slice):**
        *   **Decay Correction:** If `quantitative_corrections.decay_correction_enabled` is true, calculate the time difference between injection (from `.docx` or metadata) and image acquisition. Apply decay formula: `Activity_corrected = Activity_initial * exp(-lambda * time_elapsed)`. Lambda is calculated from `half_life_seconds`.
        *   **Scanner Calibration:** Apply `quantitative_corrections.scanner_calibration_factor`.
        *   **Unit Conversion:** Apply `quantitative_corrections.unit_conversion_factor` to convert pixel values to meaningful physical units (e.g., Bq/mL or similar).
        *   Store these correction factors and the process in logs.

4.  **ROI Identification (using `get_contours` / `get_ROIs`):**
    *   Based on `automate_processing_params.roi_extraction_method`:
        *   **`threshold`:** Apply an intensity threshold (`automate_processing_params.threshold_value`) to create a binary mask. Use `cv2.findContours` (wrapped in a utility function like `get_contours`) to identify potential ROIs. Filter ROIs by area (`automate_processing_params.minimum_roi_area_pixels`).
        *   **`manual_coordinates`:** (If implemented) Load predefined ROI coordinates from a file or the `quantitative_notes.docx`.
    *   The `get_ROIs` function would encapsulate this logic, returning a list of ROI objects or masks.

5.  **Data Saving:**
    *   For each processed image/slice and its ROIs:
        *   Save the quantitatively corrected image data (e.g., as NIfTI or NumPy array) to `paths.processed_data_dir`.
        *   Save ROI masks or coordinates in a suitable format (e.g., JSON, NPY, or overlaid on saved images).
        *   Store extracted metadata and calculated values (e.g., total activity per ROI) in a structured file (e.g., CSV or JSON) linking back to the image filenames.
    *   Filename conventions should be consistent and include identifiers (animal ID, timepoint).

6.  **Parallelization:**
    *   If `parallelization.enabled` is true, process multiple images/animals in parallel using `concurrent.futures.ProcessPoolExecutor` with `max_workers_processing`.

## 4. `automate_image_alignment.py` Logic

Based on Step 4, this script aligns the processed 2D image slices into a consistent 3D volume or aligns sequential acquisitions.

**Core Logic:**

1.  **Initialization:**
    *   Load `config.json`.
    *   Setup logging.
    *   Identify processed image files from `paths.processed_data_dir` that need alignment.

2.  **Image Loading and Preparation:**
    *   Load the series of 2D processed images for a given subject/timepoint.
    *   The `assemble_stack` utility function can be used to load multiple 2D image files (e.g., TIFFs, NPY) into a 3D NumPy array.

3.  **Alignment Strategy (based on `automate_image_alignment_params.alignment_type`):**
    *   **`3d_stack` (Volumetric Alignment):**
        *   Typically involves aligning all slices simultaneously or to a common reference.
        *   May use libraries like SimpleITK for registration.
        *   The `coarse_stack` function might be used for initial downsampling to speed up coarse registration before fine-tuning.
        *   This could involve iterative registration algorithms (e.g., rigid, affine).
    *   **`sequential` (Slice-to-Slice Alignment):**
        *   Align each slice to the previous slice (or a reference slice specified by `reference_slice_index`).
        *   `coarse_stack` can downsample slices before pairwise registration to find initial transformation.
        *   Apply transformations (e.g., translation, rotation) to align slices. Libraries like OpenCV (`cv2.estimateRigidTransform`, `cv2.warpAffine`) or scikit-image can be used.

4.  **Utility Function Usage:**
    *   **`assemble_stack(image_files_list)`:** Loads a list of 2D image file paths into a 3D NumPy array. Handles sorting by slice index if encoded in filenames.
    *   **`coarse_stack(image_stack, factor)`:** Downsamples the 3D image stack by `factor` using averaging or subsampling, useful for speeding up initial alignment.
    *   **`pad_stack_he(image_stack, he_image_stack, padding_value)`:** If aligning iQID images to H&E stained images (not explicitly Ac-225 but a general capability), this function would pad one stack to match the dimensions of another, using `padding_value`. For Ac-225 alone, this might be adapted to pad for consistent volume dimensions using `automate_image_alignment_params.padding_value_he` as a general padding value.
    *   **`crop_down(image_stack, margins_or_target_shape)`:** Crops the aligned stack to remove padding introduced during alignment or to focus on a common region. Uses `automate_image_alignment_params.crop_margins` or a target shape.

5.  **Saving Aligned Data:**
    *   Save the fully aligned 3D image stack (e.g., as a single NIfTI file or series of aligned 2D TIFFs) to `paths.aligned_images_dir`.
    *   Save transformation matrices or parameters if they need to be reapplied or audited.

6.  **Parallelization:**
    *   If `parallelization.enabled` is true, different subjects or timepoints can be aligned in parallel using `max_workers_alignment`.

## 5. `automate_dose_kernel_processing.py` Logic

Based on Step 5, this script prepares the dose kernel and convolves it with the aligned, quantitatively corrected iQID images to generate 3D dose maps.

**Core Logic:**

1.  **Initialization:**
    *   Load `config.json`.
    *   Setup logging.
    *   Load the aligned 3D iQID image (activity map) from `paths.aligned_images_dir`.

2.  **Dose Kernel Preparation:**
    *   **`load_txt_kernel(kernel_filepath)`:**
        *   Loads the dose kernel from the text file specified in `paths.dose_kernel_file`.
        *   The text file is expected to contain radial dose deposition values, typically in columns like (radius, dose_value_per_particle).
        *   Returns a 1D array or similar structure representing the radial kernel.
    *   **`mev_to_mgy(kernel_data, energy_mev_per_particle, conversion_factor)`:**
        *   Converts the kernel's dose values if they are in units like MeV/g per particle to mGy per particle (or per Bq-s).
        *   Uses `automate_dose_kernel_processing_params.kernel_energy_mev` (if needed for normalization, though often the input kernel is already normalized per decay) and `automate_dose_kernel_processing_params.mev_to_mgy_conversion_factor`.
        *   The input `kernel_data` would be the output from `load_txt_kernel`.
    *   **`radial_avg_kernel(radial_kernel_1d, target_voxel_size_mm)`:**
        *   This function seems misnamed if the kernel is already radial. It might be intended to resample or discretize the 1D radial kernel onto a 3D grid matching the `target_voxel_size_mm` or to ensure it's correctly interpreted for 3D convolution.
        *   More likely, this step involves creating a 3D kernel matrix from the 1D radial profile, assuming spherical symmetry. Each voxel in the 3D kernel matrix gets a value based on its distance from the center and the 1D radial profile.
    *   **`pad_kernel_to_vsize(kernel_3d, target_volume_shape, activity_map_voxel_size_mm)`:**
        *   Pads or crops the 3D kernel to match the voxel dimensions and potentially the overall size characteristics required for convolution with the activity map.
        *   Ensures the kernel's voxel grid aligns with the activity map's grid, using `automate_dose_kernel_processing_params.target_voxel_size_mm` as a reference for the activity map's voxel size if resampling of the activity map occurred.
        *   The kernel should be centered in a volume that allows for full convolution.

3.  **Convolution:**
    *   Perform 3D convolution of the prepared 3D dose kernel with the 3D activity map (from `automate_image_alignment.py`).
    *   Libraries like `scipy.signal.convolve` or `scipy.ndimage.convolve` are suitable for this.
    *   Ensure units are consistent (e.g., activity map in Bq/voxel or Bq/mL, kernel in mGy/Bq-s). The resulting dose map will be in mGy.

4.  **Saving Dose Maps:**
    *   Save the resulting 3D dose map (e.g., as a NIfTI file) to `paths.dose_maps_dir`.
    *   The filename should correspond to the input activity map.

## 6. Robustness and Scalability Considerations

Based on Step 6, the following ensures the pipeline is robust and can scale.

*   **Error Handling:**
    *   **Comprehensive `try-except` blocks:** Wrap file I/O, external library calls (image processing, `python-docx`), and numerical computations.
    *   **Specific exceptions:** Catch specific errors (e.g., `FileNotFoundError`, `ValueError`, `MemoryError`) for targeted recovery or messaging.
    *   **Graceful failure:** If a single file/subject fails, the pipeline should log the error and continue with others if possible.
    *   **Configuration validation:** Check `config.json` for missing keys or invalid values at startup.
*   **Logging:**
    *   **Centralized Logging:** Use Python's `logging` module.
    *   **Configurable Log Levels:** Set via `config.json` (`logging.log_level`).
    *   **Timestamped Logs:** Include timestamps, script name, function name, and severity.
    *   **Log Rotation:** For long-running processes, implement log rotation if outputting to files.
    *   **Log important parameters:** At the start of each script run, log key configuration parameters being used.
*   **Parallelization:**
    *   **`concurrent.futures`:** Utilize `ProcessPoolExecutor` for CPU-bound tasks (image processing, convolutions) and `ThreadPoolExecutor` for I/O-bound tasks (if applicable, e.g., downloading many small files, though less relevant here).
    *   **Configurable worker counts:** Allow users to set `max_workers` in `config.json` based on their hardware.
    *   **Resource Management:** Be mindful of memory consumption when parallelizing, especially with large 3D volumes.
*   **Intermediate Outputs:**
    *   **Saving at key stages:** Save outputs from `automate_processing.py` (corrected images, ROIs), `automate_image_alignment.py` (aligned volumes), and `automate_dose_kernel_processing.py` (dose maps).
    *   **Resumability:** This allows the pipeline to be restarted from an intermediate step if a later stage fails, avoiding reprocessing everything.
    *   **Clear Naming Conventions:** Ensure intermediate files are named systematically for easy identification and debugging.
    *   **Debugging:** Intermediate outputs are crucial for debugging issues at specific stages.
*   **Resource Monitoring (Advanced):**
    *   For very large datasets or long runs, consider integrating system monitoring (CPU, memory usage) to identify bottlenecks.
*   **Modularity:**
    *   The separation into three scripts already promotes modularity. Each can be developed, tested, and optimized independently.
*   **Input Validation:**
    *   Beyond `config.json`, validate key properties of input data files (e.g., image dimensions, expected metadata fields in `quantitative_notes.docx`).

By implementing these strategies, the processing pipeline will be more reliable, easier to debug, and adaptable to varying workloads.
```
