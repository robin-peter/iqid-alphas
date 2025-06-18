"""
Simple Pipeline for Automated iQID Processing (Refactored)

Streamlined pipeline that processes a sample directory by detecting its
current stage (Raw, Segmented, Aligned) and applying subsequent
processing steps (Raw -> Split -> Align; Segmented -> Align).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import shutil
import skimage.io # For loading images in alignment step

# Core component imports
from ..core.raw_image_splitter import RawImageSplitter
from ..core.alignment import ImageAligner
from ..core.processor import IQIDProcessor # May be needed for preprocessing if that's added back
from ..core.segmentation import ImageSegmenter # May be needed for per-slice segmentation if added back

# Note: Visualizer import removed as _create_overview is removed.
# HAS_IMAGING check also removed as core components will raise ImportErrors if skimage is missing.


class SimplePipeline:
    """
    Processes a sample directory through iQID workflow stages.
    Detects if data is Raw, Segmented, or Aligned and processes accordingly.
    - Raw data: Splits into slices, then aligns the slices.
    - Segmented data: Aligns the existing slices.
    - Aligned data: Reports as already processed.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline.

        Parameters:
        ----------
        config : dict, optional
            Configuration dictionary (currently unused but placeholder for future settings).
        """
        self.config = config or {}
        self.raw_splitter = RawImageSplitter()  # Default 3x3 grid
        self.aligner = ImageAligner()
        # self.processor = IQIDProcessor() # Uncomment if preprocessing is re-introduced
        # self.segmenter = ImageSegmenter() # Uncomment if per-slice segmentation is re-introduced
    
    def process_iqid_stack(self, sample_dir_str: str, output_dir_str: str, forced_stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an iQID sample directory.
        Detects stage (Raw, Segmented, Aligned) and processes accordingly,
        unless a stage is forced via the `forced_stage` parameter.

        Parameters:
        ----------
        sample_dir_str : str
            Path to the sample directory.
        output_dir_str : str
            Path to the output directory where results (e.g., aligned slices) will be saved.
        forced_stage : Optional[str], optional
            If provided (e.g., 'raw', 'segmented', 'aligned'), the pipeline will attempt
            to process from this stage directly, bypassing auto-detection. Default is None.

        Returns:
        -------
        Dict[str, Any]
            A dictionary containing the status and information about the processing.
        """
        sample_dir = Path(sample_dir_str)
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {
            'status': 'failed',
            'message': '',
            'processed_stage': None,
            'output_location': None
        }

        raw_dir = sample_dir / "Raw"
        segmented_dir = sample_dir / "1_segmented"
        aligned_dir = sample_dir / "2_aligned"
        
        current_stage = "unknown"
        raw_tiff_path: Optional[Path] = None
        segmented_slice_files: List[Path] = []

        # Stage Detection or Use Forced Stage
        if forced_stage:
            current_stage = forced_stage
            print(f"INFO: Using forced stage: {current_stage} for {sample_dir.name}")
            # Validate if forced stage is plausible
            if current_stage == "raw":
                if not raw_dir.exists() or not any(f.suffix.lower() in ['.tif', '.tiff'] for f in raw_dir.iterdir() if f.is_file()):
                    raise ValueError(f"Forced stage 'raw' but Raw directory missing or empty in {sample_dir}")
                raw_files = sorted(list(f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))
                if raw_files: raw_tiff_path = raw_files[0]
            elif current_stage == "segmented":
                if not segmented_dir.exists() or not any(f.suffix.lower() in ['.tif', '.tiff'] for f in segmented_dir.iterdir() if f.is_file()):
                    raise ValueError(f"Forced stage 'segmented' but 1_segmented directory missing or empty in {sample_dir}")
                segmented_slice_files = sorted(list(f for f in segmented_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))
            elif current_stage == "aligned":
                 if not aligned_dir.exists() or not any(f.suffix.lower() in ['.tif', '.tiff'] for f in aligned_dir.iterdir() if f.is_file()):
                    raise ValueError(f"Forced stage 'aligned' but 2_aligned directory missing or empty in {sample_dir}")
                 # If forced to aligned, we treat it like auto-detection of aligned.
            else:
                raise ValueError(f"Invalid forced_stage: {forced_stage}. Must be 'raw', 'segmented', or 'aligned'.")
        else:
            # Auto-Detection Logic
            if aligned_dir.exists() and any(f.suffix.lower() in ['.tif', '.tiff'] for f in aligned_dir.iterdir() if f.is_file()):
                current_stage = "aligned"
            elif segmented_dir.exists():
                segmented_slice_files = sorted(list(f for f in segmented_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))
                if segmented_slice_files:
                    current_stage = "segmented"
                # If segmented_dir is empty, fall through to check raw_dir
            
            if current_stage == "unknown" and raw_dir.exists(): # Check raw if no aligned/segmented found or segmented was empty
                raw_files = sorted(list(f for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))
                if raw_files:
                    raw_tiff_path = raw_files[0]
                    current_stage = "raw"
            
            if current_stage == "unknown": # If still unknown after all checks
                # More specific error messages based on what was checked
                msg = f"No processable data found in {sample_dir}."
                if not (aligned_dir.exists() or segmented_dir.exists() or raw_dir.exists()):
                    msg = f"No Raw, 1_segmented, or 2_aligned directory found in {sample_dir}."
                elif segmented_dir.exists() and not segmented_slice_files and \
                     not (raw_dir.exists() and any(f.suffix.lower() in ['.tif', '.tiff'] for f in raw_dir.iterdir() if f.is_file())):
                    msg = f"Segmented directory {segmented_dir} is empty, and no valid Raw data found."
                elif raw_dir.exists() and not raw_tiff_path:
                     msg = f"Raw directory {raw_dir} is empty."
                raise ValueError(msg)

        print(f"INFO: Determined stage: {current_stage} for {sample_dir.name}")

        # Specific handling for "aligned" stage (whether forced or auto-detected)
        if current_stage == "aligned":
            print(f"INFO: Stage is '{current_stage}'. Sample {sample_dir.name} is considered already aligned.")
            results['status'] = 'success'
            results['message'] = f"Sample {sample_dir.name} is already at or forced to 'aligned' stage. No further processing in SimplePipeline."
            results['processed_stage'] = current_stage
            results['output_location'] = str(aligned_dir) # Or sample_dir if forced to aligned without 2_aligned existing?
            print(f"QA: Already aligned (or forced to aligned) for sample {sample_dir.name}.")
            return results

        # Continue with processing for 'raw' or 'segmented'
        results['processed_stage'] = current_stage

        temp_segmented_dir: Optional[Path] = None
        alignment_input_slice_paths: List[Path] = []

        try:
            if current_stage == "raw":
                if raw_tiff_path is None: # Should not happen if current_stage is "raw"
                    raise ValueError("Raw stage detected but no raw TIFF file found.")

                # Define a unique temp directory for this sample's segmented slices
                temp_segmented_dir = output_dir / f"{sample_dir.name}_temp_segmented"
                temp_segmented_dir.mkdir(parents=True, exist_ok=True)

                print(f"INFO: Splitting raw image for {sample_dir.name}...")
                raw_slice_paths_str = self.raw_splitter.split_image(str(raw_tiff_path), str(temp_segmented_dir))
                alignment_input_slice_paths = [Path(p) for p in raw_slice_paths_str]
                print(f"QA: Raw splitting complete for {sample_dir.name}. {len(alignment_input_slice_paths)} slices generated.")
                if not alignment_input_slice_paths:
                    results['message'] = f"Raw splitting yielded no slices for {raw_tiff_path}."
                    raise RuntimeError(results['message'])
            
            elif current_stage == "segmented":
                alignment_input_slice_paths = segmented_slice_files
                print(f"QA: Starting from segmented stage for {sample_dir.name}. {len(alignment_input_slice_paths)} slices found.")
                if not alignment_input_slice_paths:
                     results['message'] = f"Segmented stage detected but no slice files found in {segmented_dir}."
                     raise RuntimeError(results['message'])

            # --- Alignment Step (if data was raw or segmented) ---
            if current_stage in ["raw", "segmented"]:
                if not alignment_input_slice_paths:
                     results['message'] = "No slices available for alignment."
                     raise RuntimeError(results['message'])

                final_aligned_dir = output_dir / f"{sample_dir.name}_aligned_output"
                final_aligned_dir.mkdir(parents=True, exist_ok=True)
                results['output_location'] = str(final_aligned_dir)

                print(f"INFO: Starting alignment for {sample_dir.name} into {final_aligned_dir}...")

                # Simplified sequential alignment:
                # Align each slice to the previously aligned slice.
                # The first slice is the initial reference.

                aligned_slice_output_paths: List[str] = []

                # Handle first slice
                reference_image_data = skimage.io.imread(str(alignment_input_slice_paths[0]))
                # TODO: Consider if preprocessing from self.processor should be applied here or to each slice

                first_slice_out_path = final_aligned_dir / f"aligned_0.tif"
                skimage.io.imsave(str(first_slice_out_path), reference_image_data, plugin='tifffile', check_contrast=False)
                aligned_slice_output_paths.append(str(first_slice_out_path))

                for i in range(1, len(alignment_input_slice_paths)):
                    moving_image_data = skimage.io.imread(str(alignment_input_slice_paths[i]))
                    # TODO: Consider preprocessing for moving_image_data

                    print(f"  Aligning slice {i} to previous...")
                    # Align to the *previous* successfully aligned image
                    aligned_image_data, _ = self.aligner.align_images(reference_image_data, moving_image_data)

                    current_aligned_slice_path = final_aligned_dir / f"aligned_{i}.tif"
                    skimage.io.imsave(str(current_aligned_slice_path), aligned_image_data, plugin='tifffile', check_contrast=False)
                    aligned_slice_output_paths.append(str(current_aligned_slice_path))

                    reference_image_data = aligned_image_data # Update reference for next iteration

                print(f"QA: Alignment complete for {sample_dir.name}. {len(aligned_slice_output_paths)} slices aligned.")
                results['message'] = f"Successfully processed {sample_dir.name} from {current_stage} stage. Aligned slices: {len(aligned_slice_output_paths)}."
                results['status'] = 'success'
            
        except Exception as e:
            print(f"ERROR: Processing {sample_dir.name} failed: {e}")
            results['message'] = str(e)
            results['status'] = 'failed'
            # Re-raise to ensure test failures if not caught by specific test asserts
            raise
        finally:
            if temp_segmented_dir and temp_segmented_dir.exists():
                print(f"INFO: Cleaning up temporary directory: {temp_segmented_dir}")
                shutil.rmtree(temp_segmented_dir)
        
        return results

    # Old methods like _load_stack, _align_stack, _save_stack, _create_overview, process_batch
    # are removed as their functionality is either integrated into process_iqid_stack
    # or is not compatible with the new stage-aware, directory-based processing.
    # The CLI will now be responsible for batching calls to this pipeline.

# ```python
# Example of how this might be called by the CLI (for conceptual understanding)
# if __name__ == '__main__':
#     # Setup a dummy ReUpload structure
#     # Sample 1: Raw
#     sample1_dir = Path("temp_sample1_raw")
#     (sample1_dir / "Raw").mkdir(parents=True, exist_ok=True)
#     dummy_raw_data = np.random.randint(0, 255, (90,90), dtype=np.uint8)
#     skimage.io.imsave(str(sample1_dir / "Raw" / "raw_image.tif"), dummy_raw_data)

#     # Sample 2: Segmented
#     sample2_dir = Path("temp_sample2_segmented")
#     (sample2_dir / "1_segmented").mkdir(parents=True, exist_ok=True)
#     for i in range(3): # Create 3 dummy segmented slices
#         dummy_seg_slice = np.random.randint(0, 255, (30,30), dtype=np.uint8)
#         skimage.io.imsave(str(sample2_dir / "1_segmented" / f"seg_slice_{i}.tif"), dummy_seg_slice)
    
#     # Sample 3: Aligned
#     sample3_dir = Path("temp_sample3_aligned")
#     (sample3_dir / "2_aligned").mkdir(parents=True, exist_ok=True)
#     dummy_aligned_slice = np.random.randint(0, 255, (30,30), dtype=np.uint8)
#     skimage.io.imsave(str(sample3_dir / "2_aligned" / "aligned_slice_0.tif"), dummy_aligned_slice)

#     pipeline = SimplePipeline()
#     output_base = Path("temp_pipeline_outputs")
#     output_base.mkdir(exist_ok=True)

#     print("\n--- Processing Sample 1 (Raw) ---")
#     pipeline.process_iqid_stack(str(sample1_dir), str(output_base / "sample1_out"))
    
#     print("\n--- Processing Sample 2 (Segmented) ---")
#     pipeline.process_iqid_stack(str(sample2_dir), str(output_base / "sample2_out"))

#     print("\n--- Processing Sample 3 (Aligned) ---")
#     pipeline.process_iqid_stack(str(sample3_dir), str(output_base / "sample3_out"))

#     # Clean up dummy samples
#     shutil.rmtree(sample1_dir)
#     shutil.rmtree(sample2_dir)
#     shutil.rmtree(sample3_dir)
#     # Keep temp_pipeline_outputs for inspection or remove it too
#     # shutil.rmtree(output_base)
# ```
