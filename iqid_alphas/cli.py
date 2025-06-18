#!/usr/bin/env python3
"""
IQID-Alphas Command Line Interface

Advanced batch processing CLI for the IQID-Alphas project providing:
- Unified interface for all processing workflows
- Batch processing with progress monitoring
- Quality control and validation
- Automated report generation
- Configuration management
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import traceback
import re # Added import re

# Add package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from iqid_alphas.pipelines.simple import SimplePipeline
from iqid_alphas.pipelines.advanced import AdvancedPipeline
from iqid_alphas.pipelines.combined import CombinedPipeline


class IQIDCLIProcessor:
    """Advanced CLI processor for IQID-Alphas batch operations."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.start_time = datetime.now()
        self.results = {
            'processed': 0,
            'success': 0,
            'failed': 0,
            'errors': [],
            'summary': {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('iqid_cli')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _detect_dataset_type(self, data_path: Path) -> str:
        """Detect if this is DataPush1 (production) or ReUpload (workflow) dataset."""
        path_str = str(data_path).lower()
        
        if "datapush1" in path_str:
            return "production"  # DataPush1: aligned data with H&E
        elif "reupload" in path_str or "iqid_reupload" in path_str:
            return "workflow"    # ReUpload: full workflow, iQID only
        else:
            # Try to detect by structure
            if (data_path / "HE").exists():
                return "production"  # Has H&E data = production dataset
            elif (data_path / "Raw").exists() or any(p.name == "Raw" for p in data_path.rglob("Raw")):
                return "workflow"    # Has Raw directories = workflow dataset
            else:
                return "unknown"
    
    def _detect_processing_stage(self, sample_dir: Path, dataset_type: str) -> str:
        """Detect processing stage based on dataset type and available directories."""
        if dataset_type == "production":
            # DataPush1: Only aligned data available
            return "aligned_ready"
        elif dataset_type == "workflow":
            # ReUpload: Check for workflow stages
            if (sample_dir / "Raw").exists():
                return "raw_available"
            elif (sample_dir / "1_segmented").exists():
                return "segmented_available"
            elif (sample_dir / "2_aligned").exists():
                return "aligned_available"
            elif any(sample_dir.glob("mBq_corr_*.tif")):
                # Direct slice files (already processed)
                return "slices_available"
        return "unknown_stage"
    
    def discover_data(self, data_path: Union[str, Path]) -> Dict[str, List[Path]]:
        """Discover available data samples with dataset-aware processing."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Detect dataset type
        dataset_type = self._detect_dataset_type(data_path)
        
        discovered = {
            'dataset_type': dataset_type,
            'iqid_samples': [],
            'he_samples': [],
            'paired_samples': [],
            # 'sample_metadata': {}, # Removed as per plan
            'dataset_info': {
                'type': dataset_type,
                # 'has_he_data' and 'multi_modal_capable' are still useful at dataset level
                'has_he_data': dataset_type == "production", # Overall dataset characteristic
                'has_workflow_stages': dataset_type == "workflow",
                'multi_modal_capable': dataset_type == "production"
            }
        }
        
        self.logger.info(f"Detected dataset type: {dataset_type}")
        
        # Dataset-specific discovery
        if dataset_type == "production":
            # These methods are assumed to populate discovered['iqid_samples'] and discovered['he_samples']
            # with dicts from the refactored _analyze_sample_directory
            self._discover_production_samples(data_path, discovered)
        elif dataset_type == "workflow":
            self._discover_workflow_samples(data_path, discovered)
        else:
            self._discover_generic_samples(data_path, discovered)

        # After individual samples are analyzed and H&E/iQID samples are listed,
        # perform pairing for production datasets.
        if dataset_type == "production" and discovered['iqid_samples'] and discovered['he_samples']:
            paired_samples_temp = self._pair_samples(discovered['iqid_samples'], discovered['he_samples'])

            # Update individual iqid_sample dicts and create final paired_samples list
            final_paired_samples = []
            iqid_samples_in_pairs = set()

            for pair in paired_samples_temp:
                if pair['he_sample']: # Successfully paired
                    pair['iqid_sample']['has_he_data'] = True
                    pair['iqid_sample']['multi_modal_ready'] = True
                    # The pair structure itself implies multi_modal_ready for the pair entry
                    pair_entry = {
                        'sample_id': pair['sample_id'],
                        'iqid_sample': pair['iqid_sample'],
                        'he_sample': pair['he_sample'],
                        'match_score': pair['match_score'],
                        'multi_modal_ready': True # Explicit for the pair
                    }
                    final_paired_samples.append(pair_entry)
                    iqid_samples_in_pairs.add(pair['iqid_sample']['sample_id'])
                else: # iQID sample was not paired
                    pair['iqid_sample']['has_he_data'] = False
                    pair['iqid_sample']['multi_modal_ready'] = False
                    # It will be handled in the next loop if not already added

            # self.logger.info(f"DEBUG: Final paired samples before assignment: {json.dumps(final_paired_samples, default=str, indent=2)}") # Removed debug print
            discovered['paired_samples'] = final_paired_samples

            # Ensure all iqid_samples (even unpaired ones) have these fields initialized
            for iq_sample in discovered['iqid_samples']:
                if iq_sample['sample_id'] not in iqid_samples_in_pairs:
                     iq_sample['has_he_data'] = False
                     iq_sample['multi_modal_ready'] = False
        else:
            # For workflow or other non-production, or if no H&E/iQID to pair
            for iq_sample in discovered['iqid_samples']:
                iq_sample['has_he_data'] = False
                iq_sample['multi_modal_ready'] = False
            for he_sample in discovered['he_samples']: # Should be empty for workflow
                he_sample['has_he_data'] = True # It is H&E data
                he_sample['multi_modal_ready'] = False # Cannot be multi-modal alone
        
        return discovered

    def _discover_production_samples(self, data_path: Path, discovered: Dict):
        """Discovers iQID and H&E samples in a DataPush1 (production) structure."""
        self.logger.info(f"Discovering production samples in: {data_path}")
        # Expected structure: data_path/iQID/3D/tissue/sample_id & data_path/HE/3D/tissue/sample_id

        iqid_base_path = data_path / "iQID" / "3D"
        he_base_path = data_path / "HE" / "3D"

        if iqid_base_path.exists():
            for tissue_dir in iqid_base_path.iterdir():
                if tissue_dir.is_dir():
                    for sample_dir in tissue_dir.iterdir():
                        if sample_dir.is_dir():
                            self.logger.debug(f"Analyzing production iQID sample: {sample_dir}")
                            sample_info = self._analyze_sample_directory(sample_dir, "production", data_modality='iqid')
                            discovered['iqid_samples'].append(sample_info)

        if he_base_path.exists():
            for tissue_dir in he_base_path.iterdir():
                if tissue_dir.is_dir():
                    for sample_dir in tissue_dir.iterdir():
                        if sample_dir.is_dir():
                            self.logger.debug(f"Analyzing production H&E sample: {sample_dir}")
                            sample_info = self._analyze_sample_directory(sample_dir, "production", data_modality='he')
                            discovered['he_samples'].append(sample_info)
        
        self.logger.info(f"Found {len(discovered['iqid_samples'])} iQID, {len(discovered['he_samples'])} H&E production samples.")


    def _discover_workflow_samples(self, data_path: Path, discovered: Dict):
        """Discovers iQID samples in a ReUpload (workflow) structure."""
        # Assumes data_path is the root of ReUpload, e.g., ReUpload/
        # And samples are direct subdirectories like ReUpload/sample_001, ReUpload/sample_002
        # OR, if data_path is ReUpload/iQID_reupload/, then samples are subdirs of that.
        self.logger.info(f"Discovering workflow samples in: {data_path}")

        # Check if data_path itself is a single sample directory (contains Raw, 1_segmented, or 2_aligned)
        is_single_sample_dir = False
        if (data_path / "Raw").exists() or \
           (data_path / "1_segmented").exists() or \
           (data_path / "2_aligned").exists():
            is_single_sample_dir = True

        if is_single_sample_dir:
            self.logger.debug(f"Analyzing provided data_path as a single workflow sample: {data_path}")
            sample_info = self._analyze_sample_directory(data_path, "workflow", data_modality='iqid')
            discovered['iqid_samples'].append(sample_info)
        else:
            # Otherwise, iterate through subdirectories in data_path (or a potential iQID_reupload subdir)
            potential_sample_parent_dir = data_path
            if data_path.name == "ReUpload" and (data_path / "iQID_reupload").exists():
                potential_sample_parent_dir = data_path / "iQID_reupload"

            for item in potential_sample_parent_dir.iterdir():
                if item.is_dir(): # Each subdirectory is a potential sample
                    # Check if this item itself looks like a sample dir (contains Raw, etc.)
                    if (item / "Raw").exists() or \
                       (item / "1_segmented").exists() or \
                       (item / "2_aligned").exists():
                        self.logger.debug(f"Analyzing workflow sample: {item}")
                        sample_info = self._analyze_sample_directory(item, "workflow", data_modality='iqid')
                        discovered['iqid_samples'].append(sample_info)
                    else:
                        self.logger.debug(f"Skipping directory {item} as it does not appear to be a valid sample (no Raw/1_segmented/2_aligned).")

        self.logger.info(f"Found {len(discovered['iqid_samples'])} workflow samples.")

    def _discover_generic_samples(self, data_path: Path, discovered: Dict):
        """Fallback discovery for unknown dataset structures."""
        self.logger.warning(f"Using generic discovery for unknown dataset structure at: {data_path}")
        for item in data_path.iterdir():
            if item.is_dir(): # Treat each directory as a potential iQID sample
                # This is a very basic guess; might need more heuristics
                self.logger.debug(f"Analyzing generic iQID sample: {item}")
                sample_info = self._analyze_sample_directory(item, "unknown", data_modality='iqid')
                discovered['iqid_samples'].append(sample_info)
        self.logger.info(f"Found {len(discovered['iqid_samples'])} generic iQID samples.")


    def _analyze_sample_directory(self, sample_dir: Path, dataset_type: str, data_modality: str = 'iqid') -> Dict:
        """
        Analyzes a sample directory to extract comprehensive metadata based on specifications.
        data_modality can be 'iqid' or 'he'.
        """
        sample_id = sample_dir.name
        
        # Use helper functions for extraction
        tissue_type = self._extract_tissue_type(sample_dir)
        preprocessing_type = self._extract_preprocessing_type(sample_dir)
        # Laterality can also be added if needed: self._extract_laterality(sample_id)
        
        # Get available stages based on directory structure
        available_stages = self._list_available_stages(sample_dir, dataset_type)

        # Determine current processing stage for this specific sample dir
        # This might differ from the overall dataset_type if we are looking at sub-folders of a ReUpload.
        # For a specific sample_dir, its own structure dictates its current stage.
        # For DataPush1, sample_dir itself contains final data.
        # For ReUpload, sample_dir is the parent (e.g. sample_001), and stages are subdirs.
        # The existing _detect_processing_stage is good for overall CLI flow.
        # Here, we need to be more specific about the sample_dir passed.

        current_processing_stage = "unknown"
        slice_files = []

        if data_modality == 'iqid':
            # For iQID, determine stage based on subdirectories if workflow, or files if production
            if dataset_type == "workflow":
                if "aligned" in available_stages and (sample_dir / "2_aligned").exists():
                    current_processing_stage = "aligned"
                    slice_files = sorted(list((sample_dir / "2_aligned").glob("*.tif*")))
                elif "segmented" in available_stages and (sample_dir / "1_segmented").exists():
                    current_processing_stage = "segmented"
                    slice_files = sorted(list((sample_dir / "1_segmented").glob("*.tif*")))
                elif "raw" in available_stages and (sample_dir / "Raw").exists():
                    current_processing_stage = "raw" # Raw stage might have one multi-page TIFF or multiple files.
                                                # Assuming split raw files are not yet here.
                    # For raw, slice_files might be the raw file itself, or pre-split slices if that step was done.
                    # This part needs to align with how RawImageSplitter output is handled or if it's pre-split.
                    # For now, let's assume raw means the container, and slice_count might be from metadata or 1.
                    slice_files = sorted(list((sample_dir / "Raw").glob("*.tif*")))

            elif dataset_type == "production": # DataPush1 iQID
                current_processing_stage = "aligned_ready" # or "aligned"
                slice_files = sorted(list(f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))

            # Fallback if no specific structure but tiffs are present
            if not slice_files and "slices_available" in available_stages:
                 current_processing_stage = "slices_available" # or map to "aligned"
                 slice_files = sorted(list(f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))

        elif data_modality == 'he': # H&E data (always 'production' dataset type context for H&E)
            current_processing_stage = "aligned_ready" # H&E is assumed ready
            slice_files = sorted(list(f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']))

        # Refine processing_stage based on what _detect_processing_stage would yield for the sample_dir itself
        # This ensures consistency if sample_dir is passed to _detect_processing_stage elsewhere.
        # The _detect_processing_stage is more about the *primary* stage of a sample folder.
        # The 'available_stages' lists what subfolders exist.
        # 'current_processing_stage' here should reflect the most advanced data found *directly* for this sample_dir
        # (considering it might be a sub-folder itself, or a main sample folder).
        # For now, the logic above for current_processing_stage and slice_files should be a good start.
        # The _detect_iqid_processing_stage, _sort_slice_files, _assess_workflow_readiness might be more detailed
        # and could be integrated if they offer more nuance than simple globbing.
        # Let's assume for now the above globbing is sufficient for this refactoring pass.

        slice_count = len(slice_files)

        # `has_he_data` and `multi_modal_ready` depend on context outside this single sample analysis
        # `discover_data` will set these based on overall dataset structure and pairing.
        # Here, we focus on what can be known from sample_dir and dataset_type.

        can_reconstruct_3d = current_processing_stage in ["aligned", "aligned_ready", "slices_available"] and slice_count > 0

        # Convert Path objects in slice_files to strings for JSON serialization if needed later
        slice_files_str = [str(p) for p in slice_files]

        sample_info = {
            'sample_dir': str(sample_dir), # Store as string for broader compatibility (e.g. JSON)
            'sample_id': sample_id,
            'dataset_type': dataset_type,
            'data_modality': data_modality, # 'iqid' or 'he'
            'tissue_type': tissue_type,
            'preprocessing_type': preprocessing_type, # From path
            'processing_stage': current_processing_stage, # Most advanced stage found for this dir
            'available_stages': available_stages, # List of Raw, Segmented, Aligned subdirs found
            'slice_count': slice_count,
            'slice_files': slice_files_str,
            'can_reconstruct_3d': can_reconstruct_3d,
            # 'has_he_data' and 'multi_modal_ready' will be determined by discover_data after pairing.
            'size_mb': sum(f.stat().st_size for f in slice_files) / (1024 * 1024) if slice_files else 0,
            # 'workflow_ready': self._assess_workflow_readiness(processing_stage, len(files)) # This was from old version, needs re-eval
        }
        return sample_info
    
    def _pair_samples(self, iqid_samples: List[Dict], he_samples: List[Dict]) -> List[Dict]:
        """Attempt to pair iQID samples with H&E samples based on naming patterns."""
        paired = []
        
        for iqid_sample in iqid_samples:
            # Extract potential sample identifier from directory name
            iqid_id = self._extract_sample_id(iqid_sample['sample_id'])
            
            best_match = None
            best_score = 0
            
            for he_sample in he_samples:
                he_id = self._extract_sample_id(he_sample['sample_id'])
                score = self._similarity_score(iqid_id, he_id)
                
                if score > best_score and score > 0.5:  # Increased threshold to 0.5
                    best_match = he_sample
                    best_score = score
            
            paired.append({
                'iqid_sample': iqid_sample,
                'he_sample': best_match,
                'sample_id': iqid_id,
                'match_score': best_score,
                'type': 'sample_pair'  # Indicates this is a sample pair, not individual image
            })
        
        return paired
    
    def _extract_sample_id(self, filename: str) -> str:
        """Extract sample identifier from filename or directory name."""
        # Handle directory names like D1M1(P1)_L, D7M2(P2)_R, etc.
        base = filename
        
        # Remove common file extensions
        if '.' in base:
            base = Path(base).stem
        
        # Handle iQID naming pattern: D1M1(P1)_L -> D1M1_L
        # Remove (P*) parts to match with H&E names
        import re
        base = re.sub(r'\(P\d+\)', '', base)
        
        # Clean up any double underscores
        base = re.sub(r'_+', '_', base)
        
        # Remove common prefixes/suffixes
        patterns_to_remove = [
            'mBq_corr_', 'aligned_', 'processed_', 'segmented_',
            '_he', '_iqid', '_corr', '_aligned'
        ]
        
        for pattern in patterns_to_remove:
            base = base.replace(pattern, '')
        
        return base.lower().strip('_')

    def _extract_tissue_type(self, path_obj: Path) -> str:
        """Extract tissue type from path hierarchy (e.g., kidney, tumor) by checking for keyword containment."""
        keywords_map = {'kidney': 'kidney', 'kidneys': 'kidney', 'tumor': 'tumor'}
        # Iterate from deeper to shallower parts of the path for more specific matches first
        # and also check the sample directory name itself if it's a direct name like "D1M1_L_kidney_data"
        path_parts_to_check = list(path_obj.parts)
        if path_obj.name not in path_parts_to_check: # Should typically be the last part
             path_parts_to_check.append(path_obj.name)

        for part in reversed(path_parts_to_check):
            low_part = part.lower()
            for keyword_search, keyword_return in keywords_map.items():
                if keyword_search in low_part:
                    return keyword_return
        return 'unknown'

    def _extract_preprocessing_type(self, path_obj: Path) -> str:
        """Extract preprocessing type from path hierarchy (e.g., 3D, Sequential)."""
        # Assumes path_obj is the sample_dir
        for part in path_obj.parts:
            low_part = part.lower()
            # Common terms indicating preprocessing type for iQID/H&E, often parent of tissue type
            if low_part in ['3d', 'sequential', 'upper', 'lower', 'upper and lower']:
                return low_part
        # Fallback or more sophisticated extraction
        if len(path_obj.parts) >= 3:
            grandparent_low = path_obj.parts[-3].lower()
            if grandparent_low in ['3d', 'sequential', 'upper', 'lower', 'upper and lower']:
                return grandparent_low
        return 'unknown'

    def _extract_laterality(self, sample_name: str) -> str:
        """Extract laterality (left/right) from sample name."""
        sample_name_low = sample_name.lower()

        # Check for _L or _R followed by a common separator or end of string
        if re.search(r'_l([._]|$)', sample_name_low): return 'left'
        if re.search(r'_r([._]|$)', sample_name_low): return 'right'

        # Check for whole word 'left' or 'right'
        if 'left' in sample_name_low: return 'left'
        if 'right' in sample_name_low: return 'right'

        # Fallback for simple L/R suffix if not part of a common word
        # and preceded by a non-alphabetic character (or start of string implicitly)
        if sample_name_low.endswith('l'):
            if len(sample_name_low) == 1: return 'left' # "L"
            if len(sample_name_low) > 1 and not sample_name_low[-2].isalpha() and \
               not sample_name_low.endswith("ial") and \
               not sample_name_low.endswith("nal") and \
               not sample_name_low.endswith("xel"): # e.g. pixel
                 return 'left'
        if sample_name_low.endswith('r'):
            if len(sample_name_low) == 1: return 'right' # "R"
            if len(sample_name_low) > 1 and not sample_name_low[-2].isalpha() and \
               not sample_name_low.endswith("ior") and \
               not sample_name_low.endswith("lar"): # e.g. similar
                return 'right'

        return 'unknown'

    def _list_available_stages(self, sample_dir: Path, dataset_type: str) -> list[str]:
        """Checks for Raw, 1_segmented, 2_aligned subdirs for 'workflow' type."""
        stages = []
        if dataset_type == "workflow":
            if (sample_dir / "Raw").exists() and any((sample_dir / "Raw").glob("*.tif*")):
                stages.append("raw")
            if (sample_dir / "1_segmented").exists() and any((sample_dir / "1_segmented").glob("*.tif*")):
                stages.append("segmented")
            if (sample_dir / "2_aligned").exists() and any((sample_dir / "2_aligned").glob("*.tif*")):
                stages.append("aligned")
        elif dataset_type == "production":
            # Production data is assumed to be aligned.
            # Check if files like mBq_corr_*.tif or P*.tif exist directly in the sample_dir
            if any(f.suffix.lower() in ['.tif', '.tiff'] for f in sample_dir.iterdir() if f.is_file()):
                 stages.append("aligned_ready") # Or just "aligned"
        if not stages and any(f.suffix.lower() in ['.tif', '.tiff'] for f in sample_dir.iterdir() if f.is_file()):
            # If no subfolders, but files directly in sample_dir, consider it 'slices_available' or 'aligned'
            # This aligns with _detect_processing_stage's "slices_available"
            stages.append("slices_available") # Could map to 'aligned'
        return stages
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Calculate similarity score between two strings."""
        if not str1 or not str2:
            return 0.0
        
        # Simple Jaccard similarity based on character sets
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def process_batch(self, config_path: str, data_path: str, 
                     pipeline_type: str = 'simple',
                     stage_arg: str = 'auto', # Added stage_arg
                     max_samples: Optional[int] = None,
                     output_dir: Optional[str] = None) -> Dict:
        """Process a batch of samples using the specified pipeline."""
        
        self.logger.info(f"🚀 Starting batch processing with {pipeline_type} pipeline, stage: {stage_arg}")
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Config: {config_path}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self.results
        
        # Discover data
        try:
            discovered = self.discover_data(data_path)
        except Exception as e:
            self.logger.error(f"Data discovery failed: {e}")
            return self.results
        
        # Select pipeline
        pipeline_classes = {
            'simple': SimplePipeline,
            'advanced': AdvancedPipeline,
            'combined': CombinedPipeline
        }
        
        if pipeline_type not in pipeline_classes:
            self.logger.error(f"Unknown pipeline type: {pipeline_type}")
            return self.results
        
        PipelineClass = pipeline_classes[pipeline_type]
        
        # Determine samples to process
        dataset_type = discovered.get('dataset_type', 'unknown')
        
        items_to_process_source = []
        if pipeline_type == 'combined':
            if not discovered['paired_samples'] and discovered['iqid_samples']:
                self.logger.warning("Combined pipeline selected, but no strictly paired samples discovered. Will attempt to process individual iQID samples if H&E is optional for the pipeline or this run.")
                # This case is tricky: CombinedPipeline usually needs pairs.
                # Forcing it onto iqid_samples means he_sample will be None.
                # The pipeline's process_image_pair must handle he_path=None gracefully or this will fail.
                # For now, let's assume if user chose combined, they expect pairs.
                # If no pairs, then it effectively means no H&E data.
                items_to_process_source = discovered['paired_samples'] # Stick to paired_samples for 'combined'
            elif not discovered['paired_samples']:
                 self.logger.error("Combined pipeline selected, but no paired samples discovered.")
                 return self.results # Exit early if combined pipeline has no pairs
            else:
                items_to_process_source = discovered['paired_samples']
        else: # 'simple' or 'advanced'
            if discovered['iqid_samples']:
                items_to_process_source = discovered['iqid_samples']
            elif discovered['paired_samples']:
                self.logger.warning("No individual iQID samples found for simple/advanced pipeline. Using iQID part of paired samples.")
                items_to_process_source = [p['iqid_sample'] for p in discovered['paired_samples'] if 'iqid_sample' in p]
            else:
                self.logger.error("No iQID samples or paired samples found for processing with simple/advanced pipeline.")
                return self.results

        if not items_to_process_source:
            self.logger.info("No samples to process after initial filtering based on pipeline type.")
            self._generate_summary() # Ensure summary is generated even if no processing occurs
            return self.results

        items_to_process = items_to_process_source[:max_samples] if max_samples is not None else items_to_process_source

        self.logger.info(f"Attempting to process {len(items_to_process)} item(s)...")
        
        for i, item_data_raw in enumerate(items_to_process, 1):
            current_sample_id = "unknown_sample"
            # These will be populated based on item_data_raw structure
            iqid_sample_info = None
            he_sample_info = None

            # Determine the primary iQID sample info and ID
            if 'iqid_sample' in item_data_raw and isinstance(item_data_raw['iqid_sample'], dict): # Paired sample structure
                iqid_sample_info = item_data_raw['iqid_sample']
                he_sample_info = item_data_raw.get('he_sample') # he_sample_info could be None
                current_sample_id = iqid_sample_info.get('sample_id', f"paired_item_{i}")
            elif 'sample_id' in item_data_raw and 'sample_dir' in item_data_raw : # iQID sample structure
                iqid_sample_info = item_data_raw
                current_sample_id = iqid_sample_info.get('sample_id', f"iqid_item_{i}")
            else:
                self.logger.error(f"Skipping item {i} due to unrecognized structure: {item_data_raw}")
                self.results['processed'] += 1
                self.results['failed'] += 1
                self.results['errors'].append(f"Unrecognized sample structure for item {i}")
                continue

            self.results['processed'] += 1 # Mark as attempted upfront
            try:
                self.logger.info(f"Processing item {i}/{len(items_to_process)}: {current_sample_id} (Dataset: {dataset_type})")
                
                pipeline_instance = PipelineClass(config) # Base config
                
                current_sample_output_base = Path(output_dir) if output_dir else Path("results")
                current_sample_output_path = current_sample_output_base / f"{pipeline_type}_output" / f"sample_{current_sample_id}"
                current_sample_output_path.mkdir(parents=True, exist_ok=True)
                current_sample_output_str = str(current_sample_output_path)
                
                # Update config for visualization output dir (pipelines might use this)
                sample_specific_config = config.copy()
                if 'visualization' not in sample_specific_config: sample_specific_config['visualization'] = {}
                sample_specific_config['visualization']['output_dir'] = current_sample_output_str
                pipeline_instance = PipelineClass(sample_specific_config) # Re-init with specific config

                # --- Dispatch to pipeline methods ---
                if pipeline_type == 'simple':
                    if not iqid_sample_info or 'sample_dir' not in iqid_sample_info:
                        raise ValueError(f"SimplePipeline for {current_sample_id} requires 'sample_dir' in iQID sample data.")

                    sample_dir_to_process = str(iqid_sample_info['sample_dir'])
                    forced_stage_for_pipeline = stage_arg if stage_arg != 'auto' else None

                    if dataset_type == "production":
                        if forced_stage_for_pipeline and forced_stage_for_pipeline != "aligned":
                            self.logger.warning(f"SimplePipeline on DataPush1 (production) data for {current_sample_id}. "
                                                f"Stage '{forced_stage_for_pipeline}' was forced. SimplePipeline will process as 'aligned' if data structure matches.")
                        # SimplePipeline's internal logic handles 'aligned' stage by returning early or processing if structure allows.
                        # If user forces 'raw' or 'segmented' on production data, SimplePipeline will attempt it if dirs exist.

                    result = pipeline_instance.process_iqid_stack(
                        sample_dir_str=sample_dir_to_process,
                        output_dir_str=current_sample_output_str, # Pass sample-specific output
                        forced_stage=forced_stage_for_pipeline
                    )

                elif pipeline_type == 'advanced':
                    if not iqid_sample_info or not iqid_sample_info.get('slice_files'):
                        raise ValueError(f"AdvancedPipeline for {current_sample_id} requires 'slice_files' in iQID sample data.")

                    mid_idx = len(iqid_sample_info['slice_files']) // 2
                    representative_slice = str(iqid_sample_info['slice_files'][mid_idx])
                    self.logger.info(f"Using representative slice for AdvancedPipeline: {representative_slice} for {current_sample_id}")
                    result = pipeline_instance.process_image(
                        image_path=representative_slice,
                        output_dir=current_sample_output_str
                    )

                elif pipeline_type == 'combined':
                    if not iqid_sample_info or not he_sample_info: # he_sample_info must exist for combined
                        self.logger.warning(f"Combined pipeline selected for {current_sample_id}, but H&E data is missing for this pair. Skipping.")
                        self.results['processed'] += 1
                        self.results['errors'].append(f"Skipped combined processing for {current_sample_id} due to missing H&E data.")
                        self.results['failed'] +=1
                        continue

                    if not iqid_sample_info.get('slice_files') or not he_sample_info.get('slice_files'):
                        raise ValueError(f"Missing slice_files for iQID or H&E in CombinedPipeline for {current_sample_id}")

                    iqid_mid_idx = len(iqid_sample_info['slice_files']) // 2
                    he_mid_idx = len(he_sample_info['slice_files']) // 2
                    iqid_slice_path = str(iqid_sample_info['slice_files'][iqid_mid_idx])
                    he_slice_path = str(he_sample_info['slice_files'][he_mid_idx])

                    self.logger.info(f"Using iQID slice: {iqid_slice_path} and H&E slice: {he_slice_path} for CombinedPipeline on {current_sample_id}.")
                    result = pipeline_instance.process_image_pair(
                        he_path=he_slice_path,
                        iqid_path=iqid_slice_path,
                        output_dir=current_sample_output_str
                    )
                else:
                    raise ValueError(f"Unknown pipeline type '{pipeline_type}' encountered in dispatch logic.")

                # Check pipeline result status
                if result and result.get('status') == 'failed':
                    self.results['failed'] += 1
                    error_detail = result.get('error', 'Pipeline reported failure')
                    error_msg_summary = f"Pipeline processing failed for {current_sample_id}: {error_detail}"
                    self.results['errors'].append(error_msg_summary)
                    self.logger.error(f"❌ {error_msg_summary}")
                else:
                    self.results['success'] += 1
                    self.logger.info(f"✅ Successfully processed {current_sample_id}")
                
            except Exception as e:
                # self.results['processed'] was already incremented
                self.results['failed'] += 1 # Catch exceptions from CLI logic or unhandled pipeline errors
                error_msg = f"Critical error processing {current_sample_id}: {str(e)}"
                self.results['errors'].append(error_msg)
                self.logger.error(f"❌ {error_msg}")
                self.logger.debug(traceback.format_exc())
        
        # Generate summary
        self._generate_summary()
        
        return self.results

    # Note: The second _analyze_sample_directory method (previously around line 370-397)
    # will be removed as its functionality (laterality extraction) is merged into the
    # primary _analyze_sample_directory and the new helpers.
    # The _extract_tissue_type, _extract_preprocessing_type, _extract_laterality helpers
    # that were part of that second method are now defined above.
    
    def _generate_summary(self):
        """Generate processing summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.results['summary'] = {
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'success_rate': (self.results['success'] / self.results['processed'] * 100) 
                           if self.results['processed'] > 0 else 0
        }
        
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 BATCH PROCESSING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total processed: {self.results['processed']}")
        self.logger.info(f"Successful: {self.results['success']}")
        self.logger.info(f"Failed: {self.results['failed']}")
        self.logger.info(f"Success rate: {self.results['summary']['success_rate']:.1f}%")
        self.logger.info(f"Duration: {duration.total_seconds():.1f} seconds")
        
        if self.results['errors']:
            self.logger.info(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                self.logger.info(f"  - {error}")
    
    # This old _analyze_sample_directory method is now removed.
    # Its functionality was merged into the primary _analyze_sample_directory (lines 170-241)
    # and the helper methods below.

    def _extract_tissue_type(self, sample_path: Path) -> str:
        """Extract tissue type from path hierarchy."""
        path_parts = sample_path.parts
        for part in path_parts:
            if part.lower() in ['kidney', 'tumor', 'kidneys']:
                return part.lower()
        return 'unknown'
    
    def _extract_preprocessing_type(self, sample_path: Path) -> str:
        """Extract preprocessing type from path hierarchy."""
        path_parts = sample_path.parts
        for part in path_parts:
            if part.lower() in ['3d', 'sequential', 'upper and lower']:
                return part.lower()
        return 'unknown'
    
    def _extract_laterality(self, sample_name: str) -> str:
        """Extract laterality (left/right) from sample name."""
        if sample_name.endswith('_L') or 'Left' in sample_name:
            return 'left'
        elif sample_name.endswith('_R') or 'Right' in sample_name:
            return 'right'
        return 'unknown'

# Removing the second, now redundant, _analyze_sample_directory method.
# The primary _analyze_sample_directory (used by discover_data) will be refactored in the next step.
# The helper methods _extract_tissue_type, _extract_preprocessing_type, _extract_laterality
# which were conceptually part of the removed method are now standalone helpers in the class.

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='IQID-Alphas Advanced Batch Processing CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic batch processing
  python -m iqid_alphas.cli process --data /path/to/data --config configs/simple.json
  
  # Advanced pipeline with custom output
  python -m iqid_alphas.cli process --data /path/to/data --config configs/advanced.json \\
                                   --pipeline advanced --output results/batch_1
  
  # Quick test with limited samples
  python -m iqid_alphas.cli process --data /path/to/data --config configs/test.json \\
                                   --max-samples 5 --verbose
  
  # Discover available data
  python -m iqid_alphas.cli discover --data /path/to/data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process batch of samples')
    process_parser.add_argument('--data', required=True, 
                              help='Path to data directory')
    process_parser.add_argument('--config', required=True,
                              help='Path to configuration file')
    process_parser.add_argument('--pipeline', choices=['simple', 'advanced', 'combined'],
                              default='simple', help='Pipeline type to use')
    process_parser.add_argument('--output', help='Output directory for results')
    process_parser.add_argument('--max-samples', type=int,
                              help='Maximum number of samples to process')
    process_parser.add_argument('--verbose', action='store_true',
                              help='Enable verbose logging')
    process_parser.add_argument(
        '--stage',
        type=str,
        default='auto', # Default to auto-detection
        choices=['auto', 'raw', 'segmented', 'aligned'],
        help='Specify the processing stage for ReUpload type datasets. (default: auto)'
    )
    
    # Discover command
    discover_parser = subparsers.add_parser('discover', help='Discover available data files')
    discover_parser.add_argument('--data', required=True,
                               help='Path to data directory')
    discover_parser.add_argument('--output', help='Save discovery results to JSON file')
    discover_parser.add_argument('--verbose', action='store_true', # Added verbose to discover
                              help='Enable verbose logging')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create', help='Create default configuration file')
    config_parser.add_argument('--validate', help='Validate configuration file')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    cli_processor = IQIDCLIProcessor()
    
    try:
        if args.command == 'process':
            results = cli_processor.process_batch(
                config_path=args.config,
                data_path=args.data,
                pipeline_type=args.pipeline,
                stage_arg=args.stage, # Pass stage argument
                max_samples=args.max_samples,
                output_dir=args.output
            )
            
            # Save results if output directory specified
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                results_file = output_path / 'batch_results.json'
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                cli_processor.logger.info(f"Results saved to: {results_file}")
        
        elif args.command == 'discover':
            discovered_data = cli_processor.discover_data(args.data)
            dataset_type = discovered_data.get('dataset_info', {}).get('type', 'unknown')

            # Header
            if dataset_type == "production":
                print("\n📁 DATA DISCOVERY RESULTS (DataPush1 - Production Dataset)")
            elif dataset_type == "workflow":
                print("\n📁 DATA DISCOVERY RESULTS (ReUpload - Workflow Dataset)")
            else:
                print("\n📁 DATA DISCOVERY RESULTS (Unknown Dataset Type)")
            print("="*50) # Adjusted length for potentially longer header

            # Summary Stats
            num_iqid_samples = len(discovered_data['iqid_samples'])
            num_he_samples = len(discovered_data['he_samples'])
            num_paired_samples = len(discovered_data['paired_samples'])

            if dataset_type == "production":
                # Count iQID samples ready for 3D based on can_reconstruct_3d and being an iQID modality
                iqid_3d_ready = sum(1 for s in discovered_data['iqid_samples'] if s.get('can_reconstruct_3d') and s.get('data_modality') == 'iqid')
                # H&E samples are assumed 'ready' if they exist in production data
                he_ready = num_he_samples
                # Paired samples ready for multi-modal are those successfully paired
                paired_multimodal_ready = num_paired_samples
                
                print(f"iQID samples (aligned, 3D ready): {iqid_3d_ready}")
                print(f"H&E samples (aligned, ready): {he_ready}")
                print(f"Paired samples (multi-modal ready): {paired_multimodal_ready}")

                print("\n🔬 Processing Status:")
                print(f"   - Dataset type: Production (aligned data)")
                all_iqid_ready_for_3d = all(s.get('can_reconstruct_3d') for s in discovered_data['iqid_samples']) if discovered_data['iqid_samples'] else True
                if all_iqid_ready_for_3d: # Simplified check
                    print(f"   - All iQID samples ready for 3D reconstruction")
                else:
                    print(f"   - Some iQID samples may require review for 3D reconstruction readiness.")

                multi_modal_available = num_paired_samples > 0
                if multi_modal_available:
                    print(f"   - Multi-modal analysis available (iQID + H&E)")
                else:
                    print(f"   - Multi-modal analysis (iQID + H&E) not available (no paired samples).")

                print("\n🧬 Tissue Distribution (iQID samples):")
                tissue_counts = {}
                for sample in discovered_data['iqid_samples']:
                    tt = sample.get('tissue_type', 'unknown')
                    tissue_counts[tt] = tissue_counts.get(tt, 0) + 1
                if tissue_counts:
                    for tissue, count in tissue_counts.items():
                        print(f"   - {tissue}: {count} samples")
                else:
                    print("   - No iQID tissue types identified.")

            elif dataset_type == "workflow":
                print(f"iQID samples (workflow stages): {num_iqid_samples}")
                print(f"H&E samples: {num_he_samples} (not typically available in this dataset)")
                print(f"Paired samples: {num_paired_samples} (single-modal dataset)")

                print("\n🔬 Processing Stage Analysis (iQID samples):")
                stage_counts = {'raw': 0, 'segmented': 0, 'aligned': 0, 'unknown_stage': 0}
                samples_ready_for_3d = 0
                for sample in discovered_data['iqid_samples']:
                    # Use 'processing_stage' which reflects the most advanced stage of the sample dir itself
                    # or primary stage if sub-stages are not the main representation for this sample in iqid_samples list
                    stage = sample.get('processing_stage', 'unknown_stage')
                    if stage == "raw_available": stage = "raw" # Normalize from _detect_processing_stage
                    if stage == "segmented_available": stage = "segmented"
                    if stage == "aligned_available" or stage == "slices_available": stage = "aligned" # Map to aligned

                    if stage in stage_counts:
                        stage_counts[stage] += 1
                    else:
                        stage_counts['unknown_stage'] +=1 # Should ideally not happen with new logic

                    if sample.get('can_reconstruct_3d'): # or stage == "aligned"
                        samples_ready_for_3d += 1
                
                print(f"   - Raw stage available: {stage_counts['raw']} samples")
                print(f"   - Segmented stage available: {stage_counts['segmented']} samples")
                print(f"   - Aligned stage available: {stage_counts['aligned']} samples")
                if stage_counts['unknown_stage'] > 0:
                     print(f"   - Unknown stage: {stage_counts['unknown_stage']} samples")
                print(f"   - Ready for 3D reconstruction: {samples_ready_for_3d} samples")

                print("\n🧬 Workflow Opportunities (iQID samples):")
                # This requires knowing the full desired workflow path.
                # Example: if a sample's latest stage is 'raw', it needs segmentation.
                # If 'segmented', it needs alignment.
                needs_segmentation = stage_counts['raw']
                needs_alignment = stage_counts['segmented']
                print(f"   - ~{needs_segmentation} samples need segmentation (raw → segmented)")
                print(f"   - ~{needs_alignment} samples need alignment (segmented → aligned)")
                # Note: This is a simplified view. A sample might have 'raw' but also 'segmented'.
                # The 'available_stages' field in sample_info would be better for a more precise count.
                # For now, this matches the spirit of the spec's output.
            
            else: # Unknown dataset type
                print(f"iQID Samples: {num_iqid_samples}")
                print(f"H&E Samples: {num_he_samples}")
                print(f"Paired Samples: {num_paired_samples}")
                print("\n🔬 Processing Status: Dataset type is unknown, cannot provide detailed status.")

            if args.output:
                # Save the raw discovered_data dictionary
                with open(args.output, 'w') as f:
                    json.dump(discovered, f, indent=2, default=str)
                print(f"\nResults saved to: {args.output}")
        
        elif args.command == 'config':
            if args.create:
                create_default_config(args.create)
            elif args.validate:
                validate_config(args.validate)
    
    except KeyboardInterrupt:
        cli_processor.logger.info("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        cli_processor.logger.error(f"❌ CLI Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            cli_processor.logger.error(traceback.format_exc())
        sys.exit(1)


def create_default_config(config_path: str):
    """Create a default configuration file."""
    default_config = {
        "processing": {
            "normalize": True,
            "remove_outliers": True,
            "gaussian_filter": True,
            "filter_sigma": 1.0
        },
        "segmentation": {
            "method": "otsu",
            "min_size": 100,
            "remove_small_objects": True
        },
        "alignment": {
            "method": "phase_correlation",
            "max_shift": 50,
            "subpixel_precision": True
        },
        "visualization": {
            "save_plots": True,
            "output_dir": "results",
            "dpi": 300,
            "format": "png"
        },
        "quality_control": {
            "enable_validation": True,
            "min_alignment_score": 0.5,
            "max_processing_time": 300
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✅ Default configuration created: {config_path}")


def validate_config(config_path: str):
    """Validate a configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_sections = ['processing', 'segmentation', 'alignment', 'visualization']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            print(f"❌ Missing required sections: {missing_sections}")
            return False
        
        print(f"✅ Configuration file is valid: {config_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return False


if __name__ == '__main__':
    main()
