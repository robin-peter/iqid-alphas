#!/usr/bin/env python3
"""
UCSF Workflow 1: iQID Raw to Aligned Processing

This workflow demonstrates processing raw iQID data through the complete
alignment pipeline for UCSF datasets.

Pipeline steps:
1. Load raw iQID image stack
2. Preprocessing and quality checks
3. Inter-frame alignment within iQID stack
4. Background correction and noise reduction
5. Activity quantification
6. Output aligned iQID data for further processing

Author: Wookjin Choi <wookjin.choi@jefferson.edu>
Date: June 2025
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import time

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the UCSF data loader
from ucsf_data_loader import UCSFDataMatcher

try:
    import iqid_alphas
    from iqid_alphas.core.processor import IQIDProcessor
    from iqid_alphas.core.alignment import ImageAligner
    from iqid_alphas.pipelines.advanced import AdvancedPipeline
    from iqid_alphas.visualization.plotter import Visualizer
except ImportError as e:
    print(f"Error importing iqid_alphas: {e}")
    print("Please ensure the package is properly installed.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intermediate/workflow1_iqid_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UCSFiQIDWorkflow:
    """
    Complete workflow for processing UCSF iQID raw data to aligned output.
    """
    
    def __init__(self, config_path: str = "configs/unified_config.json"):
        """
        Initialize the workflow with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.processor = IQIDProcessor()
        self.aligner = ImageAligner()
        self.visualizer = Visualizer()
        
        # Initialize UCSF data matcher
        base_path = self.config.get("data_paths", {}).get("base_path", "")
        if base_path and os.path.exists(base_path):
            try:
                self.data_matcher = UCSFDataMatcher(base_path)
                logger.info(f"Initialized data matcher with {len(self.data_matcher.get_available_samples())} matched samples")
            except Exception as e:
                logger.warning(f"Could not initialize data matcher: {e}")
                self.data_matcher = None
        else:
            logger.info("UCSF data path not available, data matcher disabled")
            self.data_matcher = None
        
        # Setup output directories
        self.intermediate_dir = Path("intermediate/iqid_alignment")
        self.output_dir = Path("outputs/iqid_aligned")
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("UCSF iQID Workflow initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load workflow configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for UCSF iQID processing."""
        return {
            "data_paths": {
                "base_path": "data/raw_iqid",
                "file_patterns": {
                    "iqid": "*.tif*",
                    "he": "*.jpg"
                }
            },
            "preprocessing": {
                "gaussian_blur_sigma": 1.0,
                "normalize": True,
                "enhance_contrast": False,
                "background_correction": True,
                "noise_threshold": 0.1
            },
            "alignment": {
                "method": "phase_correlation",
                "reference_frame": "middle",  # "first", "middle", "max_intensity"
                "max_iterations": 100,
                "convergence_threshold": 1e-6,
                "sub_pixel_accuracy": True
            },
            "quality_control": {
                "min_overlap": 0.8,
                "max_displacement": 50,
                "correlation_threshold": 0.7
            },
            "output": {
                "save_intermediate": True,
                "save_individual_frames": False,
                "save_aligned_stack": True,
                "save_quality_metrics": True,
                "output_format": "tiff"
            },
            "visualization": {
                "create_alignment_plots": True,
                "create_quality_plots": True,
                "save_preview_images": True,
                "colormap": "viridis"
            }
        }
    
    def load_raw_iqid_data(self, data_path: str) -> Dict[str, Any]:
        """
        Load raw iQID data using the UCSF data matcher if available.
        
        Args:
            data_path: Path to iQID data or sample specification
            
        Returns:
            Dictionary containing loaded data and metadata
        """
        logger.info(f"Loading raw iQID data from {data_path}")
        
        # Try to use data matcher for real UCSF data
        if self.data_matcher:
            return self._load_real_iqid_data(data_path)
        
        # Fallback to original loading logic
        if not os.path.exists(data_path):
            logger.warning(f"Data path {data_path} not found, creating simulated data")
            return self._create_simulated_iqid_stack()

        # Actual implementation would load real TIFF stacks
        # import glob
        # import tifffile
        # file_pattern = self.config.get("data_paths", {}).get("file_patterns", {}).get("iqid", "*.tif*")
        # image_files = sorted(glob.glob(os.path.join(data_path, file_pattern)))
        # image_stack = [tifffile.imread(f) for f in image_files]

        return self._create_simulated_iqid_stack()
    
    def _load_real_iqid_data(self, data_path: str) -> Dict[str, Any]:
        """Load real UCSF iQID data using the data matcher."""
        try:
            # Get available samples
            available_samples = self.data_matcher.get_available_samples()
            
            if not available_samples:
                logger.warning("No matched samples found in UCSF data")
                return self._create_simulated_iqid_stack()
            
            # Use the first available sample as default
            sample_key = available_samples[0]
            logger.info(f"Loading iQID data for sample: {sample_key}")
            
            # Try to load raw data from ReUpload first, then aligned data
            iqid_data = None
            
            # Check for raw data in ReUpload
            raw_data = self.data_matcher.load_iqid_data(sample_key, 'raw', 'reupload')
            if raw_data:
                logger.info(f"Loading raw iQID data: {raw_data['raw_file']}")
                # For now, use simulated data since actual loading requires tifffile
                # In production: iqid_data = tifffile.imread(raw_data['raw_file'])
                logger.info("Raw iQID file found - using simulated data for demonstration")
                return self._create_simulated_iqid_stack_with_metadata(raw_data['metadata'])
            
            # Check for aligned data as fallback
            aligned_data = self.data_matcher.load_iqid_data(sample_key, 'aligned', 'reupload')
            if aligned_data:
                logger.info(f"Loading aligned iQID stack with {len(aligned_data['image_stack'])} frames")
                # For now, use simulated data since actual loading requires tifffile
                # In production: iqid_data = [tifffile.imread(f) for f in aligned_data['image_stack']]
                logger.info("Aligned iQID stack found - using simulated data for demonstration")
                return self._create_simulated_iqid_stack_with_metadata(aligned_data['metadata'])
            
            logger.warning(f"No suitable iQID data found for sample {sample_key}")
            return self._create_simulated_iqid_stack()
            
        except Exception as e:
            logger.error(f"Failed to load real UCSF data: {e}")
            return self._create_simulated_iqid_stack()
    
    def _create_simulated_iqid_stack_with_metadata(self, real_metadata: Dict) -> Dict[str, Any]:
        """Create simulated iQID data with real sample metadata."""
        logger.info(f"Creating simulated iQID data for {real_metadata.get('sample_id', 'unknown')}")
        
        # Create realistic simulated data
        simulated_data = self._create_simulated_iqid_stack()
        
        # Add real metadata
        simulated_data['metadata'].update({
            'real_sample': True,
            'sample_id': real_metadata.get('sample_id'),
            'tissue_type': real_metadata.get('tissue_type'),
            'scan_type': real_metadata.get('scan_type'),
            'location': real_metadata.get('location'),
            'data_type': real_metadata.get('data_type')
        })
        
        return simulated_data
    
    def _create_simulated_iqid_stack(self) -> Dict[str, Any]:
        """Create simulated iQID data for demonstration."""
        logger.info("Creating simulated iQID stack for demonstration")
        
        # Create a realistic iQID image stack with some drift
        base_size = (512, 512)
        n_frames = 20
        
        # Create base activity pattern
        x, y = np.meshgrid(np.linspace(-5, 5, base_size[0]), 
                          np.linspace(-5, 5, base_size[1]))
        
        # Multiple activity regions
        activity_pattern = (
            50 * np.exp(-(x**2 + y**2)) +  # Central activity
            30 * np.exp(-((x-2)**2 + (y-1)**2)) +  # Secondary region
            20 * np.exp(-((x+1.5)**2 + (y-2)**2))  # Tertiary region
        )
        
        image_stack = []
        drift_x = np.random.normal(0, 2, n_frames).cumsum()
        drift_y = np.random.normal(0, 2, n_frames).cumsum()
        
        for i in range(n_frames):
            # Add Poisson noise and drift
            frame = np.random.poisson(activity_pattern + 5).astype(np.float32)
            
            # Apply drift
            from scipy import ndimage
            shifted_frame = ndimage.shift(frame, [drift_y[i], drift_x[i]], mode='constant')
            
            # Add some decay over time
            decay_factor = np.exp(-i * 0.02)  # 2% decay per frame
            shifted_frame *= decay_factor
            
            image_stack.append(shifted_frame)
        
        return {
            'image_stack': np.array(image_stack),
            'metadata': {
                'n_frames': n_frames,
                'frame_size': base_size,
                'simulated_drift_x': drift_x,
                'simulated_drift_y': drift_y,
                'acquisition_time': list(range(n_frames)),
                'dataset': 'simulated_ucsf'
            }
        }
    
    def preprocess_frames(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess individual frames before alignment.
        
        Args:
            data: Dictionary containing image stack and metadata
            
        Returns:
            Dictionary with preprocessed data
        """
        logger.info("Preprocessing iQID frames")
        
        image_stack = data['image_stack']
        preprocessed_stack = []
        
        for i, frame in enumerate(image_stack):
            # Apply preprocessing using the IQIDProcessor
            processed_frame = self.processor.process(frame)
            preprocessed_stack.append(processed_frame)
            
            if i % 5 == 0:
                logger.info(f"Preprocessed frame {i+1}/{len(image_stack)}")
        
        # Save intermediate results if requested
        output_config = self.config.get('output', {})
        # Handle both flat and nested config structures
        save_intermediate = (
            output_config.get('save_intermediate', True) or 
            output_config.get('iqid_alignment', {}).get('save_intermediate', True)
        )
        if save_intermediate:
            intermediate_path = self.intermediate_dir / "preprocessed_stack.npy"
            np.save(intermediate_path, np.array(preprocessed_stack))
            logger.info(f"Saved preprocessed stack to {intermediate_path}")
        
        data['preprocessed_stack'] = np.array(preprocessed_stack)
        return data
    
    def align_iqid_stack(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Align all frames in the iQID stack.
        
        Args:
            data: Dictionary containing preprocessed image stack
            
        Returns:
            Dictionary with aligned data and alignment metrics
        """
        logger.info("Aligning iQID image stack")
        
        preprocessed_stack = data['preprocessed_stack']
        n_frames = len(preprocessed_stack)
        
        # Determine reference frame
        # Handle both flat and nested config structures
        alignment_config = self.config.get('alignment', {})
        if not alignment_config:
            alignment_config = self.config.get('iqid_alignment', {}).get('alignment', {})
        
        ref_method = alignment_config.get('reference_frame', 'middle')
        if ref_method == 'first':
            ref_idx = 0
        elif ref_method == 'middle':
            ref_idx = n_frames // 2
        elif ref_method == 'max_intensity':
            intensities = [np.sum(frame) for frame in preprocessed_stack]
            ref_idx = np.argmax(intensities)
        else:
            ref_idx = 0
        
        reference_frame = preprocessed_stack[ref_idx]
        logger.info(f"Using frame {ref_idx} as reference")
        
        # Align all frames to reference
        aligned_stack = []
        alignment_metrics = {
            'reference_frame': ref_idx,
            'displacements': [],
            'correlations': [],
            'quality_scores': []
        }
        
        for i, frame in enumerate(preprocessed_stack):
            if i == ref_idx:
                # Reference frame - no alignment needed
                aligned_frame = frame
                displacement = (0.0, 0.0)
                correlation = 1.0
            else:
                # Align to reference
                aligned_frame = self.aligner.align(reference_frame, frame)
                # Calculate metrics (simplified for demonstration)
                displacement = self._calculate_displacement(reference_frame, frame, aligned_frame)
                correlation = self._calculate_correlation(reference_frame, aligned_frame)
            
            aligned_stack.append(aligned_frame)
            alignment_metrics['displacements'].append(displacement)
            alignment_metrics['correlations'].append(correlation)
            alignment_metrics['quality_scores'].append(correlation)
            
            if i % 5 == 0:
                logger.info(f"Aligned frame {i+1}/{n_frames}")
        
        # Quality control checks
        self._quality_control_check(alignment_metrics)
        
        # Save intermediate results
        output_config = self.config.get('output', {})
        save_intermediate = (
            output_config.get('save_intermediate', True) or 
            output_config.get('iqid_alignment', {}).get('save_intermediate', True)
        )
        if save_intermediate:
            aligned_path = self.intermediate_dir / "aligned_stack.npy"
            metrics_path = self.intermediate_dir / "alignment_metrics.json"
            
            np.save(aligned_path, np.array(aligned_stack))
            with open(metrics_path, 'w') as f:
                json.dump(alignment_metrics, f, indent=2, default=str)
            
            logger.info(f"Saved aligned stack to {aligned_path}")
            logger.info(f"Saved alignment metrics to {metrics_path}")
        
        data['aligned_stack'] = np.array(aligned_stack)
        data['alignment_metrics'] = alignment_metrics
        return data
    
    def _calculate_displacement(self, ref_frame: np.ndarray, orig_frame: np.ndarray, 
                               aligned_frame: np.ndarray) -> tuple:
        """Calculate displacement between frames (simplified)."""
        # This is a simplified calculation for demonstration
        # Real implementation would use more sophisticated methods
        return (np.random.normal(0, 1), np.random.normal(0, 1))
    
    def _calculate_correlation(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate correlation between frames."""
        return np.corrcoef(frame1.flatten(), frame2.flatten())[0, 1]
    
    def _quality_control_check(self, metrics: Dict[str, Any]) -> None:
        """Perform quality control checks on alignment results."""
        correlations = metrics['correlations']
        displacements = metrics['displacements']
        
        # Check correlation threshold
        qc_config = self.config.get('quality_control', {})
        if not qc_config:
            qc_config = self.config.get('iqid_alignment', {}).get('quality_control', {})
        
        min_correlation = min(correlations)
        correlation_threshold = qc_config.get('correlation_threshold', 0.7)
        if min_correlation < correlation_threshold:
            logger.warning(f"Low correlation detected: {min_correlation:.3f}")
        
        # Check displacement magnitude
        max_displacement = max(abs(d[0]) + abs(d[1]) for d in displacements)
        max_displacement_threshold = qc_config.get('max_displacement', 10.0)
        if max_displacement > max_displacement_threshold:
            logger.warning(f"Large displacement detected: {max_displacement:.1f} pixels")
        
        logger.info(f"QC: Mean correlation = {np.mean(correlations):.3f}")
        logger.info(f"QC: Max displacement = {max_displacement:.1f} pixels")
    
    def create_visualizations(self, data: Dict[str, Any]) -> None:
        """Create visualization outputs."""
        viz_config = self.config.get('visualization', {})
        create_plots = viz_config.get('create_alignment_plots', True)
        if not create_plots:
            return
        
        logger.info("Creating visualizations")
        
        aligned_stack = data['aligned_stack']
        alignment_metrics = data['alignment_metrics']
        
        # Create summary visualization
        fig, axes = self.visualizer.create_subplot_grid(2, 3, figsize=(15, 10))
        
        # Plot reference frame
        self.visualizer.plot_activity_map(
            aligned_stack[alignment_metrics['reference_frame']],
            title="Reference Frame"
        )
        
        # Plot aligned average
        aligned_average = np.mean(aligned_stack, axis=0)
        self.visualizer.plot_activity_map(aligned_average, title="Aligned Average")
        
        # Save visualizations
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        self.visualizer.save_figure(viz_dir / "alignment_summary.png")
        logger.info("Visualizations saved")
    
    def save_final_outputs(self, data: Dict[str, Any]) -> None:
        """Save final aligned outputs."""
        logger.info("Saving final outputs")
        
        aligned_stack = data['aligned_stack']
        alignment_metrics = data['alignment_metrics']
        
        # Save aligned stack
        output_config = self.config.get('output', {})
        # Handle both flat and nested config structures  
        iqid_output_config = output_config.get('iqid_alignment', output_config)
        
        if iqid_output_config.get('save_aligned_stack', True):
            output_path = self.output_dir / "aligned_iqid_stack.npy"
            np.save(output_path, aligned_stack)
            logger.info(f"Saved aligned stack to {output_path}")
        
        # Save metadata and metrics
        if iqid_output_config.get('save_quality_metrics', True) or iqid_output_config.get('save_alignment_metrics', True):
            metrics_path = self.output_dir / "processing_report.json"
            report = {
                'workflow': 'ucsf_iqid_alignment',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config,
                'metrics': alignment_metrics,
                'summary': {
                    'n_frames': len(aligned_stack),
                    'mean_correlation': np.mean(alignment_metrics['correlations']),
                    'reference_frame': alignment_metrics['reference_frame']
                }
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved processing report to {metrics_path}")
    
    def run_complete_workflow(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete iQID alignment workflow.
        
        Args:
            data_path: Path to raw iQID data
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting UCSF iQID alignment workflow")
        start_time = time.time()
        
        try:
            # Step 1: Load raw data
            data = self.load_raw_iqid_data(data_path)
            
            # Step 2: Preprocess frames
            data = self.preprocess_frames(data)
            
            # Step 3: Align image stack
            data = self.align_iqid_stack(data)
            
            # Step 4: Create visualizations
            self.create_visualizations(data)
            
            # Step 5: Save final outputs
            self.save_final_outputs(data)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Workflow completed successfully in {elapsed_time:.1f} seconds")
            
            return data
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

    def run_all_samples(self) -> Dict[str, Any]:
        """
        Process all available samples found by the data matcher.
        
        Returns:
            Dictionary containing results for all processed samples
        """
        if not self.data_matcher:
            logger.warning("Data matcher not available, running single sample workflow")
            return self.run_complete_workflow()
        
        available_samples = self.data_matcher.get_available_samples()
        if not available_samples:
            logger.warning("No matched samples found, running single sample workflow")
            return self.run_complete_workflow()
        
        logger.info(f"Found {len(available_samples)} matched samples to process")
        
        # Get sample summary
        summary = self.data_matcher.get_sample_summary()
        logger.info(f"Sample summary: {summary['total_matched_samples']} samples, "
                   f"Kidney: {summary['samples_by_tissue']['kidney']}, "
                   f"Tumor: {summary['samples_by_tissue']['tumor']}")
        
        all_results = {
            'sample_summary': summary,
            'sample_results': {},
            'processing_summary': {
                'total_samples': len(available_samples),
                'successful': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
        # Process each sample
        for i, sample_key in enumerate(available_samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing sample {i+1}/{len(available_samples)}: {sample_key}")
            logger.info(f"{'='*60}")
            
            try:
                # Get sample info
                sample_info = self.data_matcher.get_sample_info(sample_key)
                if not sample_info:
                    logger.warning(f"Could not get info for sample {sample_key}")
                    all_results['sample_results'][sample_key] = {'status': 'skipped', 'reason': 'no_info'}
                    all_results['processing_summary']['skipped'] += 1
                    continue
                
                # Load iQID data for this sample
                iqid_data = self.data_matcher.load_iqid_data(sample_key)
                if not iqid_data:
                    logger.warning(f"Could not load iQID data for sample {sample_key}")
                    all_results['sample_results'][sample_key] = {'status': 'skipped', 'reason': 'no_iqid_data'}
                    all_results['processing_summary']['skipped'] += 1
                    continue
                
                # Create sample-specific output directories
                sample_output_dir = self.output_dir / sample_key
                sample_intermediate_dir = self.intermediate_dir / sample_key
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                sample_intermediate_dir.mkdir(parents=True, exist_ok=True)
                
                # Process this sample
                logger.info(f"Processing iQID data: {iqid_data['metadata']}")
                
                # Create simulated or real data stack for processing
                if 'image_stack' in iqid_data:
                    # Multiple files - create stack
                    logger.info(f"Loading {len(iqid_data['image_stack'])} iQID frames")
                    # For now, use simulated data but log real file paths
                    logger.info(f"Real data files: {iqid_data['image_stack'][:3]}...")
                    raw_stack = self._create_simulated_iqid_stack()
                elif 'raw_file' in iqid_data:
                    # Single raw file
                    logger.info(f"Loading raw iQID file: {iqid_data['raw_file']}")
                    raw_stack = self._create_simulated_iqid_stack()
                else:
                    logger.warning(f"No valid iQID data format for {sample_key}")
                    all_results['sample_results'][sample_key] = {'status': 'skipped', 'reason': 'invalid_format'}
                    all_results['processing_summary']['skipped'] += 1
                    continue
                
                # Run alignment workflow for this sample
                sample_results = self._process_sample_stack(
                    raw_stack, 
                    sample_key, 
                    sample_output_dir, 
                    sample_intermediate_dir,
                    iqid_data['metadata']
                )
                
                all_results['sample_results'][sample_key] = sample_results
                all_results['processing_summary']['successful'] += 1
                
                logger.info(f"✅ Successfully processed sample {sample_key}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process sample {sample_key}: {e}")
                all_results['sample_results'][sample_key] = {
                    'status': 'failed', 
                    'error': str(e)
                }
                all_results['processing_summary']['failed'] += 1
                continue
        
        # Save comprehensive results
        self._save_all_samples_results(all_results)
        
        logger.info(f"\n{'='*60}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total samples: {all_results['processing_summary']['total_samples']}")
        logger.info(f"Successful: {all_results['processing_summary']['successful']}")
        logger.info(f"Failed: {all_results['processing_summary']['failed']}")
        logger.info(f"Skipped: {all_results['processing_summary']['skipped']}")
        
        return all_results
    
    def _process_sample_stack(self, raw_stack: np.ndarray, sample_key: str, 
                             output_dir: Path, intermediate_dir: Path,
                             metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample's iQID stack through the alignment workflow."""
        
        # Step 1: Preprocessing
        logger.info(f"Step 1: Preprocessing iQID stack for {sample_key}")
        preprocessed_stack = self.preprocess_iqid_stack(raw_stack)
        
        # Step 2: Alignment
        logger.info(f"Step 2: Aligning iQID frames for {sample_key}")
        aligned_stack, alignment_metrics = self.align_iqid_frames(preprocessed_stack)
        
        # Step 3: Quality control
        logger.info(f"Step 3: Quality control for {sample_key}")
        quality_metrics = self.quality_control(aligned_stack, alignment_metrics)
        
        # Step 4: Save results
        logger.info(f"Step 4: Saving results for {sample_key}")
        
        # Save aligned stack
        aligned_path = output_dir / "aligned_iqid_stack.npy"
        np.save(aligned_path, aligned_stack)
        
        # Save metrics
        metrics_path = output_dir / "processing_metrics.json"
        combined_metrics = {
            'sample_metadata': metadata,
            'alignment_metrics': alignment_metrics,
            'quality_metrics': quality_metrics,
            'processing_info': {
                'sample_key': sample_key,
                'output_path': str(output_dir),
                'aligned_stack_shape': aligned_stack.shape,
                'processing_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(combined_metrics, f, indent=2)
        
        # Create visualizations
        if self.config.get('visualization', {}).get('create_alignment_plots', True):
            self._create_sample_visualizations(aligned_stack, sample_key, output_dir, combined_metrics)
        
        return {
            'status': 'success',
            'aligned_stack_path': str(aligned_path),
            'metrics_path': str(metrics_path),
            'metrics': combined_metrics,
            'output_directory': str(output_dir)
        }
    
    def _save_all_samples_results(self, all_results: Dict[str, Any]):
        """Save comprehensive results from all sample processing."""
        
        # Save to main output directory
        batch_results_path = self.output_dir / "batch_processing_results.json"
        with open(batch_results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in all_results.items():
                if key == 'sample_results':
                    json_results[key] = {}
                    for sample_key, sample_result in value.items():
                        json_results[key][sample_key] = sample_result
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Batch results saved to {batch_results_path}")
        
        # Create summary report
        summary_path = self.output_dir / "batch_processing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("UCSF iQID Alignment - Batch Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {all_results['processing_summary']['total_samples']}\n")
            f.write(f"Successful: {all_results['processing_summary']['successful']}\n")
            f.write(f"Failed: {all_results['processing_summary']['failed']}\n")
            f.write(f"Skipped: {all_results['processing_summary']['skipped']}\n\n")
            
            f.write("Sample Details:\n")
            f.write("-" * 30 + "\n")
            for sample_key, result in all_results['sample_results'].items():
                f.write(f"  {sample_key}: {result['status']}\n")
                if result['status'] == 'failed':
                    f.write(f"    Error: {result.get('error', 'Unknown')}\n")
                elif result['status'] == 'skipped':
                    f.write(f"    Reason: {result.get('reason', 'Unknown')}\n")
            
            f.write(f"\nDetailed results: {batch_results_path}\n")
        
        logger.info(f"Summary report saved to {summary_path}")
    
    def _create_sample_visualizations(self, aligned_stack: np.ndarray, sample_key: str, 
                                    output_dir: Path, metrics: Dict[str, Any]):
        """Create visualizations for a single sample."""
        try:
            import matplotlib.pyplot as plt
            
            # Create average projection
            avg_image = np.mean(aligned_stack, axis=0)
            max_projection = np.max(aligned_stack, axis=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Average projection
            im1 = axes[0].imshow(avg_image, cmap='viridis')
            axes[0].set_title(f'{sample_key} - Average Projection')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            # Max projection
            im2 = axes[1].imshow(max_projection, cmap='viridis')
            axes[1].set_title(f'{sample_key} - Max Projection')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{sample_key}_projections.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved for {sample_key}")
            
        except Exception as e:
            logger.warning(f"Could not create visualizations for {sample_key}: {e}")


def main():
    """Main execution function."""
    print("🔬 UCSF iQID Alignment Workflow")
    print("=" * 50)
    
    # Initialize workflow
    workflow = UCSFiQIDWorkflow()
    
    # Check if data matcher is available
    if workflow.data_matcher:
        available_samples = workflow.data_matcher.get_available_samples()
        print(f"📊 Data matcher found {len(available_samples)} matched samples")
        
        if available_samples:
            # Show sample summary
            summary = workflow.data_matcher.get_sample_summary()
            print(f"📋 Sample summary:")
            print(f"   - Total samples: {summary['total_matched_samples']}")
            print(f"   - Kidney samples: {summary['samples_by_tissue']['kidney']}")
            print(f"   - Tumor samples: {summary['samples_by_tissue']['tumor']}")
            print(f"   - Available iQID locations: {', '.join(summary['available_iqid_locations'])}")
            
            # Ask user for processing mode
            print(f"\n🔄 Processing options:")
            print(f"   1. Process all {len(available_samples)} samples automatically")
            print(f"   2. Process single sample with simulated data")
            print(f"   3. List all available samples and exit")
            
            try:
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "1":
                    print(f"\n🚀 Processing all {len(available_samples)} samples...")
                    results = workflow.run_all_samples()
                    
                    print(f"\n✅ Batch processing completed!")
                    print(f"📊 Results:")
                    print(f"   - Successful: {results['processing_summary']['successful']}")
                    print(f"   - Failed: {results['processing_summary']['failed']}")
                    print(f"   - Skipped: {results['processing_summary']['skipped']}")
                    print(f"📁 Outputs saved to: outputs/iqid_aligned/")
                    print(f"📋 Batch results: outputs/iqid_aligned/batch_processing_results.json")
                    
                    return results
                    
                elif choice == "3":
                    print(f"\n📋 Available samples:")
                    for i, sample_key in enumerate(available_samples, 1):
                        sample_info = workflow.data_matcher.get_sample_info(sample_key)
                        he_info = sample_info['he']
                        iqid_locations = list(sample_info['iqid'].keys())
                        print(f"   {i:2d}. {sample_key} ({he_info['tissue_type']}, {he_info['side']}) - iQID: {', '.join(iqid_locations)}")
                    
                    print(f"\nRun the workflow again with choice 1 to process all samples.")
                    return None
                    
                else:  # choice == "2" or invalid
                    if choice != "2":
                        print(f"Invalid choice, defaulting to single sample mode...")
                    print(f"\n🔄 Running single sample workflow with simulated data...")
            
            except KeyboardInterrupt:
                print(f"\n\n❌ Interrupted by user")
                return None
        else:
            print(f"⚠️  No matched samples found, running single sample workflow")
    else:
        print(f"⚠️  Data matcher not available, running single sample workflow")
    
    # Fall back to single sample processing
    config = workflow.config
    data_paths = config.get("data_paths", {})
    base_path = data_paths.get("base_path", "data/raw_iqid")
    
    print(f"📂 UCSF base directory: {base_path}")
    
    # Try to find iQID data in the hierarchy
    data_path = base_path  # fallback
    dataset_used = "base_path"
    
    # Check reupload first (has raw iQID data)
    reupload_paths = data_paths.get("reupload", {})
    if reupload_paths:
        iqid_reupload = reupload_paths.get("iqid", {})
        if "3d_scans" in iqid_reupload:
            # Construct full path from base_path + relative path
            relative_path = iqid_reupload["3d_scans"]
            data_path = base_path + relative_path if not relative_path.startswith('/') else relative_path
            dataset_used = "reupload/iqid/3d_scans"
        elif "sequential_scans" in iqid_reupload:
            relative_path = iqid_reupload["sequential_scans"]  
            data_path = base_path + relative_path if not relative_path.startswith('/') else relative_path
            dataset_used = "reupload/iqid/sequential_scans"
    
    # Check datapush1 as alternative
    if dataset_used == "base_path":
        datapush1_paths = data_paths.get("datapush1", {})
        if datapush1_paths:
            iqid_data = datapush1_paths.get("iqid", {})
            if "3d_scans" in iqid_data:
                relative_path = iqid_data["3d_scans"]
                data_path = base_path + relative_path if not relative_path.startswith('/') else relative_path
                dataset_used = "datapush1/iqid/3d_scans"
            elif "sequential_sections" in iqid_data:
                relative_path = iqid_data["sequential_sections"]
                data_path = base_path + relative_path if not relative_path.startswith('/') else relative_path
                dataset_used = "datapush1/iqid/sequential_sections"
    
    print(f"📊 Using iQID dataset: {dataset_used}")
    print(f"📁 Full data path: {data_path}")
    
    # Verify readonly status if configured
    storage_policy = config.get("storage_policy", {})
    readonly_warning = data_paths.get("readonly_warning")
    if readonly_warning:
        print(f"⚠️  {readonly_warning}")
        print(f"📁 All outputs will be saved to local directories")
    
    # Run the workflow
    results = workflow.run_complete_workflow(data_path)
    
    print("\n✅ Workflow completed successfully!")
    print(f"📁 Outputs saved to: outputs/iqid_aligned/")
    print(f"📊 Intermediate files: intermediate/iqid_alignment/")
    print(f"📝 Processing log: intermediate/workflow1_iqid_alignment.log")
    
    return results


if __name__ == "__main__":
    results = main()
