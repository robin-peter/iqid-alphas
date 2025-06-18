import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging # Added: import logging
import sys # Added: import sys for StreamHandler

from ..core.processor import IQIDProcessor
from ..core.segmentation import ImageSegmenter
from ..core.alignment import ImageAligner
from ..visualization.plotter import Visualizer


class CombinedPipeline:
    """
    Combined pipeline for H&E and iQID image analysis.
    
    Features:
    - Dual image processing (H&E + iQID)
    - Image alignment and registration
    - Combined segmentation analysis
    - Cross-modal quantitative analysis
    - Comprehensive visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the combined pipeline.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or self._default_config()
        self.processor = IQIDProcessor()
        self.segmenter = ImageSegmenter()
        self.aligner = ImageAligner()
        self.visualizer = Visualizer()
        self.logger = self._setup_logger() # Added logger

    def _setup_logger(self) -> logging.Logger:
        # Basic logger for pipeline, can be expanded
        # import logging # No longer needed here
        # import sys # No longer needed here
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers: # Avoid duplicate handlers
            logger.setLevel(logging.INFO) # Now logging.INFO is accessible
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for combined pipeline."""
        return {
            'alignment': {
                'method': 'phase_correlation',
                'enable_quality_check': True
            },
            'segmentation': {
                'he_method': 'adaptive',
                'iqid_method': 'otsu',
                'combine_masks': True
            },
            'analysis': {
                'cross_modal_analysis': True,
                'spatial_correlation': True,
                'quantitative_metrics': True
            },
            'output': {
                'save_aligned_images': True,
                'create_overlay_plots': True,
                'generate_combined_report': True
            }
        }
    
    def process_image_pair(self, he_path: str, iqid_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a pair of H&E and iQID images.
        
        Parameters
        ----------
        he_path : str
            Path to H&E image
        iqid_path : str
            Path to iQID image
        output_dir : str
            Output directory
            
        Returns
        -------
        Dict[str, Any]
            Combined processing results
        """
        self.logger.info(f"Combined processing: H&E={Path(he_path).name}, iQID={Path(iqid_path).name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'he_file': he_path,
            'iqid_file': iqid_path,
            'output_dir': output_dir,
            'pipeline': 'combined',
            'status': 'success', # Default to success
            'error': None
        }
        
        try:
            # Step 1: Load both images
            he_image = self.processor.load_image(he_path, 'he')
            iqid_image = self.processor.load_image(iqid_path, 'iqid')
            
            # Preprocess both images
            he_processed = self.processor.preprocess_image(he_image)
            iqid_processed = self.processor.preprocess_image(iqid_image)
            
            results['preprocessing'] = {
                'he_shape': he_image.shape,
                'iqid_shape': iqid_image.shape,
                'status': 'success'
            }
            
            # Step 2: Align images
            aligned_iqid, alignment_info = self.aligner.align_images(he_processed, iqid_processed)
            alignment_quality = self.aligner.calculate_alignment_quality(he_processed, aligned_iqid)
            
            results['alignment'] = {
                'transformation': alignment_info,
                'quality': alignment_quality,
                'status': 'success'
            }
            
            # Step 3: Combined segmentation
            segmentation_results = self.segmenter.segment_combined(he_processed, aligned_iqid)
            
            tissue_mask = segmentation_results['tissue_mask']
            activity_mask = segmentation_results['activity_mask']
            combined_mask = segmentation_results.get('combined_mask', activity_mask)
            
            results['segmentation'] = {
                'tissue_area': int(np.sum(tissue_mask)),
                'activity_area': int(np.sum(activity_mask)),
                'combined_area': int(np.sum(combined_mask)),
                'status': 'success'
            }
            
            # Step 4: Cross-modal analysis
            cross_modal_analysis = self._perform_cross_modal_analysis(
                he_processed, aligned_iqid, tissue_mask, activity_mask, combined_mask
            )
            
            results['analysis'] = cross_modal_analysis
            
            # Step 5: Create combined visualizations
            output_config = self.config.get('output', {})
            # visualization_config = self.config.get('visualization', {}) # output_dir is now passed to save_figure

            if output_config.get('create_overlay_plots', True): # Default to True if not specified
                plot_paths = []
                try:
                    # Plot iQID with activity mask
                    fig_iqid, _ = self.visualizer.plot_segmentation_overlay(
                        image=aligned_iqid,
                        segments=activity_mask,
                        alpha=0.5,
                        title=f"iQID with Activity Mask - {Path(iqid_path).name}"
                    )
                    iqid_plot_file = output_path / f"{Path(iqid_path).stem}_activity_overlay.png"
                    self.visualizer.save_figure(str(iqid_plot_file)) # Uses self.visualizer.current_figure
                    plot_paths.append(str(iqid_plot_file))
                    self.visualizer.close() # Close figure

                    # Plot H&E with tissue mask
                    fig_he, _ = self.visualizer.plot_segmentation_overlay(
                        image=he_processed,
                        segments=tissue_mask,
                        alpha=0.5,
                        title=f"H&E with Tissue Mask - {Path(he_path).name}"
                    )
                    he_plot_file = output_path / f"{Path(he_path).stem}_tissue_overlay.png"
                    self.visualizer.save_figure(str(he_plot_file))
                    plot_paths.append(str(he_plot_file))
                    self.visualizer.close()

                    results['visualization'] = {'overlay_plots': plot_paths}
                except Exception as e_vis:
                    self.logger.warning(f"Visualization failed during CombinedPipeline: {e_vis}")
                    results['visualization'] = {'error': str(e_vis)}
            
            # Step 6: Save aligned images
            if output_config.get('save_aligned_images', True): # Default to True
                aligned_file = output_path / f"aligned_iqid.png" # output_path is sample specific from CLI
                try:
                    # Ensure aligned_iqid is in a savable format (e.g., convert float to uint16 if appropriate)
                    saveable_aligned_iqid = aligned_iqid.copy()
                    if np.issubdtype(saveable_aligned_iqid.dtype, np.floating):
                         saveable_aligned_iqid = (saveable_aligned_iqid / np.max(saveable_aligned_iqid) * 255).astype(np.uint8) # Example scaling
                    # import matplotlib.pyplot as plt # Not needed if using skimage.io.imsave
                    # plt.imsave(aligned_file, saveable_aligned_iqid, cmap='viridis')
                    skimage.io.imsave(str(aligned_file), saveable_aligned_iqid, plugin='tifffile', check_contrast=False)

                    results['aligned_iqid_file'] = str(aligned_file)
                except Exception as e_save_align:
                    self.logger.warning(f"Failed to save aligned iQID image: {e_save_align}")
                    if 'files_saved' not in results: results['files_saved'] = {}
                    results['files_saved']['aligned_iqid_error'] = str(e_save_align)
            
            # Step 7: Generate combined report
            if output_config.get('generate_combined_report', True): # Default to True
                report = self._generate_combined_report(results)
                report_file = output_path / "combined_report.html"
                with open(report_file, 'w') as f:
                    f.write(report)
                results['report_file'] = str(report_file)
            
            # Save results
            results_file = output_path / "combined_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info("✓ Combined processing complete")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"✗ Combined processing failed: {e}")
        
        return results
    
    def _perform_cross_modal_analysis(self, he_image: np.ndarray, iqid_image: np.ndarray,
                                    tissue_mask: np.ndarray, activity_mask: np.ndarray,
                                    combined_mask: np.ndarray) -> Dict[str, Any]:
        """Perform cross-modal analysis between H&E and iQID images."""
        analysis = {}
        
        # Individual image analysis
        he_analysis = self.processor.analyze_image(he_image)
        iqid_analysis = self.processor.analyze_image(iqid_image)
        
        # Tissue analysis from H&E
        tissue_analysis = self.segmenter.analyze_segments(he_image, tissue_mask)
        
        # Activity analysis from iQID
        activity_analysis = self.segmenter.analyze_segments(iqid_image, activity_mask)
        
        # Combined region analysis
        combined_analysis = self.segmenter.analyze_segments(iqid_image, combined_mask)
        
        analysis['individual'] = {
            'he_image': he_analysis,
            'iqid_image': iqid_analysis
        }
        
        analysis['regions'] = {
            'tissue': tissue_analysis,
            'activity': activity_analysis,
            'combined': combined_analysis
        }
        
        # Cross-modal metrics
        analysis['cross_modal'] = self._calculate_cross_modal_metrics(
            he_image, iqid_image, tissue_mask, activity_mask
        )
        
        return analysis
    
    def _calculate_cross_modal_metrics(self, he_image: np.ndarray, iqid_image: np.ndarray,
                                     tissue_mask: np.ndarray, activity_mask: np.ndarray) -> Dict[str, float]:
        """Calculate cross-modal correlation metrics."""
        metrics = {}
        
        # Ensure same shape for correlation analysis
        min_shape = [min(he_image.shape[i], iqid_image.shape[i]) for i in range(2)]
        he_crop = he_image[:min_shape[0], :min_shape[1]]
        iqid_crop = iqid_image[:min_shape[0], :min_shape[1]]
        tissue_crop = tissue_mask[:min_shape[0], :min_shape[1]]
        activity_crop = activity_mask[:min_shape[0], :min_shape[1]]
        
        # Convert to grayscale if needed
        if len(he_crop.shape) > 2:
            he_gray = np.mean(he_crop, axis=2)
        else:
            he_gray = he_crop
            
        if len(iqid_crop.shape) > 2:
            iqid_gray = np.mean(iqid_crop, axis=2)
        else:
            iqid_gray = iqid_crop
        
        # Overall correlation
        try:
            correlation = np.corrcoef(he_gray.flatten(), iqid_gray.flatten())[0, 1]
            metrics['overall_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            metrics['overall_correlation'] = 0.0
        
        # Tissue region correlation
        if np.sum(tissue_crop) > 0:
            try:
                he_tissue = he_gray[tissue_crop]
                iqid_tissue = iqid_gray[tissue_crop]
                tissue_corr = np.corrcoef(he_tissue, iqid_tissue)[0, 1]
                metrics['tissue_correlation'] = float(tissue_corr) if not np.isnan(tissue_corr) else 0.0
            except Exception:
                metrics['tissue_correlation'] = 0.0
        else:
            metrics['tissue_correlation'] = 0.0
        
        # Activity specificity
        if np.sum(activity_crop) > 0 and np.sum(tissue_crop) > 0:
            activity_in_tissue = np.sum(activity_crop & tissue_crop)
            activity_specificity = activity_in_tissue / np.sum(activity_crop)
            metrics['activity_specificity'] = float(activity_specificity)
        else:
            metrics['activity_specificity'] = 0.0
        
        return metrics
    
    def _generate_combined_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report for combined analysis."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Combined H&E + iQID Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background: #f9f9f9; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Combined H&E + iQID Analysis Report</h1>
                <p><strong>H&E File:</strong> {results.get('he_file', 'Unknown')}</p>
                <p><strong>iQID File:</strong> {results.get('iqid_file', 'Unknown')}</p>
                <p><strong>Status:</strong> <span class="{results.get('status', 'Unknown')}">{results.get('status', 'Unknown')}</span></p>
            </div>
            
            <div class="section">
                <h2>Alignment Results</h2>
                {self._format_alignment_results(results.get('alignment', {}))}
            </div>
            
            <div class="section">
                <h2>Segmentation Results</h2>
                {self._format_segmentation_results(results.get('segmentation', {}))}
            </div>
            
            <div class="section">
                <h2>Cross-Modal Analysis</h2>
                {self._format_cross_modal_results(results.get('analysis', {}))}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_alignment_results(self, alignment: Dict[str, Any]) -> str:
        """Format alignment results for HTML."""
        quality = alignment.get('quality', {})
        html = f"""
        <div class="metric">Alignment Quality Score: {quality.get('alignment_score', 0):.3f}</div>
        <div class="metric">Correlation: {quality.get('correlation', 0):.3f}</div>
        <div class="metric">MSE: {quality.get('mse', 0):.3f}</div>
        """
        return html
    
    def _format_segmentation_results(self, segmentation: Dict[str, Any]) -> str:
        """Format segmentation results for HTML."""
        html = f"""
        <div class="metric">Tissue Area: {segmentation.get('tissue_area', 0)} pixels</div>
        <div class="metric">Activity Area: {segmentation.get('activity_area', 0)} pixels</div>
        <div class="metric">Combined Area: {segmentation.get('combined_area', 0)} pixels</div>
        """
        return html
    
    def _format_cross_modal_results(self, analysis: Dict[str, Any]) -> str:
        """Format cross-modal analysis for HTML."""
        cross_modal = analysis.get('cross_modal', {})
        html = f"""
        <div class="metric">Overall Correlation: {cross_modal.get('overall_correlation', 0):.3f}</div>
        <div class="metric">Tissue Correlation: {cross_modal.get('tissue_correlation', 0):.3f}</div>
        <div class="metric">Activity Specificity: {cross_modal.get('activity_specificity', 0):.3f}</div>
        """
        return html


def run_combined_pipeline(he_path: str, iqid_path: str, output_dir: str, 
                         config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick function to run combined pipeline.
    
    Parameters
    ----------
    he_path : str
        Path to H&E image
    iqid_path : str
        Path to iQID image
    output_dir : str
        Output directory
    config : dict, optional
        Configuration
        
    Returns
    -------
    Dict[str, Any]
        Results
    """
    pipeline = CombinedPipeline(config)
    return pipeline.process_image_pair(he_path, iqid_path, output_dir)
