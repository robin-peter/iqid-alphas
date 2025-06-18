import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import logging # Added: import logging
import sys # Added: import sys for StreamHandler

from ..core.processor import IQIDProcessor
from ..core.segmentation import ImageSegmenter
from ..visualization.plotter import Visualizer


class AdvancedPipeline:
    """
    Advanced pipeline for comprehensive iQID processing.
    
    Features:
    - Advanced preprocessing options
    - Multiple segmentation methods
    - Quantitative analysis with quality metrics
    - Advanced visualization
    - Comprehensive reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced pipeline.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary
        """
        self.config = config or self._default_config()
        self.processor = IQIDProcessor()
        self.segmenter = ImageSegmenter()
        self.visualizer = Visualizer()
        self.logger = self._setup_logger() # Added logger

    def _setup_logger(self) -> logging.Logger:
        # Basic logger for pipeline, can be expanded
        # import logging # No longer needed here, moved to top-level
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers: # Avoid duplicate handlers if already configured
            logger.setLevel(logging.INFO) # Now logging.INFO is accessible
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced pipeline."""
        return {
            'processing': {
                'gaussian_blur_sigma': 1.0,
                'normalize': True,
                'enhance_contrast': True,
                'background_subtraction': True
            },
            'segmentation': {
                'tissue_method': 'adaptive',
                'activity_method': 'otsu',
                'min_tissue_area': 500,
                'min_activity_area': 50,
                'morphological_cleanup': True
            },
            'analysis': {
                'calculate_statistics': True,
                'generate_profiles': True,
                'quality_assessment': True
            },
            'output': {
                'save_all_intermediates': True,
                'create_comprehensive_plots': True,
                'generate_report': True
            }
        }
    
    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process image through advanced pipeline.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        output_dir : str
            Output directory
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive processing results
        """
        self.logger.info(f"Advanced processing: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'input_file': image_path,
            'output_dir': output_dir,
            'pipeline': 'advanced',
            'config': self.config,
            'status': 'success', # Default to success
            'error': None
        }
        
        try:
            # Step 1: Load and advanced preprocessing
            image = self.processor.load_image(image_path)
            
            # Advanced preprocessing
            processed_image = self.processor.preprocess_image(
                image, 
                **self.config.get('processing', self.config.get('preprocessing', {})) # Use 'processing' or fallback to 'preprocessing'
            )
            
            results['preprocessing'] = {
                'original_shape': image.shape,
                'processed_shape': processed_image.shape,
                'status': 'success'
            }
            
            # Step 2: Advanced segmentation
            segmentation_config = self.config.get('segmentation', {})
            tissue_method = segmentation_config.get('tissue_method', 'adaptive') # Default from _default_config
            activity_method = segmentation_config.get('activity_method', 'otsu') # Default from _default_config
            min_tissue_area = segmentation_config.get('min_tissue_area', 500) # Default
            min_activity_area = segmentation_config.get('min_activity_area', 50) # Default
            morphological_cleanup = segmentation_config.get('morphological_cleanup', True) # Default

            tissue_segmenter = ImageSegmenter(tissue_method)
            activity_segmenter = ImageSegmenter(activity_method)
            
            tissue_mask = tissue_segmenter.segment_tissue(
                processed_image,
                min_size=min_tissue_area,
                morphological_closing=morphological_cleanup # Pass relevant cleanup args
            )
            
            activity_mask = activity_segmenter.segment_activity(
                processed_image,
                min_activity_size=min_activity_area,
                morphological_closing=morphological_cleanup # Pass relevant cleanup args
            )
            
            results['segmentation'] = {
                'tissue_area': int(np.sum(tissue_mask)),
                'activity_area': int(np.sum(activity_mask)),
                'overlap_area': int(np.sum(tissue_mask & activity_mask)),
                'status': 'success'
            }
            
            # Step 3: Comprehensive analysis
            whole_image_analysis = self.processor.analyze_image(processed_image)
            tissue_analysis = self.segmenter.analyze_segments(processed_image, tissue_mask)
            activity_analysis = self.segmenter.analyze_segments(processed_image, activity_mask)
            
            # Quality assessment
            quality_metrics = self._assess_quality(
                processed_image, tissue_mask, activity_mask
            )
            
            results['analysis'] = {
                'whole_image': whole_image_analysis,
                'tissue_regions': tissue_analysis,
                'activity_regions': activity_analysis,
                'quality_metrics': quality_metrics
            }
            
            # Step 4: Advanced visualization
            output_specific_config = self.config.get('output', {})
            # visualization_config = self.config.get('visualization', {}) # output_dir is now in output_path

            if output_specific_config.get('create_comprehensive_plots', True):
                plot_paths = []
                base_plot_name = Path(image_path).stem
                try:
                    # 1. Processed image activity map
                    self.visualizer.plot_activity_map(
                        processed_image,
                        title=f"Processed Image - {base_plot_name}"
                    )
                    proc_plot_file = output_path / f"{base_plot_name}_processed_map.png"
                    self.visualizer.save_figure(str(proc_plot_file))
                    plot_paths.append(str(proc_plot_file))
                    self.visualizer.close()

                    # 2. Tissue mask overlay
                    self.visualizer.plot_segmentation_overlay(
                        processed_image,
                        tissue_mask,
                        alpha=0.5,
                        title=f"Tissue Mask Overlay - {base_plot_name}"
                    )
                    tissue_plot_file = output_path / f"{base_plot_name}_tissue_overlay.png"
                    self.visualizer.save_figure(str(tissue_plot_file))
                    plot_paths.append(str(tissue_plot_file))
                    self.visualizer.close()

                    # 3. Activity mask overlay
                    self.visualizer.plot_segmentation_overlay(
                        processed_image,
                        activity_mask,
                        alpha=0.5,
                        title=f"Activity Mask Overlay - {base_plot_name}"
                    )
                    activity_plot_file = output_path / f"{base_plot_name}_activity_overlay.png"
                    self.visualizer.save_figure(str(activity_plot_file))
                    plot_paths.append(str(activity_plot_file))
                    self.visualizer.close()

                    results['visualization'] = {'plots': plot_paths}
                except Exception as e_vis:
                    self.logger.warning(f"Visualization failed in AdvancedPipeline: {e_vis}")
                    results['visualization'] = {'error': str(e_vis)}
            
            # Step 5: Generate comprehensive report
            if output_specific_config.get('generate_report', True): # Default true
                report = self._generate_report(results)
                report_file = output_path / f"{Path(image_path).stem}_report.html"
                with open(report_file, 'w') as f:
                    f.write(report)
                results['report_file'] = str(report_file)
            
            # Step 6: Save all results
            results_file = output_path / f"{Path(image_path).stem}_advanced_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"✓ Advanced processing complete: {Path(image_path).name}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"✗ Advanced processing failed: {Path(image_path).name} - {e}")
        
        return results
    
    def _assess_quality(self, image: np.ndarray, tissue_mask: np.ndarray, 
                       activity_mask: np.ndarray) -> Dict[str, float]:
        """Assess image and segmentation quality."""
        quality = {}
        
        # Image quality metrics
        quality['image_contrast'] = float(np.std(image))
        quality['image_snr'] = float(np.mean(image) / np.std(image)) if np.std(image) > 0 else 0.0
        
        # Segmentation quality metrics
        total_pixels = image.size
        quality['tissue_coverage'] = float(np.sum(tissue_mask) / total_pixels)
        quality['activity_coverage'] = float(np.sum(activity_mask) / total_pixels)
        
        if np.sum(tissue_mask) > 0 and np.sum(activity_mask) > 0:
            quality['activity_tissue_overlap'] = float(
                np.sum(tissue_mask & activity_mask) / np.sum(activity_mask)
            )
        else:
            quality['activity_tissue_overlap'] = 0.0
        
        # Overall quality score
        quality['overall_score'] = float(
            0.3 * min(quality['image_contrast'] / 100, 1.0) +
            0.3 * min(quality['tissue_coverage'] * 2, 1.0) +
            0.4 * quality['activity_tissue_overlap']
        )
        
        return quality
    
    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced iQID Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background: #f9f9f9; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Advanced iQID Analysis Report</h1>
                <p><strong>File:</strong> {results.get('input_file', 'Unknown')}</p>
                <p><strong>Status:</strong> {results.get('status', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Segmentation Results</h2>
                <div class="metric">Tissue Area: {results.get('segmentation', {}).get('tissue_area', 0)} pixels</div>
                <div class="metric">Activity Area: {results.get('segmentation', {}).get('activity_area', 0)} pixels</div>
                <div class="metric">Overlap Area: {results.get('segmentation', {}).get('overlap_area', 0)} pixels</div>
            </div>
            
            <div class="section">
                <h2>Quality Metrics</h2>
                {self._format_quality_metrics(results.get('analysis', {}).get('quality_metrics', {}))}
            </div>
            
            <div class="section">
                <h2>Analysis Summary</h2>
                {self._format_analysis_summary(results.get('analysis', {}))}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_quality_metrics(self, metrics: Dict[str, float]) -> str:
        """Format quality metrics for HTML report."""
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        for key, value in metrics.items():
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.3f}</td></tr>"
        html += "</table>"
        return html
    
    def _format_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Format analysis summary for HTML report."""
        html = "<table>"
        html += "<tr><th>Region</th><th>Mean Intensity</th><th>Total Intensity</th><th>Area</th></tr>"
        
        regions = ['whole_image', 'tissue_regions', 'activity_regions']
        for region in regions:
            if region in analysis:
                data = analysis[region]
                html += f"""
                <tr>
                    <td>{region.replace('_', ' ').title()}</td>
                    <td>{data.get('mean_intensity', 0):.2f}</td>
                    <td>{data.get('total_intensity', 0):.2f}</td>
                    <td>{data.get('area', 0)}</td>
                </tr>
                """
        
        html += "</table>"
        return html


def run_advanced_pipeline(image_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick function to run advanced pipeline.
    
    Parameters
    ----------
    image_path : str
        Path to image
    output_dir : str
        Output directory
    config : dict, optional
        Configuration
        
    Returns
    -------
    Dict[str, Any]
        Results
    """
    pipeline = AdvancedPipeline(config)
    return pipeline.process_image(image_path, output_dir)
