"""
Advanced Pipeline for Comprehensive iQID Analysis

Advanced pipeline with full feature set including detailed analysis,
quality metrics, and comprehensive reporting.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..core.processor import IQIDProcessor
from ..core.segmentation import ImageSegmenter
from ..core.alignment import ImageAligner
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
        
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced pipeline."""
        return {
            'preprocessing': {
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
        print(f"Advanced processing: {image_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'input_file': image_path,
            'output_dir': output_dir,
            'pipeline': 'advanced',
            'config': self.config
        }
        
        try:
            # Step 1: Load and advanced preprocessing
            image = self.processor.load_image(image_path)
            
            # Advanced preprocessing
            processed_image = self.processor.preprocess_image(
                image, 
                **self.config['preprocessing']
            )
            
            results['preprocessing'] = {
                'original_shape': image.shape,
                'processed_shape': processed_image.shape,
                'status': 'success'
            }
            
            # Step 2: Advanced segmentation
            tissue_segmenter = ImageSegmenter(self.config['segmentation']['tissue_method'])
            activity_segmenter = ImageSegmenter(self.config['segmentation']['activity_method'])
            
            tissue_mask = tissue_segmenter.segment_tissue(
                processed_image,
                min_size=self.config['segmentation']['min_tissue_area']
            )
            
            activity_mask = activity_segmenter.segment_activity(
                processed_image,
                min_activity_size=self.config['segmentation']['min_activity_area']
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
            if self.config['output']['create_comprehensive_plots']:
                # Create comprehensive visualization
                fig = self.visualizer.create_comprehensive_plot(
                    original_image=image,
                    processed_image=processed_image,
                    tissue_mask=tissue_mask,
                    activity_mask=activity_mask,
                    title=f"Advanced Analysis: {Path(image_path).name}"
                )
                
                plot_file = output_path / f"{Path(image_path).stem}_comprehensive.png"
                fig.savefig(plot_file, dpi=300, bbox_inches='tight')
                results['visualization'] = {'comprehensive_plot': str(plot_file)}
            
            # Step 5: Generate comprehensive report
            if self.config['output']['generate_report']:
                report = self._generate_report(results)
                report_file = output_path / f"{Path(image_path).stem}_report.html"
                with open(report_file, 'w') as f:
                    f.write(report)
                results['report_file'] = str(report_file)
            
            # Step 6: Save all results
            results_file = output_path / f"{Path(image_path).stem}_advanced_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            results['status'] = 'success'
            print(f"✓ Advanced processing complete: {Path(image_path).name}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"✗ Advanced processing failed: {Path(image_path).name} - {e}")
        
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
