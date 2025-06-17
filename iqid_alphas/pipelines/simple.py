"""
Simple Pipeline for Basic iQID Processing

A streamlined pipeline for basic iQID image analysis.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..core.processor import IQIDProcessor
from ..core.segmentation import ImageSegmenter
from ..visualization.plotter import Visualizer


class SimplePipeline:
    """
    Simple pipeline for basic iQID processing.
    
    This pipeline provides essential functionality:
    - Image loading and preprocessing
    - Basic segmentation
    - Quantitative analysis
    - Simple visualization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simple pipeline.
        
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
        """Get default configuration."""
        return {
            'processing': {
                'gaussian_blur_sigma': 1.0,
                'normalize': True
            },
            'segmentation': {
                'method': 'otsu',
                'min_area': 100
            },
            'output': {
                'save_images': True,
                'save_data': True,
                'create_plots': True
            }
        }
    
    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single image through the simple pipeline.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        output_dir : str
            Output directory
            
        Returns
        -------
        Dict[str, Any]
            Processing results
        """
        print(f"Processing image: {image_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'input_file': image_path,
            'output_dir': output_dir,
            'pipeline': 'simple'
        }
        
        try:
            # Step 1: Load and preprocess image
            image = self.processor.load_image(image_path)
            processed_image = self.processor.preprocess_image(image)
            results['preprocessing'] = {'status': 'success'}
            
            # Step 2: Segment image
            tissue_mask = self.segmenter.segment_tissue(processed_image)
            results['segmentation'] = {
                'tissue_area': int(np.sum(tissue_mask)),
                'status': 'success'
            }
            
            # Step 3: Analyze image
            analysis = self.processor.analyze_image(processed_image)
            tissue_analysis = self.segmenter.analyze_segments(processed_image, tissue_mask)
            
            results['analysis'] = {
                'whole_image': analysis,
                'tissue_regions': tissue_analysis
            }
            
            # Step 4: Create visualizations
            if self.config['output']['create_plots']:
                fig = self.visualizer.create_simple_plot(
                    processed_image, tissue_mask, 
                    title=f"Analysis: {Path(image_path).name}"
                )
                plot_file = output_path / f"{Path(image_path).stem}_analysis.png"
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                results['visualization'] = {'plot_file': str(plot_file)}
            
            # Step 5: Save results
            if self.config['output']['save_data']:
                results_file = output_path / f"{Path(image_path).stem}_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                results['results_file'] = str(results_file)
            
            results['status'] = 'success'
            print(f"✓ Successfully processed: {Path(image_path).name}")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"✗ Failed to process: {Path(image_path).name} - {e}")
        
        return results
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.tif*") -> Dict[str, Any]:
        """
        Process multiple images in batch.
        
        Parameters
        ----------
        input_dir : str
            Input directory
        output_dir : str
            Output directory
        file_pattern : str, optional
            File pattern to match
            
        Returns
        -------
        Dict[str, Any]
            Batch results
        """
        input_path = Path(input_dir)
        image_files = list(input_path.glob(file_pattern))
        
        if not image_files:
            return {
                'status': 'failed',
                'error': f'No images found in {input_dir}'
            }
        
        batch_results = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'results': []
        }
        
        print(f"Starting batch processing: {len(image_files)} images")
        
        for image_file in image_files:
            result = self.process_image(str(image_file), output_dir)
            batch_results['results'].append(result)
            
            if result['status'] == 'success':
                batch_results['processed'] += 1
            else:
                batch_results['failed'] += 1
        
        # Save batch summary
        summary_file = Path(output_dir) / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"Batch processing complete: {batch_results['processed']}/{len(image_files)} successful")
        return batch_results


def run_simple_pipeline(image_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick function to run simple pipeline on a single image.
    
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
    pipeline = SimplePipeline(config)
    return pipeline.process_image(image_path, output_dir)
