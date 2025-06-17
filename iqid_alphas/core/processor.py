"""
Core IQID Image Processor

Consolidated functionality for iQID image processing including:
- Image loading and preprocessing
- Basic quantitative analysis
- File I/O operations
"""

import os
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from skimage import io, filters, exposure, transform
    from scipy import ndimage
    import matplotlib.pyplot as plt
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False
    print("Warning: Imaging libraries not available. Install with: pip install scikit-image scipy matplotlib")


class IQIDProcessor:
    """
    Main processor class for iQID image analysis.
    
    This class provides core functionality for loading, preprocessing,
    and analyzing iQID images.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the IQID processor.
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.images = {}
        self.results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'processing': {
                'gaussian_blur_sigma': 1.0,
                'normalize': True,
                'enhance_contrast': False
            },
            'analysis': {
                'roi_threshold': 0.5,
                'min_area': 100
            },
            'output': {
                'save_intermediates': False,
                'file_format': 'png'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def load_image(self, image_path: str, key: str = 'main') -> np.ndarray:
        """
        Load an image from file.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        key : str, optional
            Key to store the image under (default: 'main')
            
        Returns
        -------
        np.ndarray
            Loaded image array
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        try:
            image = io.imread(image_path)
            self.images[key] = image
            print(f"Loaded image: {image.shape} from {image_path}")
            return image
        except Exception as e:
            raise FileNotFoundError(f"Could not load image from {image_path}: {e}")
    
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess an image with basic operations.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        **kwargs
            Additional preprocessing parameters
            
        Returns
        -------
        np.ndarray
            Preprocessed image
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        processed = image.copy()
        
        # Apply Gaussian blur
        sigma = kwargs.get('gaussian_blur_sigma', self.config['processing']['gaussian_blur_sigma'])
        if sigma > 0:
            processed = filters.gaussian(processed, sigma=sigma)
        
        # Normalize intensity
        if kwargs.get('normalize', self.config['processing']['normalize']):
            processed = exposure.rescale_intensity(processed)
        
        # Enhance contrast
        if kwargs.get('enhance_contrast', self.config['processing']['enhance_contrast']):
            processed = exposure.equalize_adapthist(processed)
        
        return processed
    
    def analyze_image(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform basic quantitative analysis on an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        **kwargs
            Analysis parameters
            
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        results = {}
        
        # Basic statistics
        results['mean_intensity'] = float(np.mean(image))
        results['std_intensity'] = float(np.std(image))
        results['min_intensity'] = float(np.min(image))
        results['max_intensity'] = float(np.max(image))
        results['total_intensity'] = float(np.sum(image))
        
        # Image properties
        results['image_shape'] = image.shape
        results['image_size'] = image.size
        results['image_dtype'] = str(image.dtype)
        
        # Threshold-based analysis
        threshold = kwargs.get('threshold', self.config['analysis']['roi_threshold'])
        if image.max() > 0:
            normalized = image / image.max()
            roi_mask = normalized > threshold
            results['roi_area'] = int(np.sum(roi_mask))
            results['roi_mean_intensity'] = float(np.mean(image[roi_mask])) if np.sum(roi_mask) > 0 else 0.0
            results['roi_total_intensity'] = float(np.sum(image[roi_mask]))
        
        return results
    
    def process_single_image(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        output_dir : str, optional
            Directory to save outputs
            
        Returns
        -------
        Dict[str, Any]
            Processing results
        """
        # Load image
        image = self.load_image(image_path)
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Analyze
        results = self.analyze_image(processed_image)
        
        # Add metadata
        results['input_file'] = str(image_path)
        results['processing_config'] = self.config
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir, Path(image_path).stem)
        
        return results
    
    def process_batch(self, input_dir: str, output_dir: str, file_pattern: str = "*.tif*") -> Dict[str, Any]:
        """
        Process multiple images in batch.
        
        Parameters
        ----------
        input_dir : str
            Directory containing input images
        output_dir : str
            Directory to save outputs
        file_pattern : str, optional
            Pattern to match image files (default: "*.tif*")
            
        Returns
        -------
        Dict[str, Any]
            Batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_files = list(input_path.glob(file_pattern))
        if not image_files:
            raise FileNotFoundError(f"No images found in {input_dir} matching {file_pattern}")
        
        batch_results = {
            'summary': {
                'total_images': len(image_files),
                'processed': 0,
                'failed': 0
            },
            'individual_results': {},
            'failed_files': []
        }
        
        print(f"Processing {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                results = self.process_single_image(str(image_file), str(output_path))
                batch_results['individual_results'][image_file.name] = results
                batch_results['summary']['processed'] += 1
                print(f"✓ Processed: {image_file.name}")
            except Exception as e:
                batch_results['failed_files'].append({
                    'file': image_file.name,
                    'error': str(e)
                })
                batch_results['summary']['failed'] += 1
                print(f"✗ Failed: {image_file.name} - {e}")
        
        # Save batch summary
        summary_file = output_path / 'batch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"Batch processing complete: {batch_results['summary']['processed']}/{len(image_files)} successful")
        return batch_results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str, filename: str):
        """Save processing results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_file = output_path / f"{filename}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def create_visualization(self, image: np.ndarray, title: str = "IQID Image") -> 'plt.Figure':
        """
        Create a basic visualization of an image.
        
        Parameters
        ----------
        image : np.ndarray
            Image to visualize
        title : str, optional
            Plot title
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        if not HAS_IMAGING:
            raise ImportError("Matplotlib not available")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(image.shape) == 2:  # Grayscale
            im = ax.imshow(image, cmap='viridis')
            plt.colorbar(im, ax=ax)
        else:  # Color
            ax.imshow(image)
        
        ax.set_title(title)
        ax.axis('off')
        
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded images and results."""
        return {
            'loaded_images': list(self.images.keys()),
            'num_images': len(self.images),
            'results_available': list(self.results.keys()),
            'config': self.config
        }
    
    def process(self, image_data: np.ndarray) -> np.ndarray:
        """
        Process a single image array.
        
        Parameters
        ----------
        image_data : np.ndarray
            Input image data as numpy array
            
        Returns
        -------
        np.ndarray
            Processed image data
        """
        if image_data is None:
            raise ValueError("Image data cannot be None")
        
        if not isinstance(image_data, np.ndarray):
            raise TypeError("Image data must be a numpy array")
        
        if image_data.size == 0:
            raise ValueError("Image data cannot be empty")
        
        if not HAS_IMAGING:
            # Basic processing without imaging libraries
            processed = image_data.astype(np.float32)
            if self.config['processing']['normalize']:
                processed = (processed - processed.min()) / (processed.max() - processed.min())
            return processed
        
        # Convert to float32 for processing
        processed = image_data.astype(np.float32)
        
        # Apply Gaussian blur if specified
        sigma = self.config['processing']['gaussian_blur_sigma']
        if sigma > 0:
            processed = filters.gaussian(processed, sigma=sigma)
        
        # Normalize if specified
        if self.config['processing']['normalize']:
            processed = exposure.rescale_intensity(processed)
        
        # Enhance contrast if specified
        if self.config['processing']['enhance_contrast']:
            processed = exposure.equalize_adapthist(processed)
        
        return processed

def quick_process(image_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Quick processing function for single images.
    
    Parameters
    ----------
    image_path : str
        Path to image file
    output_dir : str, optional
        Output directory
        
    Returns
    -------
    Dict[str, Any]
        Processing results
    """
    processor = IQIDProcessor()
    return processor.process_single_image(image_path, output_dir)


def batch_process(input_dir: str, output_dir: str, file_pattern: str = "*.tif*") -> Dict[str, Any]:
    """
    Quick batch processing function.
    
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
    processor = IQIDProcessor()
    return processor.process_batch(input_dir, output_dir, file_pattern)
