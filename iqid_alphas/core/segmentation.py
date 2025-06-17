"""
Image Segmentation Module

Simplified image segmentation functionality for tissue and activity detection.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

try:
    from skimage import filters, morphology, measure, segmentation
    from scipy import ndimage
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


class ImageSegmenter:
    """
    Simplified image segmentation class.
    
    Provides basic segmentation capabilities for tissue detection
    and activity region identification.
    """
    
    def __init__(self, method: str = 'otsu'):
        """
        Initialize the image segmenter.
        
        Parameters
        ----------
        method : str, optional
            Segmentation method ('otsu', 'adaptive', 'threshold')
        """
        self.method = method
        
    def segment_tissue(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Segment tissue regions from image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Binary tissue mask
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray_image = np.mean(image, axis=2)
        else:
            gray_image = image.copy()
        
        # Apply segmentation method
        if self.method == 'otsu':
            threshold = filters.threshold_otsu(gray_image)
            mask = gray_image > threshold
        elif self.method == 'adaptive':
            mask = gray_image > filters.threshold_local(gray_image, block_size=35)
        else:  # Simple threshold
            threshold = kwargs.get('threshold', 0.5 * gray_image.max())
            mask = gray_image > threshold
        
        # Clean up mask
        mask = self._clean_mask(mask, **kwargs)
        
        return mask.astype(bool)
    
    def segment_activity(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Segment high activity regions from image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (typically iQID activity image)
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Binary activity mask
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray_image = np.mean(image, axis=2)
        else:
            gray_image = image.copy()
        
        # Use higher threshold for activity detection
        if self.method == 'otsu':
            threshold = filters.threshold_otsu(gray_image)
            # Use higher threshold for activity
            activity_threshold = threshold + 0.3 * (gray_image.max() - threshold)
            mask = gray_image > activity_threshold
        else:
            # Use percentile-based threshold
            threshold = np.percentile(gray_image[gray_image > 0], 85)
            mask = gray_image > threshold
        
        # Clean up mask
        mask = self._clean_mask(mask, min_size=kwargs.get('min_activity_size', 50))
        
        return mask.astype(bool)
    
    def _clean_mask(self, mask: np.ndarray, min_size: int = 100, **kwargs) -> np.ndarray:
        """Clean up binary mask by removing small objects and holes."""
        if not HAS_IMAGING:
            return mask
            
        # Remove small objects
        cleaned = morphology.remove_small_objects(mask, min_size=min_size)
        
        # Fill small holes
        if kwargs.get('fill_holes', True):
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_size//2)
        
        # Optional morphological operations
        if kwargs.get('morphological_closing', True):
            cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
        
        return cleaned
    
    def analyze_segments(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze segmented regions.
        
        Parameters
        ----------
        image : np.ndarray
            Original image
        mask : np.ndarray
            Binary mask
            
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        if not HAS_IMAGING:
            return {'error': 'Imaging libraries not available'}
            
        results = {}
        
        # Basic statistics
        if np.sum(mask) > 0:
            results['area'] = int(np.sum(mask))
            results['mean_intensity'] = float(np.mean(image[mask]))
            results['total_intensity'] = float(np.sum(image[mask]))
            results['max_intensity'] = float(np.max(image[mask]))
            results['std_intensity'] = float(np.std(image[mask]))
        else:
            results['area'] = 0
            results['mean_intensity'] = 0.0
            results['total_intensity'] = 0.0
            results['max_intensity'] = 0.0
            results['std_intensity'] = 0.0
        
        # Region properties
        try:
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask, intensity_image=image)
            
            results['num_regions'] = len(props)
            if props:
                results['largest_region_area'] = int(max(prop.area for prop in props))
                results['mean_region_area'] = float(np.mean([prop.area for prop in props]))
                results['centroid'] = [float(props[0].centroid[0]), float(props[0].centroid[1])] if props else [0.0, 0.0]
        except Exception:
            results['num_regions'] = 0
            results['largest_region_area'] = 0
            results['mean_region_area'] = 0.0
            results['centroid'] = [0.0, 0.0]
        
        return results
    
    def segment_combined(self, he_image: np.ndarray, iqid_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform combined segmentation using both H&E and iQID images.
        
        Parameters
        ----------
        he_image : np.ndarray
            H&E histology image
        iqid_image : np.ndarray
            iQID activity image
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing tissue and activity masks
        """
        results = {}
        
        # Segment tissue from H&E
        tissue_segmenter = ImageSegmenter('adaptive')  # Better for H&E
        results['tissue_mask'] = tissue_segmenter.segment_tissue(he_image)
        
        # Segment activity from iQID
        activity_segmenter = ImageSegmenter('otsu')
        results['activity_mask'] = activity_segmenter.segment_activity(iqid_image)
        
        # Combined mask (tissue AND activity)
        if results['tissue_mask'].shape == results['activity_mask'].shape:
            results['combined_mask'] = results['tissue_mask'] & results['activity_mask']
        else:
            print("Warning: Image shapes don't match, skipping combined mask")
            results['combined_mask'] = results['activity_mask']
        
        return results
    
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Simple wrapper for image segmentation.
        
        Parameters
        ----------
        image : np.ndarray
            Input image to segment
        **kwargs
            Additional parameters
            
        Returns
        -------
        np.ndarray
            Segmented image/mask
        """
        # Default to tissue segmentation
        return self.segment_tissue(image, **kwargs)


def quick_segment(image: np.ndarray, segment_type: str = 'tissue', method: str = 'otsu') -> np.ndarray:
    """
    Quick segmentation function.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    segment_type : str, optional
        Type of segmentation ('tissue' or 'activity')
    method : str, optional
        Segmentation method
        
    Returns
    -------
    np.ndarray
        Binary mask
    """
    segmenter = ImageSegmenter(method=method)
    
    if segment_type == 'tissue':
        return segmenter.segment_tissue(image)
    else:
        return segmenter.segment_activity(image)
