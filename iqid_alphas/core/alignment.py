"""
Image Alignment Module

Simplified image alignment and registration functionality.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings

try:
    from skimage import transform, measure # feature might not be needed if using phase_cross_correlation
    from skimage.registration import phase_cross_correlation # More modern way
    from scipy import ndimage
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


class ImageAligner:
    """
    Simplified image alignment class.
    
    Provides basic image registration and alignment capabilities
    for iQID and H&E image pairs.
    """
    
    def __init__(self, method: str = 'phase_correlation'):
        """
        Initialize the image aligner.
        
        Parameters
        ----------
        method : str, optional
            Alignment method ('phase_correlation', 'feature_matching')
        """
        self.method = method
        self.transformation = None
        
    def align_images(self, fixed_image: np.ndarray, moving_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Align two images.
        
        Parameters
        ----------
        fixed_image : np.ndarray
            Reference image (fixed)
        moving_image : np.ndarray
            Image to be aligned (moving)
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Aligned image and transformation parameters
        """
        if not HAS_IMAGING:
            raise ImportError("Imaging libraries not available")
            
        if self.method == 'phase_correlation':
            return self._phase_correlation_alignment(fixed_image, moving_image)
        else:
            return self._simple_translation_alignment(fixed_image, moving_image)
    
    def _phase_correlation_alignment(self, fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Phase correlation based alignment."""
        try:
            # Convert to grayscale if needed
            if len(fixed.shape) > 2:
                fixed = np.mean(fixed, axis=2)
            if len(moving.shape) > 2:
                moving = np.mean(moving, axis=2)
            
            # Compute phase correlation using phase_cross_correlation
            # For scikit-image >= 0.19, it returns (shifts, error, phasediff) when upsample_factor > 1
            # For scikit-image < 0.19 or basic usage, it might just return shifts.
            # Let's try without return_error and see if it returns three values with upsample_factor.
            # If not, we'll need to adjust.
            result = phase_cross_correlation(fixed, moving, upsample_factor=10)
            if isinstance(result, tuple) and len(result) == 3:
                shift, error, diffphase = result
            else: # Assuming it just returned shifts
                shift = result
                error, diffphase = None, None # Or some default/calculated values if needed
            # shift is (dy, dx)
            
            # Apply transformation
            aligned = ndimage.shift(moving, shift) # ndimage.shift expects (dy, dx)
            
            transformation = {
                'method': 'phase_correlation',
                'shift': shift.tolist(), # shift is already [dy, dx]
                'error': float(error) if error is not None else None, # error from phase_cross_correlation
                'phasediff': float(diffphase) if diffphase is not None else None, # phasediff from phase_cross_correlation
                'translation_x': float(shift[1]), # dx
                'translation_y': float(shift[0])  # dy
            }
            
            return aligned, transformation
            
        except Exception as e:
            # Catch specific error if phase_cross_correlation is not found due to version, or general errors
            print(f"Phase correlation with phase_cross_correlation failed: {e}, using simple alignment")
            return self._simple_translation_alignment(fixed, moving)
    
    def _simple_translation_alignment(self, fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simple center-based alignment."""
        # Calculate center-of-mass based alignment
        fixed_gray = np.mean(fixed, axis=2) if len(fixed.shape) > 2 else fixed
        moving_gray = np.mean(moving, axis=2) if len(moving.shape) > 2 else moving
        
        # Find centers of mass
        fixed_com = ndimage.center_of_mass(fixed_gray)
        moving_com = ndimage.center_of_mass(moving_gray)
        
        # Calculate shift
        shift = [fixed_com[0] - moving_com[0], fixed_com[1] - moving_com[1]]
        
        # Apply shift
        aligned = ndimage.shift(moving, shift)
        
        transformation = {
            'method': 'simple_translation',
            'shift': shift,
            'translation_x': float(shift[1]),
            'translation_y': float(shift[0])
        }
        
        return aligned, transformation
    
    def calculate_alignment_quality(self, fixed: np.ndarray, aligned: np.ndarray) -> Dict[str, float]:
        """
        Calculate alignment quality metrics.
        
        Parameters
        ----------
        fixed : np.ndarray
            Reference image
        aligned : np.ndarray
            Aligned image
            
        Returns
        -------
        Dict[str, float]
            Quality metrics
        """
        if not HAS_IMAGING:
            return {'correlation': 0.0, 'mse': float('inf')}
            
        # Convert to same format
        if len(fixed.shape) > 2:
            fixed = np.mean(fixed, axis=2)
        if len(aligned.shape) > 2:
            aligned = np.mean(aligned, axis=2)
        
        # Ensure same shape
        min_shape = [min(fixed.shape[i], aligned.shape[i]) for i in range(2)]
        fixed = fixed[:min_shape[0], :min_shape[1]]
        aligned = aligned[:min_shape[0], :min_shape[1]]
        
        # Calculate correlation
        correlation = np.corrcoef(fixed.flatten(), aligned.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calculate MSE
        mse = np.mean((fixed - aligned) ** 2)
        
        return {
            'correlation': float(correlation),
            'mse': float(mse),
            'alignment_score': float(max(0, correlation))  # Normalized score
        }
    
    def align(self, fixed_image: np.ndarray, moving_image: np.ndarray) -> np.ndarray:
        """
        Simple wrapper for image alignment.
        
        Parameters
        ----------
        fixed_image : np.ndarray
            Reference image (fixed)
        moving_image : np.ndarray
            Image to be aligned (moving)
            
        Returns
        -------
        np.ndarray
            Aligned moving image
        """
        aligned_image, _ = self.align_images(fixed_image, moving_image)
        return aligned_image


def align_image_pair(image1: np.ndarray, image2: np.ndarray, method: str = 'phase_correlation') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quick function to align two images.
    
    Parameters
    ----------
    image1 : np.ndarray
        Reference image
    image2 : np.ndarray
        Image to align
    method : str, optional
        Alignment method
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Aligned image and transformation info
    """
    aligner = ImageAligner(method=method)
    return aligner.align_images(image1, image2)
