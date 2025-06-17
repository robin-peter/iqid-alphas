"""
Image segmentation utilities for H&E and iQID images.
"""

import numpy as np
import cv2
from skimage import filters, morphology, segmentation, measure
from sklearn.cluster import KMeans
import logging

def segment_he_tissue(image, method='filled_tissue_mask', kernel_size=5):
    """
    Segment tissue regions from H&E stained images.
    
    Parameters:
    -----------
    image : np.ndarray
        H&E image (RGB or grayscale)
    method : str
        Segmentation method ('filled_tissue_mask', 'kmeans', 'otsu')
    kernel_size : int
        Morphological kernel size
        
    Returns:
    --------
    np.ndarray
        Binary mask of tissue regions
    """
    try:
        if len(image.shape) == 3:
            # Convert RGB to grayscale using luminance
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if method == 'filled_tissue_mask':
            # Otsu thresholding
            threshold = filters.threshold_otsu(gray)
            binary = gray < threshold  # Tissue is typically darker
            
            # Morphological operations to clean up
            kernel = morphology.disk(kernel_size)
            binary = morphology.binary_closing(binary, kernel)
            binary = morphology.binary_opening(binary, kernel)
            
            # Fill holes
            binary = morphology.binary_fill_holes(binary)
            
        elif method == 'kmeans':
            # K-means clustering for tissue segmentation
            pixel_values = gray.reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=42)
            labels = kmeans.fit_predict(pixel_values)
            
            # Assume tissue is the darkest cluster
            centers = kmeans.cluster_centers_.flatten()
            tissue_label = np.argmin(centers)
            
            binary = (labels == tissue_label).reshape(gray.shape)
            
        elif method == 'otsu':
            # Simple Otsu thresholding
            threshold = filters.threshold_otsu(gray)
            binary = gray < threshold
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
            
        return binary.astype(np.uint8)
        
    except Exception as e:
        logging.error(f"Failed to segment H&E tissue: {str(e)}")
        raise

def segment_iqid_activity(image, method='morphological', min_size=50, watershed_markers='auto'):
    """
    Segment radioactive activity regions from iQID images.
    
    Parameters:
    -----------
    image : np.ndarray
        iQID activity image
    method : str
        Segmentation method ('morphological', 'watershed', 'adaptive')
    min_size : int
        Minimum size of activity regions to keep
    watershed_markers : str or int
        Watershed marker method ('auto' or number of markers)
        
    Returns:
    --------
    np.ndarray
        Labeled image with activity regions
    """
    try:
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
            
        if method == 'morphological':
            # Threshold for high activity regions
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Morphological operations
            kernel = morphology.disk(3)
            binary = morphology.binary_opening(binary, kernel)
            binary = morphology.binary_closing(binary, kernel)
            
            # Remove small objects
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            
            # Label connected components
            labeled = measure.label(binary)
            
        elif method == 'watershed':
            # Watershed segmentation for overlapping regions
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Distance transform
            distance = morphology.distance_transform_edt(binary)
            
            if watershed_markers == 'auto':
                # Find local maxima as markers
                local_maxima = morphology.local_maxima(distance, min_distance=20)
                markers = measure.label(local_maxima)
            else:
                # Use specified number of markers
                markers = np.zeros_like(gray, dtype=int)
                markers[binary] = 1
                
            # Watershed
            labeled = segmentation.watershed(-distance, markers, mask=binary)
            
            # Remove small regions
            for region in measure.regionprops(labeled):
                if region.area < min_size:
                    labeled[labeled == region.label] = 0
                    
        elif method == 'adaptive':
            # Adaptive thresholding for varying intensity
            adaptive_thresh = cv2.adaptiveThreshold(
                (gray * 255).astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            
            binary = adaptive_thresh > 0
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            labeled = measure.label(binary)
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
            
        return labeled
        
    except Exception as e:
        logging.error(f"Failed to segment iQID activity: {str(e)}")
        raise

def extract_region_properties(labeled_image, intensity_image=None):
    """
    Extract properties from segmented regions.
    
    Parameters:
    -----------
    labeled_image : np.ndarray
        Labeled segmentation result
    intensity_image : np.ndarray, optional
        Intensity image for measuring properties
        
    Returns:
    --------
    list
        List of region properties
    """
    try:
        if intensity_image is None:
            props = measure.regionprops(labeled_image)
        else:
            props = measure.regionprops(labeled_image, intensity_image)
            
        # Extract relevant properties
        region_data = []
        for prop in props:
            region_info = {
                'label': prop.label,
                'area': prop.area,
                'centroid': prop.centroid,
                'bbox': prop.bbox,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity
            }
            
            if intensity_image is not None:
                region_info.update({
                    'mean_intensity': prop.mean_intensity,
                    'max_intensity': prop.max_intensity,
                    'min_intensity': prop.min_intensity
                })
                
            region_data.append(region_info)
            
        return region_data
        
    except Exception as e:
        logging.error(f"Failed to extract region properties: {str(e)}")
        raise

def combine_he_iqid_masks(he_mask, iqid_mask, overlap_method='intersection'):
    """
    Combine H&E tissue mask with iQID activity mask.
    
    Parameters:
    -----------
    he_mask : np.ndarray
        H&E tissue segmentation mask
    iqid_mask : np.ndarray
        iQID activity segmentation mask
    overlap_method : str
        Method for combining masks ('intersection', 'union', 'iqid_in_tissue')
        
    Returns:
    --------
    np.ndarray
        Combined mask
    """
    try:
        # Ensure masks are the same size
        if he_mask.shape != iqid_mask.shape:
            # Resize to match (assuming registration has been done)
            from skimage.transform import resize
            iqid_mask = resize(iqid_mask, he_mask.shape, preserve_range=True, order=0)
            iqid_mask = iqid_mask.astype(he_mask.dtype)
            
        if overlap_method == 'intersection':
            # Only regions that are both tissue and activity
            combined = np.logical_and(he_mask > 0, iqid_mask > 0)
            
        elif overlap_method == 'union':
            # All regions that are either tissue or activity
            combined = np.logical_or(he_mask > 0, iqid_mask > 0)
            
        elif overlap_method == 'iqid_in_tissue':
            # Only iQID activity that overlaps with tissue
            tissue_binary = he_mask > 0
            combined = np.where(tissue_binary, iqid_mask, 0)
            
        else:
            raise ValueError(f"Unknown overlap method: {overlap_method}")
            
        return combined.astype(np.uint8)
        
    except Exception as e:
        logging.error(f"Failed to combine H&E and iQID masks: {str(e)}")
        raise
