#!/usr/bin/env python3
"""
Comprehensive Activity Segmentation Module
Advanced segmentation methods for iQID activity images
"""

import numpy as np
import cv2
from skimage import segmentation, measure, morphology, filters
from skimage.feature import peak_local_maxima
from scipy import ndimage
import matplotlib.pyplot as plt

class ComprehensiveActivitySegmenter:
    """Comprehensive segmentation for iQID activity images"""
    
    def __init__(self):
        self.default_params = {
            'min_size': 50,
            'max_size': 10000,
            'watershed_markers': 'auto',
            'morphology_kernel_size': 3,
            'gaussian_sigma': 1.0
        }
    
    def morphological_segmentation(self, iqid_image, min_size=50, watershed_markers='auto'):
        """Morphological segmentation with watershed"""
        
        # Normalize image
        if iqid_image.max() > iqid_image.min():
            normalized = (iqid_image - iqid_image.min()) / (iqid_image.max() - iqid_image.min())
        else:
            normalized = iqid_image.astype(np.float32)
        
        # Apply Gaussian filter
        smoothed = filters.gaussian(normalized, sigma=self.default_params['gaussian_sigma'])
        
        # Threshold using Otsu's method
        threshold = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold
        
        # Morphological operations
        kernel_size = self.default_params['morphology_kernel_size']
        kernel = morphology.disk(kernel_size)
        binary_cleaned = morphology.opening(binary, kernel)
        binary_cleaned = morphology.closing(binary_cleaned, kernel)
        
        # Generate watershed markers
        if watershed_markers == 'auto':
            distance = ndimage.distance_transform_edt(binary_cleaned)
            coords = peak_local_maxima(distance, min_distance=10, threshold_abs=0.3*distance.max())
            markers = np.zeros_like(distance, dtype=int)
            for i, coord in enumerate(coords):
                markers[coord] = i + 1
        else:
            markers = ndimage.label(binary_cleaned)[0]
        
        # Apply watershed
        labels = segmentation.watershed(-distance, markers, mask=binary_cleaned)
        
        # Extract individual masks
        masks = []
        properties = []
        
        for region in measure.regionprops(labels):
            if region.area >= min_size:
                mask = labels == region.label
                masks.append(mask.astype(np.uint8))
                
                properties.append({
                    'area': region.area,
                    'centroid': region.centroid,
                    'bbox': region.bbox,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity
                })
        
        return masks, properties
    
    def adaptive_threshold_segmentation(self, iqid_image, block_size=11, C=2):
        """Adaptive threshold segmentation"""
        
        # Convert to uint8 for adaptive thresholding
        normalized = cv2.normalize(iqid_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create masks from contours
        masks = []
        properties = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.default_params['min_size']:
                mask = np.zeros_like(binary, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                masks.append((mask > 0).astype(np.uint8))
                
                # Calculate properties
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                x, y, w, h = cv2.boundingRect(contour)
                
                properties.append({
                    'area': area,
                    'centroid': (cy, cx),
                    'bbox': (y, x, y+h, x+w),
                    'contour': contour
                })
        
        return masks, properties
    
    def region_growing_segmentation(self, iqid_image, seed_threshold=0.8, growth_threshold=0.3):
        """Region growing segmentation"""
        
        # Normalize image
        if iqid_image.max() > iqid_image.min():
            normalized = (iqid_image - iqid_image.min()) / (iqid_image.max() - iqid_image.min())
        else:
            normalized = iqid_image.astype(np.float32)
        
        # Find seed points (high-intensity pixels)
        seeds = normalized > seed_threshold
        
        # Label connected components of seeds
        seed_labels, num_seeds = ndimage.label(seeds)
        
        masks = []
        properties = []
        
        for seed_id in range(1, num_seeds + 1):
            # Get seed region
            seed_mask = seed_labels == seed_id
            
            if np.sum(seed_mask) < 5:  # Skip very small seeds
                continue
            
            # Grow region
            grown_mask = self._grow_region(normalized, seed_mask, growth_threshold)
            
            if np.sum(grown_mask) >= self.default_params['min_size']:
                masks.append(grown_mask.astype(np.uint8))
                
                # Calculate properties
                region_props = measure.regionprops(grown_mask.astype(int))[0]
                properties.append({
                    'area': region_props.area,
                    'centroid': region_props.centroid,
                    'bbox': region_props.bbox,
                    'eccentricity': region_props.eccentricity
                })
        
        return masks, properties
    
    def _grow_region(self, image, seed_mask, threshold):
        """Grow region from seed using threshold"""
        
        grown = seed_mask.copy()
        changed = True
        
        while changed:
            changed = False
            
            # Dilate current region by 1 pixel
            dilated = morphology.binary_dilation(grown)
            
            # Find new pixels to potentially add
            candidates = dilated & ~grown
            
            # Check if candidates meet threshold criteria
            for y, x in np.argwhere(candidates):
                # Check neighboring pixels in current region
                neighbors = grown[max(0, y-1):y+2, max(0, x-1):x+2]
                if np.any(neighbors):
                    neighbor_values = image[max(0, y-1):y+2, max(0, x-1):x+2][neighbors]
                    if len(neighbor_values) > 0:
                        mean_neighbor = np.mean(neighbor_values)
                        if abs(image[y, x] - mean_neighbor) < threshold:
                            grown[y, x] = True
                            changed = True
        
        return grown
    
    def multi_scale_segmentation(self, iqid_image, scales=[1, 2, 4]):
        """Multi-scale segmentation approach"""
        
        all_masks = []
        all_properties = []
        
        for scale in scales:
            # Downsample image
            h, w = iqid_image.shape
            downsampled = cv2.resize(iqid_image, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
            
            # Segment at this scale
            masks, properties = self.morphological_segmentation(downsampled, min_size=self.default_params['min_size']//scale**2)
            
            # Upsample masks back to original size
            for mask in masks:
                upsampled_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                all_masks.append(upsampled_mask)
            
            # Adjust properties for scale
            for prop in properties:
                scaled_prop = prop.copy()
                scaled_prop['area'] *= scale**2
                scaled_prop['centroid'] = (prop['centroid'][0]*scale, prop['centroid'][1]*scale)
                scaled_prop['bbox'] = tuple(coord*scale for coord in prop['bbox'])
                all_properties.append(scaled_prop)
        
        # Remove overlapping masks (keep largest)
        final_masks, final_properties = self._remove_overlapping_masks(all_masks, all_properties)
        
        return final_masks, final_properties
    
    def _remove_overlapping_masks(self, masks, properties, overlap_threshold=0.5):
        """Remove overlapping masks, keeping the largest ones"""
        
        if not masks:
            return [], []
        
        # Sort by area (largest first)
        sorted_indices = sorted(range(len(masks)), key=lambda i: properties[i]['area'], reverse=True)
        
        final_masks = []
        final_properties = []
        
        for i in sorted_indices:
            current_mask = masks[i]
            current_prop = properties[i]
            
            # Check overlap with already selected masks
            overlaps = False
            for existing_mask in final_masks:
                intersection = np.logical_and(current_mask, existing_mask)
                union = np.logical_or(current_mask, existing_mask)
                
                if np.sum(union) > 0:
                    overlap_ratio = np.sum(intersection) / np.sum(union)
                    if overlap_ratio > overlap_threshold:
                        overlaps = True
                        break
            
            if not overlaps:
                final_masks.append(current_mask)
                final_properties.append(current_prop)
        
        return final_masks, final_properties
    
    def segment_image(self, iqid_image, method='morphological', **kwargs):
        """Main segmentation interface"""
        
        if method == 'morphological':
            return self.morphological_segmentation(iqid_image, **kwargs)
        elif method == 'adaptive_threshold':
            return self.adaptive_threshold_segmentation(iqid_image, **kwargs)
        elif method == 'region_growing':
            return self.region_growing_segmentation(iqid_image, **kwargs)
        elif method == 'multi_scale':
            return self.multi_scale_segmentation(iqid_image, **kwargs)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    
    def visualize_segmentation(self, iqid_image, masks, title="Segmentation Results"):
        """Visualize segmentation results"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(iqid_image, cmap='hot')
        axes[0].set_title('Original iQID Image')
        axes[0].axis('off')
        
        # Segmentation overlay
        overlay = iqid_image.copy()
        colored_masks = np.zeros((*iqid_image.shape, 3))
        
        for i, mask in enumerate(masks):
            color = plt.cm.Set3(i / len(masks))[:3]
            colored_masks[mask > 0] = color
        
        axes[1].imshow(iqid_image, cmap='gray', alpha=0.7)
        axes[1].imshow(colored_masks, alpha=0.5)
        axes[1].set_title(f'{title} ({len(masks)} regions)')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig

# Convenience function for external use
def segment_activity_image(iqid_image, method='morphological', **kwargs):
    """Convenience function for activity segmentation"""
    segmenter = ComprehensiveActivitySegmenter()
    return segmenter.segment_image(iqid_image, method, **kwargs)
