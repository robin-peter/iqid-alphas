#!/usr/bin/env python3
"""
Enhanced Value Range Processing Module
Advanced value range analysis and processing for H&E and iQID images
"""

import numpy as np
import cv2
from skimage import exposure, filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

class AdvancedValueRangeProcessor:
    """Advanced processor for value range analysis and optimization"""
    
    def __init__(self):
        self.he_typical_range = (0, 255)
        self.iqid_typical_range = (0, 4095)
        self.processing_params = {
            'he_enhancement_factor': 1.2,
            'iqid_enhancement_factor': 1.5,
            'contrast_threshold': 0.1,
            'saturation_limit': 0.95
        }
    
    def process_he_image(self, he_image, mode='tissue_optimized'):
        """Process H&E image with tissue-optimized enhancement"""
        
        if he_image.ndim == 3:
            # Convert to float for processing
            he_float = he_image.astype(np.float32) / 255.0
            
            if mode == 'tissue_optimized':
                # Enhance tissue contrast
                he_enhanced = self._enhance_tissue_contrast(he_float)
            else:
                # Standard processing
                he_enhanced = he_float * self.processing_params['he_enhancement_factor']
                he_enhanced = np.clip(he_enhanced, 0, 1)
            
            return he_enhanced
        else:
            # Grayscale processing
            he_norm = he_image.astype(np.float32) / 255.0
            he_enhanced = exposure.equalize_adapthist(he_norm)
            return he_enhanced
    
    def process_iqid_image(self, iqid_image, mode='autorad_optimized'):
        """Process iQID image with autoradiography-optimized enhancement"""
        
        # Normalize to 0-1 range
        iqid_max = iqid_image.max()
        if iqid_max > 0:
            iqid_norm = iqid_image.astype(np.float32) / iqid_max
        else:
            iqid_norm = iqid_image.astype(np.float32)
        
        if mode == 'autorad_optimized':
            # Enhance activity regions
            iqid_enhanced = self._enhance_activity_regions(iqid_norm)
        else:
            # Standard processing
            iqid_enhanced = iqid_norm * self.processing_params['iqid_enhancement_factor']
            iqid_enhanced = np.clip(iqid_enhanced, 0, 1)
        
        return iqid_enhanced
    
    def _enhance_tissue_contrast(self, he_image):
        """Enhance tissue contrast in H&E images"""
        
        # Convert to grayscale for analysis
        if he_image.ndim == 3:
            gray = rgb2gray(he_image)
        else:
            gray = he_image
        
        # Apply adaptive histogram equalization
        enhanced_gray = exposure.equalize_adapthist(gray, clip_limit=0.03)
        
        if he_image.ndim == 3:
            # Apply enhancement to each channel
            enhanced = np.zeros_like(he_image)
            for i in range(3):
                channel_norm = he_image[:, :, i]
                channel_enhanced = channel_norm * (enhanced_gray / (gray + 1e-8))
                enhanced[:, :, i] = np.clip(channel_enhanced, 0, 1)
            return enhanced
        else:
            return enhanced_gray
    
    def _enhance_activity_regions(self, iqid_image):
        """Enhance activity regions in iQID images"""
        
        # Apply gamma correction to enhance low-intensity regions
        gamma_corrected = exposure.adjust_gamma(iqid_image, gamma=0.7)
        
        # Apply adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(gamma_corrected, clip_limit=0.02)
        
        return enhanced
    
    def analyze_value_ranges(self, he_image, iqid_image):
        """Comprehensive value range analysis"""
        
        analysis = {
            'he_statistics': self._calculate_image_statistics(he_image, 'H&E'),
            'iqid_statistics': self._calculate_image_statistics(iqid_image, 'iQID'),
            'comparison_metrics': {}
        }
        
        # Calculate comparison metrics
        if he_image.ndim == 3:
            he_gray = rgb2gray(he_image)
        else:
            he_gray = he_image
        
        # Normalize for comparison
        he_norm = (he_gray - he_gray.min()) / (he_gray.max() - he_gray.min() + 1e-8)
        iqid_norm = (iqid_image - iqid_image.min()) / (iqid_image.max() - iqid_image.min() + 1e-8)
        
        # Cross-correlation
        if he_norm.shape == iqid_norm.shape:
            correlation = np.corrcoef(he_norm.flatten(), iqid_norm.flatten())[0, 1]
            analysis['comparison_metrics']['cross_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # Mutual information (simplified)
        analysis['comparison_metrics']['mutual_information'] = self._calculate_mutual_information(he_norm, iqid_norm)
        
        return analysis
    
    def _calculate_image_statistics(self, image, image_type):
        """Calculate comprehensive image statistics"""
        
        stats = {
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'median': float(np.median(image)),
            'range': float(image.max() - image.min())
        }
        
        # Add type-specific statistics
        if image_type == 'H&E' and image.ndim == 3:
            # Channel statistics
            stats['channel_statistics'] = {}
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = image[:, :, i]
                stats['channel_statistics'][channel] = {
                    'min': float(channel_data.min()),
                    'max': float(channel_data.max()),
                    'mean': float(channel_data.mean()),
                    'std': float(channel_data.std())
                }
        
        # Contrast analysis
        if image.max() > image.min():
            stats['contrast'] = float((image.max() - image.min()) / (image.max() + image.min()))
        else:
            stats['contrast'] = 0.0
        
        return stats
    
    def _calculate_mutual_information(self, img1, img2):
        """Calculate simplified mutual information between two images"""
        
        if img1.shape != img2.shape:
            # Resize to match
            from skimage.transform import resize
            img2 = resize(img2, img1.shape, anti_aliasing=True)
        
        # Quantize images for histogram calculation
        img1_quant = (img1 * 255).astype(np.uint8)
        img2_quant = (img2 * 255).astype(np.uint8)
        
        # Calculate joint histogram
        hist_2d, _, _ = np.histogram2d(img1_quant.flatten(), img2_quant.flatten(), bins=256)
        
        # Normalize
        hist_2d = hist_2d / hist_2d.sum()
        
        # Calculate marginal histograms
        hist_1d_x = hist_2d.sum(axis=1)
        hist_1d_y = hist_2d.sum(axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(256):
            for j in range(256):
                if hist_2d[i, j] > 0 and hist_1d_x[i] > 0 and hist_1d_y[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_1d_x[i] * hist_1d_y[j]))
        
        return float(mi)
    
    def generate_processing_report(self, analysis_results):
        """Generate comprehensive processing report"""
        
        report = {
            'processing_timestamp': np.datetime64('now').isoformat(),
            'processor_version': 'AdvancedValueRange_v1.0',
            'image_analysis': analysis_results,
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        return report
    
    def _generate_recommendations(self, analysis):
        """Generate processing recommendations based on analysis"""
        
        recommendations = []
        
        # H&E recommendations
        he_contrast = analysis['he_statistics']['contrast']
        if he_contrast < 0.2:
            recommendations.append("H&E image has low contrast - consider histogram equalization")
        elif he_contrast > 0.8:
            recommendations.append("H&E image has high contrast - consider gamma correction")
        
        # iQID recommendations
        iqid_stats = analysis['iqid_statistics']
        if iqid_stats['std'] < 0.1 * iqid_stats['mean']:
            recommendations.append("iQID image has low variance - check for proper activity distribution")
        
        # Correlation recommendations
        if 'cross_correlation' in analysis['comparison_metrics']:
            correlation = analysis['comparison_metrics']['cross_correlation']
            if correlation < 0.2:
                recommendations.append("Low correlation between H&E and iQID - verify registration quality")
            elif correlation > 0.7:
                recommendations.append("Good correlation between H&E and iQID - proceed with analysis")
        
        return recommendations

# Convenience functions for standalone use
def analyze_image_pair(he_image, iqid_image):
    """Analyze H&E-iQID image pair"""
    processor = AdvancedValueRangeProcessor()
    return processor.analyze_value_ranges(he_image, iqid_image)

def enhance_he_image(he_image, mode='tissue_optimized'):
    """Enhance H&E image"""
    processor = AdvancedValueRangeProcessor()
    return processor.process_he_image(he_image, mode)

def enhance_iqid_image(iqid_image, mode='autorad_optimized'):
    """Enhance iQID image"""
    processor = AdvancedValueRangeProcessor()
    return processor.process_iqid_image(iqid_image, mode)
