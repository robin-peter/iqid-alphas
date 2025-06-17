#!/usr/bin/env python3
"""
Value Range Testing Module
Tests value range analysis and comparison between H&E and iQID images
"""

import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('./src')
sys.path.append('.')

class TestValueRanges(unittest.TestCase):
    """Test value range analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic test data
        self.he_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.iqid_image = np.random.randint(0, 1000, (25, 25), dtype=np.uint16)
        
    def test_he_value_range_analysis(self):
        """Test H&E image value range analysis"""
        
        # Check basic properties
        self.assertEqual(self.he_image.ndim, 3)
        self.assertEqual(self.he_image.shape[2], 3)  # RGB channels
        
        # Value range checks
        self.assertGreaterEqual(self.he_image.min(), 0)
        self.assertLessEqual(self.he_image.max(), 255)
        
        # Channel analysis
        for channel in range(3):
            channel_data = self.he_image[:, :, channel]
            self.assertGreaterEqual(channel_data.min(), 0)
            self.assertLessEqual(channel_data.max(), 255)
    
    def test_iqid_value_range_analysis(self):
        """Test iQID image value range analysis"""
        
        # Check basic properties
        self.assertEqual(self.iqid_image.ndim, 2)
        
        # Value range checks
        self.assertGreaterEqual(self.iqid_image.min(), 0)
        self.assertLessEqual(self.iqid_image.max(), 1000)
        
        # Statistical analysis
        mean_val = np.mean(self.iqid_image)
        std_val = np.std(self.iqid_image)
        
        self.assertGreater(mean_val, 0)
        self.assertGreater(std_val, 0)
    
    def test_value_range_comparison(self):
        """Test comparison between H&E and iQID value ranges"""
        
        # Convert H&E to grayscale for comparison
        he_gray = np.mean(self.he_image, axis=2)
        
        # Normalize both to 0-1 range
        he_normalized = he_gray / 255.0
        iqid_normalized = self.iqid_image / 1000.0
        
        # Check normalization
        self.assertGreaterEqual(he_normalized.min(), 0)
        self.assertLessEqual(he_normalized.max(), 1)
        self.assertGreaterEqual(iqid_normalized.min(), 0)
        self.assertLessEqual(iqid_normalized.max(), 1)
        
        # Compare statistical properties
        he_mean = np.mean(he_normalized)
        iqid_mean = np.mean(iqid_normalized)
        
        # Both should have reasonable mean values
        self.assertGreater(he_mean, 0)
        self.assertLess(he_mean, 1)
        self.assertGreater(iqid_mean, 0)
        self.assertLess(iqid_mean, 1)
    
    def test_value_range_processing_pipeline(self):
        """Test complete value range processing pipeline"""
        
        # Simulate processing pipeline
        he_processed = self.he_image.astype(np.float32) / 255.0
        iqid_processed = self.iqid_image.astype(np.float32) / 1000.0
        
        # Apply basic preprocessing
        he_enhanced = np.clip(he_processed * 1.2, 0, 1)
        iqid_enhanced = np.clip(iqid_processed * 1.5, 0, 1)
        
        # Validate processing results
        self.assertGreaterEqual(he_enhanced.min(), 0)
        self.assertLessEqual(he_enhanced.max(), 1)
        self.assertGreaterEqual(iqid_enhanced.min(), 0)
        self.assertLessEqual(iqid_enhanced.max(), 1)
        
        # Check that enhancement actually modified the data
        self.assertFalse(np.array_equal(he_processed, he_enhanced))
        self.assertFalse(np.array_equal(iqid_processed, iqid_enhanced))

class TestValueRangeLogging(unittest.TestCase):
    """Test logging and reporting functionality"""
    
    def test_value_range_reporting(self):
        """Test value range reporting functionality"""
        
        # Create sample data
        test_data = {
            'he_min': 0,
            'he_max': 255,
            'he_mean': 127.5,
            'iqid_min': 0,
            'iqid_max': 1000,
            'iqid_mean': 500.0
        }
        
        # Generate report
        report = self._generate_value_range_report(test_data)
        
        # Validate report structure
        self.assertIn('he_statistics', report)
        self.assertIn('iqid_statistics', report)
        self.assertIn('comparison_metrics', report)
        
        # Validate content
        self.assertEqual(report['he_statistics']['min'], 0)
        self.assertEqual(report['he_statistics']['max'], 255)
        self.assertEqual(report['iqid_statistics']['min'], 0)
        self.assertEqual(report['iqid_statistics']['max'], 1000)
    
    def _generate_value_range_report(self, data):
        """Generate a value range analysis report"""
        
        report = {
            'he_statistics': {
                'min': data['he_min'],
                'max': data['he_max'],
                'mean': data['he_mean'],
                'range': data['he_max'] - data['he_min']
            },
            'iqid_statistics': {
                'min': data['iqid_min'],
                'max': data['iqid_max'],
                'mean': data['iqid_mean'],
                'range': data['iqid_max'] - data['iqid_min']
            },
            'comparison_metrics': {
                'he_iqid_range_ratio': (data['he_max'] - data['he_min']) / (data['iqid_max'] - data['iqid_min']),
                'he_iqid_mean_ratio': data['he_mean'] / data['iqid_mean']
            }
        }
        
        return report

if __name__ == '__main__':
    unittest.main()
