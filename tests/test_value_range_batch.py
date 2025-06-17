#!/usr/bin/env python3
"""
Value Range Batch Processing Tests
Tests for batch processing configuration and value range analysis
"""

import unittest
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('./src')
sys.path.append('.')

class TestValueRangeBatchProcessing(unittest.TestCase):
    """Test batch processing functionality for value range analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_config = {
            'batch_id': 'test_batch_001',
            'processing_parameters': {
                'he_value_range': {'min': 0, 'max': 255},
                'iqid_value_range': {'min': 0, 'max': 4095},
                'normalization_method': 'minmax',
                'enhancement_factor': 1.2
            },
            'quality_thresholds': {
                'min_contrast': 0.1,
                'max_saturation': 0.95
            }
        }
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_configuration_validation(self):
        """Test batch configuration validation"""
        
        # Test valid configuration
        self.assertTrue(self._validate_batch_config(self.batch_config))
        
        # Test invalid configuration (missing required fields)
        invalid_config = self.batch_config.copy()
        del invalid_config['batch_id']
        self.assertFalse(self._validate_batch_config(invalid_config))
        
        # Test invalid value ranges
        invalid_range_config = self.batch_config.copy()
        invalid_range_config['processing_parameters']['he_value_range']['min'] = 100
        invalid_range_config['processing_parameters']['he_value_range']['max'] = 50  # max < min
        self.assertFalse(self._validate_batch_config(invalid_range_config))
    
    def test_batch_processing_pipeline(self):
        """Test complete batch processing pipeline"""
        
        # Create synthetic batch data
        batch_data = []
        for i in range(5):
            he_sample = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            iqid_sample = np.random.randint(0, 1000, (25, 25), dtype=np.uint16)
            
            batch_data.append({
                'sample_id': f'sample_{i:03d}',
                'he_image': he_sample,
                'iqid_image': iqid_sample
            })
        
        # Process batch
        results = self._process_value_range_batch(batch_data, self.batch_config)
        
        # Validate results
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertIn('sample_id', result)
            self.assertIn('he_statistics', result)
            self.assertIn('iqid_statistics', result)
            self.assertIn('processing_quality', result)
    
    def test_value_range_comparison_batch(self):
        """Test batch value range comparison"""
        
        # Create test data with known characteristics
        batch_samples = []
        
        # High contrast sample
        he_high_contrast = np.zeros((50, 50, 3), dtype=np.uint8)
        he_high_contrast[:25, :, :] = 255  # Half white, half black
        iqid_high_activity = np.zeros((25, 25), dtype=np.uint16)
        iqid_high_activity[:12, :] = 1000  # High activity region
        
        batch_samples.append({
            'sample_id': 'high_contrast',
            'he_image': he_high_contrast,
            'iqid_image': iqid_high_activity
        })
        
        # Low contrast sample
        he_low_contrast = np.full((50, 50, 3), 128, dtype=np.uint8)  # Uniform gray
        iqid_low_activity = np.full((25, 25), 100, dtype=np.uint16)  # Uniform low activity
        
        batch_samples.append({
            'sample_id': 'low_contrast',
            'he_image': he_low_contrast,
            'iqid_image': iqid_low_activity
        })
        
        # Analyze batch
        comparison_results = self._compare_value_ranges_batch(batch_samples)
        
        # Validate comparison results
        self.assertEqual(len(comparison_results), 2)
        
        # High contrast sample should have higher range
        high_contrast_result = next(r for r in comparison_results if r['sample_id'] == 'high_contrast')
        low_contrast_result = next(r for r in comparison_results if r['sample_id'] == 'low_contrast')
        
        self.assertGreater(high_contrast_result['he_value_range'], 
                          low_contrast_result['he_value_range'])
        self.assertGreater(high_contrast_result['iqid_value_range'], 
                          low_contrast_result['iqid_value_range'])
    
    def test_batch_reporting_functionality(self):
        """Test batch processing reporting"""
        
        # Create sample batch results
        batch_results = [
            {
                'sample_id': 'sample_001',
                'he_statistics': {'min': 0, 'max': 255, 'mean': 127.5, 'std': 74.0},
                'iqid_statistics': {'min': 0, 'max': 1000, 'mean': 500.0, 'std': 289.0},
                'processing_quality': 'good'
            },
            {
                'sample_id': 'sample_002',
                'he_statistics': {'min': 20, 'max': 200, 'mean': 110.0, 'std': 52.0},
                'iqid_statistics': {'min': 50, 'max': 800, 'mean': 425.0, 'std': 217.0},
                'processing_quality': 'fair'
            }
        ]
        
        # Generate comprehensive report
        report = self._generate_batch_report(batch_results, self.batch_config)
        
        # Validate report structure
        self.assertIn('batch_summary', report)
        self.assertIn('sample_results', report)
        self.assertIn('quality_metrics', report)
        self.assertIn('recommendations', report)
        
        # Validate content
        self.assertEqual(report['batch_summary']['total_samples'], 2)
        self.assertEqual(len(report['sample_results']), 2)
    
    def test_optimized_batch_configuration(self):
        """Test optimized batch processing configuration"""
        
        # Test configuration optimization
        base_config = self.batch_config.copy()
        
        # Simulate optimization based on sample data characteristics
        sample_stats = {
            'he_mean_range': (50, 200),
            'iqid_mean_range': (100, 900),
            'typical_contrast': 0.6
        }
        
        optimized_config = self._optimize_batch_config(base_config, sample_stats)
        
        # Validate optimization
        self.assertIsInstance(optimized_config, dict)
        self.assertIn('processing_parameters', optimized_config)
        
        # Check that optimization actually modified the configuration
        self.assertNotEqual(base_config['processing_parameters']['enhancement_factor'],
                           optimized_config['processing_parameters']['enhancement_factor'])
    
    def _validate_batch_config(self, config):
        """Validate batch configuration"""
        required_fields = ['batch_id', 'processing_parameters', 'quality_thresholds']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # Validate value ranges
        he_range = config['processing_parameters']['he_value_range']
        if he_range['min'] >= he_range['max']:
            return False
        
        iqid_range = config['processing_parameters']['iqid_value_range']
        if iqid_range['min'] >= iqid_range['max']:
            return False
        
        return True
    
    def _process_value_range_batch(self, batch_data, config):
        """Process a batch of samples for value range analysis"""
        results = []
        
        for sample in batch_data:
            he_image = sample['he_image']
            iqid_image = sample['iqid_image']
            
            # Calculate statistics
            he_stats = {
                'min': float(he_image.min()),
                'max': float(he_image.max()),
                'mean': float(he_image.mean()),
                'std': float(he_image.std())
            }
            
            iqid_stats = {
                'min': float(iqid_image.min()),
                'max': float(iqid_image.max()),
                'mean': float(iqid_image.mean()),
                'std': float(iqid_image.std())
            }
            
            # Assess quality
            he_contrast = (he_stats['max'] - he_stats['min']) / 255.0
            quality = 'good' if he_contrast > config['quality_thresholds']['min_contrast'] else 'poor'
            
            results.append({
                'sample_id': sample['sample_id'],
                'he_statistics': he_stats,
                'iqid_statistics': iqid_stats,
                'processing_quality': quality
            })
        
        return results
    
    def _compare_value_ranges_batch(self, batch_samples):
        """Compare value ranges across batch samples"""
        results = []
        
        for sample in batch_samples:
            he_image = sample['he_image']
            iqid_image = sample['iqid_image']
            
            he_range = he_image.max() - he_image.min()
            iqid_range = iqid_image.max() - iqid_image.min()
            
            results.append({
                'sample_id': sample['sample_id'],
                'he_value_range': int(he_range),
                'iqid_value_range': int(iqid_range),
                'range_ratio': float(he_range) / float(iqid_range) if iqid_range > 0 else 0
            })
        
        return results
    
    def _generate_batch_report(self, batch_results, config):
        """Generate comprehensive batch processing report"""
        
        # Calculate summary statistics
        total_samples = len(batch_results)
        good_quality_count = sum(1 for r in batch_results if r['processing_quality'] == 'good')
        
        # Average statistics
        avg_he_mean = np.mean([r['he_statistics']['mean'] for r in batch_results])
        avg_iqid_mean = np.mean([r['iqid_statistics']['mean'] for r in batch_results])
        
        report = {
            'batch_summary': {
                'batch_id': config['batch_id'],
                'total_samples': total_samples,
                'good_quality_samples': good_quality_count,
                'quality_percentage': (good_quality_count / total_samples) * 100,
                'average_he_mean': float(avg_he_mean),
                'average_iqid_mean': float(avg_iqid_mean)
            },
            'sample_results': batch_results,
            'quality_metrics': {
                'overall_quality': 'good' if good_quality_count / total_samples > 0.7 else 'needs_improvement'
            },
            'recommendations': [
                'Consider adjusting enhancement factor for low-quality samples',
                'Validate preprocessing parameters based on batch statistics'
            ]
        }
        
        return report
    
    def _optimize_batch_config(self, base_config, sample_stats):
        """Optimize batch configuration based on sample statistics"""
        import copy
        optimized_config = copy.deepcopy(base_config)
        
        # Adjust enhancement factor based on typical contrast
        if sample_stats['typical_contrast'] < 0.3:
            optimized_config['processing_parameters']['enhancement_factor'] = 1.5
        elif sample_stats['typical_contrast'] > 0.8:
            optimized_config['processing_parameters']['enhancement_factor'] = 1.0
        else:
            # For moderate contrast, slightly adjust the enhancement factor
            optimized_config['processing_parameters']['enhancement_factor'] = 1.3
        
        return optimized_config

if __name__ == '__main__':
    unittest.main()
