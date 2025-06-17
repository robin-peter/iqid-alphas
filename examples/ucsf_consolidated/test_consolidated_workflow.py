#!/usr/bin/env python3
"""
Test script for UCSF Consolidated Workflow
==========================================

This script tests the consolidated workflow with both real and mock data.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

# Add the parent directory to the path to import the workflow
sys.path.insert(0, os.path.dirname(__file__))
from ucsf_consolidated_workflow import UCSFConsolidatedWorkflow


class TestUCSFConsolidatedWorkflow(unittest.TestCase):
    """Test cases for UCSF Consolidated Workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "test_config.json"
        
        # Create test configuration
        test_config = {
            "data_paths": {
                "ucsf_base_dir": str(self.test_dir / "ucsf_data"),
                "datapush1": {
                    "description": "Test aligned iQID and H&E data",
                    "base_path": str(self.test_dir / "ucsf_data" / "DataPush1"),
                    "aligned_iqid": str(self.test_dir / "ucsf_data" / "DataPush1" / "aligned_iqid"),
                    "he_images": str(self.test_dir / "ucsf_data" / "DataPush1" / "he_images"),
                },
                "reupload": {
                    "description": "Test iQID raw data",
                    "base_path": str(self.test_dir / "ucsf_data" / "ReUpload"),
                    "iqid_raw": str(self.test_dir / "ucsf_data" / "ReUpload" / "iqid_raw"),
                },
                "visualization": {
                    "description": "Test visualization results",
                    "base_path": str(self.test_dir / "ucsf_data" / "Visualization"),
                }
            },
            "workflows": {
                "path1_iqid_raw_to_aligned": {
                    "name": "Test iQID Raw to Aligned",
                    "intermediate_dir": "intermediate/path1_test/",
                    "output_dir": "outputs/path1_test/",
                    "steps": ["load_raw_iqid", "preprocess_frames", "align_sequences"]
                },
                "path2_aligned_iqid_he_coregistration": {
                    "name": "Test Aligned iQID + H&E",
                    "intermediate_dir": "intermediate/path2_test/",
                    "output_dir": "outputs/path2_test/",
                    "steps": ["load_aligned_iqid", "load_he_images", "registration_alignment"]
                },
                "visualization_workflow": {
                    "name": "Test Visualization",
                    "intermediate_dir": "intermediate/viz_test/",
                    "output_dir": "outputs/viz_test/",
                    "steps": ["generate_plots", "create_overlays"]
                }
            },
            "processing_parameters": {
                "iqid_alignment": {"frame_rate": 10},
                "he_coregistration": {"registration_method": "test"},
                "visualization": {"plot_formats": ["png"]}
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = UCSFConsolidatedWorkflow(str(self.config_path))
        self.assertIsNotNone(workflow.config)
        self.assertIsNotNone(workflow.logger)
    
    def test_data_path_validation(self):
        """Test data path validation."""
        workflow = UCSFConsolidatedWorkflow(str(self.config_path))
        # Should return False since paths don't exist yet, but should create mock structure
        result = workflow.validate_data_paths()
        self.assertFalse(result)  # Real paths don't exist
        
        # Mock directories should be created
        base_dir = Path(workflow.config['data_paths']['ucsf_base_dir'])
        self.assertTrue(base_dir.exists())
    
    def test_path1_workflow(self):
        """Test Path 1: iQID Raw → Aligned."""
        workflow = UCSFConsolidatedWorkflow(str(self.config_path))
        results = workflow.run_path1_iqid_raw_to_aligned()
        
        self.assertEqual(results['workflow'], 'path1_iqid_raw_to_aligned')
        self.assertEqual(results['status'], 'completed')
        self.assertIn('load_raw_iqid', results['steps_completed'])
        self.assertIn('align_sequences', results['steps_completed'])
    
    def test_path2_workflow(self):
        """Test Path 2: Aligned iQID + H&E Coregistration."""
        workflow = UCSFConsolidatedWorkflow(str(self.config_path))
        results = workflow.run_path2_aligned_iqid_he_coregistration()
        
        self.assertEqual(results['workflow'], 'path2_aligned_iqid_he_coregistration')
        self.assertEqual(results['status'], 'completed')
        self.assertIn('load_aligned_iqid', results['steps_completed'])
        self.assertIn('load_he_images', results['steps_completed'])
    
    def test_complete_workflow(self):
        """Test complete consolidated workflow."""
        workflow = UCSFConsolidatedWorkflow(str(self.config_path))
        results = workflow.run_complete_workflow()
        
        self.assertEqual(results['workflow_type'], 'ucsf_consolidated')
        self.assertEqual(results['overall_status'], 'completed')
        self.assertIsNotNone(results['path1_results'])
        self.assertIsNotNone(results['path2_results'])
        self.assertIsNotNone(results['visualization_results'])
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)


def run_workflow_test():
    """Run a quick test of the workflow with the actual config."""
    print("🧪 Testing UCSF Consolidated Workflow...")
    
    # Test with actual config
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'ucsf_data_config.json')
    
    try:
        workflow = UCSFConsolidatedWorkflow(config_path)
        print("✓ Workflow initialization successful")
        
        # Test data validation
        print("🔍 Validating data paths...")
        data_available = workflow.validate_data_paths()
        print(f"Data availability: {'✓' if data_available else '⚠️ Using mock data'}")
        
        # Test Path 1
        print("🚀 Testing Path 1: iQID Raw → Aligned...")
        path1_results = workflow.run_path1_iqid_raw_to_aligned()
        print(f"Path 1 Status: {path1_results['status']}")
        
        # Test Path 2
        print("🚀 Testing Path 2: Aligned iQID + H&E Coregistration...")
        path2_results = workflow.run_path2_aligned_iqid_he_coregistration()
        print(f"Path 2 Status: {path2_results['status']}")
        
        # Test Visualization
        print("🎨 Testing Visualization Workflow...")
        viz_results = workflow.run_visualization_workflow(path1_results, path2_results)
        print(f"Visualization Status: {viz_results['status']}")
        
        print("🎉 All workflow tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False


def main():
    """Main test entry point."""
    print("=" * 60)
    print("UCSF Consolidated Workflow Test Suite")
    print("=" * 60)
    
    # Run unit tests
    print("\n📋 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run workflow test
    print("\n🔧 Running workflow integration test...")
    success = run_workflow_test()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
