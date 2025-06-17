#!/usr/bin/env python3
"""
Test script for UCSF workflows

This script validates that the UCSF workflows can be imported and executed
without errors using simulated data.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

def test_workflow_imports():
    """Test that all workflow modules can be imported."""
    print("Testing workflow imports...")
    
    try:
        from workflow1_iqid_alignment import UCSFiQIDWorkflow
        print("  ✓ Workflow 1 import successful")
    except ImportError as e:
        print(f"  ✗ Workflow 1 import failed: {e}")
        return False
    
    try:
        from workflow2_he_iqid_coregistration import UCSFHEiQIDWorkflow
        print("  ✓ Workflow 2 import successful")
    except ImportError as e:
        print(f"  ✗ Workflow 2 import failed: {e}")
        return False
    
    try:
        from run_complete_pipeline import UCSFCompletePipeline
        print("  ✓ Complete pipeline import successful")
    except ImportError as e:
        print(f"  ✗ Complete pipeline import failed: {e}")
        return False
    
    return True

def test_workflow_initialization():
    """Test that workflows can be initialized."""
    print("\nTesting workflow initialization...")
    
    try:
        from workflow1_iqid_alignment import UCSFiQIDWorkflow
        workflow1 = UCSFiQIDWorkflow()
        print("  ✓ Workflow 1 initialization successful")
    except Exception as e:
        print(f"  ✗ Workflow 1 initialization failed: {e}")
        return False
    
    try:
        from workflow2_he_iqid_coregistration import UCSFHEiQIDWorkflow
        workflow2 = UCSFHEiQIDWorkflow()
        print("  ✓ Workflow 2 initialization successful")
    except Exception as e:
        print(f"  ✗ Workflow 2 initialization failed: {e}")
        return False
    
    try:
        from run_complete_pipeline import UCSFCompletePipeline
        complete_pipeline = UCSFCompletePipeline()
        print("  ✓ Complete pipeline initialization successful")
    except Exception as e:
        print(f"  ✗ Complete pipeline initialization failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    
    # Test config files exist
    config_files = [
        "configs/iqid_alignment_config.json",
        "configs/he_iqid_config.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"  ✓ Config file exists: {config_file}")
        else:
            print(f"  ✗ Config file missing: {config_file}")
            return False
    
    # Test config loading
    try:
        from workflow1_iqid_alignment import UCSFiQIDWorkflow
        workflow = UCSFiQIDWorkflow()
        config = workflow.config
        print(f"  ✓ Workflow 1 config loaded: {len(config)} sections")
    except Exception as e:
        print(f"  ✗ Workflow 1 config loading failed: {e}")
        return False
    
    return True

def test_simulated_data_generation():
    """Test simulated data generation."""
    print("\nTesting simulated data generation...")
    
    try:
        from workflow1_iqid_alignment import UCSFiQIDWorkflow
        workflow = UCSFiQIDWorkflow()
        
        # Test simulated iQID data
        iqid_data = workflow._create_simulated_iqid_stack()
        print(f"  ✓ Simulated iQID data: {iqid_data['image_stack'].shape}")
        
        from workflow2_he_iqid_coregistration import UCSFHEiQIDWorkflow
        workflow2 = UCSFHEiQIDWorkflow()
        
        # Test simulated H&E data
        he_data = workflow2._create_simulated_he_images()
        print(f"  ✓ Simulated H&E data: {he_data['he_image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simulated data generation failed: {e}")
        return False

def test_basic_processing():
    """Test basic processing functionality."""
    print("\nTesting basic processing...")
    
    try:
        import iqid_alphas
        import numpy as np
        
        # Test core components used by workflows
        processor = iqid_alphas.IQIDProcessor()
        aligner = iqid_alphas.ImageAligner()
        segmenter = iqid_alphas.ImageSegmenter()
        visualizer = iqid_alphas.Visualizer()
        
        # Test with small sample data
        sample_data = np.random.random((50, 50)).astype(np.float32)
        
        # Test processing
        processed = processor.process(sample_data)
        print(f"  ✓ Basic processing: {sample_data.shape} -> {processed.shape}")
        
        # Test alignment
        aligned = aligner.align(sample_data, sample_data)
        print(f"  ✓ Basic alignment: {aligned.shape}")
        
        # Test segmentation
        segments = segmenter.segment(sample_data)
        print(f"  ✓ Basic segmentation: {segments.shape}")
        
        # Test visualization (without display)
        visualizer.plot_activity_map(sample_data)
        print("  ✓ Basic visualization")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic processing test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "data",
        "configs",
        "intermediate",
        "outputs"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ Directory exists: {dir_name}")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Directory created: {dir_name}")
            except Exception as e:
                print(f"  ✗ Cannot create directory {dir_name}: {e}")
                return False
    
    return True

def main():
    """Run all tests."""
    print("🔬 UCSF Workflows Validation Test")
    print("=" * 50)
    
    tests = [
        ("Workflow Imports", test_workflow_imports),
        ("Workflow Initialization", test_workflow_initialization),
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
        ("Simulated Data Generation", test_simulated_data_generation),
        ("Basic Processing", test_basic_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n💥 {test_name} ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ All tests passed! UCSF workflows are ready to use.")
        print("\nNext steps:")
        print("1. Add your UCSF data to data/ directories")
        print("2. Run: python run_complete_pipeline.py")
        print("3. Check outputs/ for results")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
