#!/usr/bin/env python3
"""
Comprehensive Workflow Test Suite for IQID-Alphas

This script tests all available workflows and pipelines in the IQID-Alphas project.
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_simple_pipeline():
    """Test the SimplePipeline."""
    print("🔬 Testing SimplePipeline...")
    try:
        from iqid_alphas.pipelines.simple import SimplePipeline, run_simple_pipeline
        
        # Test instantiation
        pipeline = SimplePipeline()
        print("✅ SimplePipeline instantiated successfully")
        
        # Test basic functionality with mock data
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_simple_pipeline("mock_test.tif", temp_dir)
            print(f"✅ SimplePipeline execution result: {result['status']}")
        
        return True
    except Exception as e:
        print(f"❌ SimplePipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_advanced_pipeline():
    """Test the AdvancedPipeline."""
    print("\n🔬 Testing AdvancedPipeline...")
    try:
        from iqid_alphas.pipelines.advanced import AdvancedPipeline
        
        pipeline = AdvancedPipeline()
        print("✅ AdvancedPipeline instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ AdvancedPipeline test failed: {e}")
        return False

def test_combined_pipeline():
    """Test the CombinedPipeline."""
    print("\n🔬 Testing CombinedPipeline...")
    try:
        from iqid_alphas.pipelines.combined import CombinedPipeline
        
        pipeline = CombinedPipeline()
        print("✅ CombinedPipeline instantiated successfully")
        return True
    except Exception as e:
        print(f"❌ CombinedPipeline test failed: {e}")
        return False

def test_core_components():
    """Test core components."""
    print("\n🔬 Testing Core Components...")
    components_tested = 0
    components_passed = 0
    
    # Test IQIDProcessor
    try:
        from iqid_alphas.core.processor import IQIDProcessor
        processor = IQIDProcessor()
        print("✅ IQIDProcessor instantiated successfully")
        components_passed += 1
    except Exception as e:
        print(f"❌ IQIDProcessor failed: {e}")
    components_tested += 1
    
    # Test ImageSegmenter
    try:
        from iqid_alphas.core.segmentation import ImageSegmenter
        segmenter = ImageSegmenter()
        print("✅ ImageSegmenter instantiated successfully")
        components_passed += 1
    except Exception as e:
        print(f"❌ ImageSegmenter failed: {e}")
    components_tested += 1
    
    # Test ImageAligner
    try:
        from iqid_alphas.core.alignment import ImageAligner
        aligner = ImageAligner()
        print("✅ ImageAligner instantiated successfully")
        components_passed += 1
    except Exception as e:
        print(f"❌ ImageAligner failed: {e}")
    components_tested += 1
    
    # Test Visualizer
    try:
        from iqid_alphas.visualization.plotter import Visualizer
        visualizer = Visualizer()
        print("✅ Visualizer instantiated successfully")
        components_passed += 1
    except Exception as e:
        print(f"❌ Visualizer failed: {e}")
    components_tested += 1
    
    return components_passed, components_tested

def test_ucsf_workflows():
    """Test UCSF workflows."""
    print("\n🔬 Testing UCSF Workflows...")
    
    # Change to examples directory
    examples_dir = project_root / "examples" / "ucsf_consolidated"
    if not examples_dir.exists():
        print("❌ UCSF examples directory not found")
        return False
    
    try:
        sys.path.insert(0, str(examples_dir))
        
        # Test workflow initialization
        from ucsf_consolidated_workflow import UCSFConsolidatedWorkflow
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal config
            config = {
                "base_dir": temp_dir,
                "ucsf_data_base": temp_dir,
                "workflow_config": {
                    "path1_iqid_alignment": {
                        "enabled": True
                    },
                    "path2_coregistration": {
                        "enabled": True  
                    }
                }
            }
            
            workflow = UCSFConsolidatedWorkflow(config)
            print("✅ UCSFConsolidatedWorkflow instantiated successfully")
            
            # Test data validation
            workflow.validate_ucsf_data_paths()
            print("✅ Data path validation completed")
            
        return True
    except Exception as e:
        print(f"❌ UCSF workflow test failed: {e}")
        return False

def test_package_imports():
    """Test main package imports."""
    print("\n🔬 Testing Package Imports...")
    try:
        import iqid_alphas
        print("✅ Main iqid_alphas package imported successfully")
        
        # Test main components
        processor = iqid_alphas.IQIDProcessor()
        print("✅ IQIDProcessor accessible from main package")
        
        simple_pipeline = iqid_alphas.SimplePipeline()
        print("✅ SimplePipeline accessible from main package")
        
        return True
    except Exception as e:
        print(f"❌ Package import test failed: {e}")
        return False

def run_all_tests():
    """Run all workflow tests."""
    print("=" * 60)
    print("🧪 IQID-Alphas Comprehensive Workflow Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test individual components
    tests = [
        ("Package Imports", test_package_imports),
        ("Simple Pipeline", test_simple_pipeline),
        ("Advanced Pipeline", test_advanced_pipeline),
        ("Combined Pipeline", test_combined_pipeline),
        ("UCSF Workflows", test_ucsf_workflows)
    ]
    
    for test_name, test_func in tests:
        total_tests += 1
        try:
            if test_func():
                tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    # Test core components
    print("\n" + "=" * 40)
    core_passed, core_total = test_core_components()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Main Tests: {tests_passed}/{total_tests} passed")
    print(f"Core Components: {core_passed}/{core_total} passed")
    print(f"Overall Success Rate: {(tests_passed + core_passed)}/{(total_tests + core_total)} ({((tests_passed + core_passed)/(total_tests + core_total)*100):.1f}%)")
    
    if tests_passed == total_tests and core_passed == core_total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
