#!/usr/bin/env python3
"""
Test script for CLI functionality
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cli_discovery():
    """Test the CLI discovery functionality."""
    print("Testing CLI discovery...")
    
    try:
        from iqid_alphas.cli import IQIDCLIProcessor
        
        cli_processor = IQIDCLIProcessor()
        print("✅ CLI processor created successfully")
        
        # Test with a small dataset
        test_data_path = "data/DataPush1/iQID"
        print(f"Testing discovery with: {test_data_path}")
        
        discovered = cli_processor.discover_data(test_data_path)
        print(f"✅ Discovery completed:")
        print(f"   - iQID files: {len(discovered['iqid_files'])}")
        print(f"   - H&E files: {len(discovered['he_files'])}")
        print(f"   - Paired samples: {len(discovered['paired_samples'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_simple_processing():
    """Test simple pipeline processing."""
    print("\nTesting CLI simple processing...")
    
    try:
        from iqid_alphas.pipelines.simple import SimplePipeline
        import json
        
        # Load config
        with open('configs/cli_quick_config.json', 'r') as f:
            config = json.load(f)
        
        pipeline = SimplePipeline(config)
        print("✅ SimplePipeline created successfully")
        
        # Get one file for testing
        from iqid_alphas.cli import IQIDCLIProcessor
        cli_processor = IQIDCLIProcessor()
        discovered = cli_processor.discover_data("data/DataPush1/iQID")
        
        if discovered['iqid_files']:
            test_file = discovered['iqid_files'][0]
            print(f"Testing with file: {test_file}")
            
            # Create output directory
            output_dir = "results/cli_test_direct"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Test processing
            print("Starting processing...")
            result = pipeline.process_iqid_stack(str(test_file), output_dir)
            print("✅ Processing completed successfully")
            print(f"   Result keys: {list(result.keys())}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("🧪 CLI Test Suite")
    print("=" * 40)
    
    # Test 1: Discovery
    discovery_ok = test_cli_discovery()
    
    # Test 2: Simple processing (only if discovery worked)
    if discovery_ok:
        processing_ok = test_cli_simple_processing()
    else:
        processing_ok = False
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Discovery: {'✅' if discovery_ok else '❌'}")
    print(f"   Processing: {'✅' if processing_ok else '❌'}")
    
    if discovery_ok and processing_ok:
        print("\n🎉 All tests passed! CLI is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
