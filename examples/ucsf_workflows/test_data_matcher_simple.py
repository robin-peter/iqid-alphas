#!/usr/bin/env python3
"""
Test script for UCSF data matcher and sample processing.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ucsf_data_loader import UCSFDataMatcher
import json

def test_data_matcher():
    """Test the UCSF data matcher functionality."""
    print("🔬 UCSF Data Matcher Test")
    print("=" * 50)
    
    # Load config to get base path
    config_path = "configs/unified_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = config.get("data_paths", {}).get("base_path", "")
    
    if not base_path or not os.path.exists(base_path):
        print(f"⚠️  UCSF base path not available: {base_path}")
        print("Data matcher cannot be initialized without UCSF data")
        return
    
    print(f"📁 UCSF base path: {base_path}")
    
    try:
        # Initialize data matcher
        data_matcher = UCSFDataMatcher(base_path)
        print(f"✅ Data matcher initialized successfully")
        
        # Get available samples
        available_samples = data_matcher.get_available_samples()
        print(f"📊 Found {len(available_samples)} available samples")
        
        if available_samples:
            # Get sample summary
            summary = data_matcher.get_sample_summary()
            print(f"\n📋 Sample Summary:")
            print(f"   - Total matched samples: {summary['total_matched_samples']}")
            print(f"   - Kidney samples: {summary['samples_by_tissue']['kidney']}")
            print(f"   - Tumor samples: {summary['samples_by_tissue']['tumor']}")
            print(f"   - Left/Right: {summary['samples_by_side']['L']}/{summary['samples_by_side']['R']}")
            print(f"   - Available iQID locations: {', '.join(summary['available_iqid_locations'])}")
            
            print(f"\n📋 Available Samples:")
            for i, sample_key in enumerate(available_samples[:10], 1):  # Show first 10
                sample_info = data_matcher.get_sample_info(sample_key)
                he_info = sample_info['he']
                iqid_locations = list(sample_info['iqid'].keys())
                print(f"   {i:2d}. {sample_key} ({he_info['tissue_type']}, {he_info['side']}) - iQID: {', '.join(iqid_locations)}")
            
            if len(available_samples) > 10:
                print(f"   ... and {len(available_samples) - 10} more")
            
            # Test loading data for first sample
            if available_samples:
                test_sample = available_samples[0]
                print(f"\n🔍 Testing data loading for sample: {test_sample}")
                
                # Test H&E data loading
                he_data = data_matcher.load_he_data(test_sample)
                if he_data:
                    print(f"   ✅ H&E data loaded: {he_data['metadata']}")
                else:
                    print(f"   ❌ Failed to load H&E data")
                
                # Test iQID data loading
                iqid_data = data_matcher.load_iqid_data(test_sample)
                if iqid_data:
                    print(f"   ✅ iQID data loaded: {iqid_data['metadata']}")
                else:
                    print(f"   ❌ Failed to load iQID data")
        else:
            print("⚠️  No matched samples found")
            
    except Exception as e:
        print(f"❌ Error initializing data matcher: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_matcher()
