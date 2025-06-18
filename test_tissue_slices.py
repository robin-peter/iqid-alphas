#!/usr/bin/env python3
"""
Test the updated CLI with tissue slice understanding
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def test_tissue_slice_discovery():
    """Test CLI discovery with correct tissue slice understanding."""
    print("🧬 Testing Tissue Slice Discovery")
    print("=" * 40)
    
    try:
        from iqid_alphas.cli import IQIDCLIProcessor
        
        cli_processor = IQIDCLIProcessor()
        print("✅ CLI processor created")
        
        # Test discovery
        test_path = "data/DataPush1"
        print(f"Discovering tissue slices in: {test_path}")
        
        discovered = cli_processor.discover_data(test_path)
        
        print(f"\n📊 Discovery Results:")
        print(f"   iQID samples (slice stacks): {len(discovered['iqid_samples'])}")
        print(f"   H&E samples (slice stacks): {len(discovered['he_samples'])}")
        print(f"   Paired samples: {len(discovered['paired_samples'])}")
        
        # Show sample details
        if discovered['iqid_samples']:
            print(f"\n🔬 Sample Details:")
            for i, sample in enumerate(discovered['iqid_samples'][:3], 1):
                print(f"   {i}. {sample['sample_id']}")
                print(f"      - Tissue: {sample.get('tissue_type', 'unknown')}")
                print(f"      - Preprocessing: {sample.get('preprocessing_type', 'unknown')}")
                print(f"      - Slice count: {sample.get('slice_count', 0)}")
                print(f"      - Can reconstruct 3D: {sample.get('can_reconstruct_3d', False)}")
                print(f"      - Size: {sample.get('size_mb', 0):.1f} MB")
                
                if sample.get('slice_files'):
                    print(f"      - First slice: {sample['slice_files'][0].name}")
                    print(f"      - Last slice: {sample['slice_files'][-1].name}")
                print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_tissue_slice_discovery()
    if success:
        print("🎉 Tissue slice discovery working correctly!")
    else:
        print("❌ Discovery failed")
