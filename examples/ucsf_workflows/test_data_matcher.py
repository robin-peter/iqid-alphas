#!/usr/bin/env python3
"""
Test the UCSF data matcher to show available samples and their matching.
"""

from ucsf_data_loader import UCSFDataMatcher
import json

def main():
    print("🔬 UCSF Data Matcher Test")
    print("=" * 50)
    
    # Initialize data matcher
    base_path = "/home/wxc151/data/UCSF-Collab/data/"
    matcher = UCSFDataMatcher(base_path)
    
    # Get sample summary
    summary = matcher.get_sample_summary()
    
    print(f"📊 Sample Summary:")
    print(f"   Total matched samples: {summary['total_matched_samples']}")
    print(f"   Kidney samples: {summary['samples_by_tissue']['kidney']}")
    print(f"   Tumor samples: {summary['samples_by_tissue']['tumor']}")
    print(f"   Left side samples: {summary['samples_by_side']['L']}")
    print(f"   Right side samples: {summary['samples_by_side']['R']}")
    print(f"   Available iQID locations: {summary['available_iqid_locations']}")
    
    print(f"\n📋 Detailed Sample Information:")
    for sample in summary['sample_details']:
        print(f"\n🔬 Sample: {sample['sample_key']}")
        print(f"   Tissue: {sample['tissue_type']}")
        print(f"   Side: {sample['side']}")
        print(f"   H&E files: {sample['he_files']}")
        print(f"   iQID locations: {sample['iqid_locations']}")
        for location, data_types in sample['iqid_data_types'].items():
            print(f"   {location} data types: {data_types}")
    
    # Test loading specific sample data
    available_samples = matcher.get_available_samples()
    if available_samples:
        sample_key = available_samples[0]
        print(f"\n🧪 Testing data loading for sample: {sample_key}")
        
        # Load H&E data
        he_data = matcher.load_he_data(sample_key)
        if he_data:
            print(f"   H&E: {len(he_data['images'])} images found")
            print(f"   H&E path: {he_data['path']}")
        
        # Load iQID data
        iqid_data = matcher.load_iqid_data(sample_key, 'raw', 'reupload')
        if iqid_data:
            print(f"   iQID raw: {iqid_data['raw_file']}")
        
        aligned_data = matcher.load_iqid_data(sample_key, 'aligned', 'reupload') 
        if aligned_data:
            print(f"   iQID aligned: {len(aligned_data['image_stack'])} frames")

if __name__ == "__main__":
    main()
