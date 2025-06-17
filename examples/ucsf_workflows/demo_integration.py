#!/usr/bin/env python3
"""
UCSF Workflow Integration Demonstration

This script demonstrates the complete integration of UCSF iQID and H&E workflows
with real data discovery, sample matching, and batch processing capabilities.
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ucsf_data_loader import UCSFDataMatcher

def demonstrate_data_discovery():
    """Demonstrate the data discovery and matching capabilities."""
    print("🔬 UCSF Workflow Integration Demonstration")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/unified_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = config.get("data_paths", {}).get("base_path", "")
    
    print(f"📁 Configuration loaded from: {config_path}")
    print(f"📁 UCSF data base path: {base_path}")
    
    if not base_path or not os.path.exists(base_path):
        print(f"⚠️  UCSF data path not available")
        print("   This demonstration requires access to the UCSF dataset")
        print("   The workflows would fall back to simulated data processing")
        return False
    
    try:
        # Initialize data matcher
        print(f"\\n🔍 Initializing data matcher...")
        data_matcher = UCSFDataMatcher(base_path)
        print(f"✅ Data matcher initialized successfully")
        
        # Show discovery results
        available_samples = data_matcher.get_available_samples()
        summary = data_matcher.get_sample_summary()
        
        print(f"\\n📊 Data Discovery Results:")
        print(f"   - Total matched samples: {summary['total_matched_samples']}")
        print(f"   - Kidney samples: {summary['samples_by_tissue']['kidney']}")
        print(f"   - Tumor samples: {summary['samples_by_tissue']['tumor']}")
        print(f"   - Left/Right distribution: {summary['samples_by_side']['L']}/{summary['samples_by_side']['R']}")
        print(f"   - Available iQID locations: {', '.join(summary['available_iqid_locations'])}")
        
        print(f"\\n📋 Matched Samples:")
        print("     Sample ID    Tissue   Side   H&E Files   iQID Locations")
        print("     " + "-" * 60)
        
        for sample_key in available_samples:
            sample_info = data_matcher.get_sample_info(sample_key)
            he_info = sample_info['he']
            iqid_locations = list(sample_info['iqid'].keys())
            
            print(f"     {sample_key:<12} {he_info['tissue_type']:<8} {he_info['side']:<6} "
                  f"{he_info['file_count']:<10} {', '.join(iqid_locations)}")
        
        # Demonstrate data loading for first sample
        if available_samples:
            demo_sample = available_samples[0]
            print(f"\\n🔍 Data Loading Demonstration - Sample: {demo_sample}")
            
            # Load H&E data
            he_data = data_matcher.load_he_data(demo_sample)
            if he_data:
                print(f"   ✅ H&E Data Loaded:")
                print(f"      - Sample ID: {he_data['metadata']['sample_id']}")
                print(f"      - Tissue Type: {he_data['metadata']['tissue_type']}")
                print(f"      - File Count: {he_data['metadata']['file_count']}")
                print(f"      - First few files: {', '.join(he_data['metadata']['files'][:3])}...")
                print(f"      - Source Path: {he_data['source_path']}")
            
            # Load iQID data
            iqid_data = data_matcher.load_iqid_data(demo_sample)
            if iqid_data:
                print(f"   ✅ iQID Data Loaded:")
                print(f"      - Sample ID: {iqid_data['metadata']['sample_id']}")
                print(f"      - Location: {iqid_data['metadata']['location']}")
                print(f"      - Data Type: {iqid_data['metadata']['data_type']}")
                if 'frame_count' in iqid_data['metadata']:
                    print(f"      - Frame Count: {iqid_data['metadata']['frame_count']}")
                if 'image_stack' in iqid_data:
                    print(f"      - Stack Files: {len(iqid_data['image_stack'])} files")
                    print(f"      - First few files: {', '.join([Path(f).name for f in iqid_data['image_stack'][:3]])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_workflow_capabilities():
    """Demonstrate workflow processing capabilities."""
    print(f"\\n🚀 Workflow Processing Capabilities:")
    print(f"\\n📋 Available Processing Scripts:")
    
    scripts = [
        ("test_data_matcher_simple.py", "Test data matching and sample discovery"),
        ("run_all_samples.py", "Automated batch processing of all samples"),
        ("interactive_workflow_runner.py", "Interactive sample selection and processing"),
        ("workflow1_iqid_alignment.py", "iQID alignment workflow (updated for real data)"),
        ("workflow2_he_iqid_coregistration.py", "H&E-iQID co-registration workflow (updated for real data)")
    ]
    
    for script, description in scripts:
        status = "✅" if os.path.exists(script) else "❌"
        print(f"   {status} {script:<35} - {description}")
    
    print(f"\\n📁 Output Organization:")
    print(f"   outputs/")
    print(f"   ├── iqid_aligned/                     # iQID alignment results per sample")
    print(f"   ├── he_iqid_analysis/                 # H&E-iQID co-registration results")
    print(f"   ├── batch_processing_results.json     # Comprehensive batch results") 
    print(f"   └── batch_processing_summary.txt      # Human-readable summary")
    
    print(f"\\n🔧 Configuration Features:")
    print(f"   ✅ Unified config file (configs/unified_config.json)")
    print(f"   ✅ Base path + relative path structure")
    print(f"   ✅ Read-only data protection")
    print(f"   ✅ Flexible file pattern matching")
    print(f"   ✅ Customizable processing parameters")

def demonstrate_sample_processing():
    """Show an example of sample processing workflow."""
    print(f"\\n🔄 Sample Processing Workflow:")
    print(f"\\n   1. Data Discovery & Matching")
    print(f"      └── Scan UCSF directory structure")
    print(f"      └── Match H&E and iQID samples by ID")
    print(f"      └── Validate data availability")
    
    print(f"\\n   2. iQID Alignment Workflow")
    print(f"      └── Load raw iQID event images")
    print(f"      └── Preprocess frames (noise reduction, normalization)")
    print(f"      └── Align frames using phase correlation")
    print(f"      └── Quality control and validation")
    print(f"      └── Save aligned stack and metrics")
    
    print(f"\\n   3. H&E-iQID Co-registration Workflow")
    print(f"      └── Load H&E histology images")
    print(f"      └── Preprocess H&E (stain normalization, contrast)")
    print(f"      └── Register H&E to iQID coordinate system")
    print(f"      └── Segment tissue regions")
    print(f"      └── Map iQID activity to tissue regions")
    print(f"      └── Quantitative analysis and reporting")
    
    print(f"\\n   4. Results Organization")
    print(f"      └── Sample-specific output directories")
    print(f"      └── JSON metadata and processing metrics")
    print(f"      └── Visualization generation")
    print(f"      └── Batch processing summaries")

def main():
    """Main demonstration function."""
    print("\\n" + "=" * 60)
    print("UCSF iQID-H&E Workflow Integration - Complete Demonstration")
    print("=" * 60)
    print("Author: Wookjin Choi <wookjin.choi@jefferson.edu>")
    print("Date: June 2025")
    
    # Demonstrate data discovery
    data_available = demonstrate_data_discovery()
    
    # Show workflow capabilities
    demonstrate_workflow_capabilities()
    
    # Show processing workflow
    demonstrate_sample_processing()
    
    print(f"\\n💡 Usage Examples:")
    print(f"\\n   # Test data matching")
    print(f"   python test_data_matcher_simple.py")
    
    print(f"\\n   # Process all samples automatically")
    print(f"   python run_all_samples.py")
    
    print(f"\\n   # Interactive sample selection")
    print(f"   python interactive_workflow_runner.py")
    
    if data_available:
        print(f"\\n✅ Integration Status: READY")
        print(f"   - Real UCSF data discovered and matched")
        print(f"   - {len(UCSFDataMatcher(os.path.join(os.path.expanduser('~'), 'data/UCSF-Collab/data/')).get_available_samples())} samples available for processing")
        print(f"   - Both workflows updated for real data")
        print(f"   - Batch processing capabilities implemented")
    else:
        print(f"\\n⚠️  Integration Status: SIMULATED MODE")
        print(f"   - UCSF data path not available")
        print(f"   - Workflows will use simulated data")
        print(f"   - All functionality available except real data loading")
    
    print(f"\\n📚 Next Steps:")
    print(f"   1. Run any of the example scripts above")
    print(f"   2. Review outputs in the outputs/ directory")
    print(f"   3. Check README_INTEGRATION.md for detailed documentation")
    print(f"   4. Customize processing parameters in configs/unified_config.json")
    
    print(f"\\n🎉 Integration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
