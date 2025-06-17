#!/usr/bin/env python3
"""
UCSF Batch Processing Demo

This script demonstrates the comprehensive batch processing capabilities
of the UCSF workflow system, showing how to process all available samples
in the dataset with individual and summary visualizations.
"""

import os
import sys
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / '..' / 'ucsf_workflows'))

from ucsf_batch_processor import UCSFBatchProcessor


def run_full_demo():
    """Run a comprehensive demonstration of the batch processing system."""
    
    print("🔬 UCSF Batch Processing System Demo")
    print("=" * 60)
    print("This demo shows the complete batch processing workflow:")
    print("• Automatic sample discovery")
    print("• Individual sample processing")
    print("• Quality assessment and validation")
    print("• Comprehensive summary visualizations")
    print()
    
    # Initialize the batch processor
    print("🚀 Initializing batch processor...")
    config_path = 'configs/ucsf_batch_config.json'
    
    try:
        processor = UCSFBatchProcessor(config_path)
        print("✅ Batch processor initialized successfully")
        
        # Run batch processing (limiting to 5 samples for demo)
        print("\n🔄 Running batch processing (demo with 5 samples)...")
        results = processor.run_batch_processing(max_samples=5)
        
        # Display results summary
        print("\n📊 Batch Processing Results:")
        print(f"  Total samples processed: {results['processing_summary']['total_samples']}")
        print(f"  Successful: {results['processing_summary']['successful']}")
        print(f"  Failed: {results['processing_summary']['failed']}")
        
        if 'statistical_summary' in results:
            stats = results['statistical_summary']
            print(f"\n📈 Quality Statistics:")
            print(f"  Mean quality score: {stats['quality_statistics']['mean_quality_score']:.3f}")
            print(f"  Quality range: {stats['quality_statistics']['min_quality_score']:.3f} - {stats['quality_statistics']['max_quality_score']:.3f}")
            
        print(f"\n🎯 Generated Outputs:")
        print(f"  Individual visualizations: outputs/batch_visualizations/")
        print(f"  Summary dashboard: outputs/batch_summary/")
        print(f"  Detailed results: outputs/batch_processing/")
        
        print(f"\n✨ Key Features Demonstrated:")
        print(f"  ✓ Automatic sample discovery")
        print(f"  ✓ Dual-path processing (raw→aligned + coregistration)")
        print(f"  ✓ Individual sample quality assessment")
        print(f"  ✓ Per-sample visualization dashboards")
        print(f"  ✓ Batch summary statistics and plots")
        print(f"  ✓ Quality metrics CSV export")
        print(f"  ✓ Processing performance analysis")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return False


def show_usage_examples():
    """Show various usage examples for the batch processing system."""
    
    print("\n🔧 Usage Examples:")
    print("=" * 40)
    
    examples = [
        ("Process all samples", "python run_batch_processing.py"),
        ("Quick test (3 samples)", "python run_batch_processing.py --quick"),
        ("Demo with mock data", "python run_batch_processing.py --demo"),
        ("Limit to N samples", "python run_batch_processing.py --samples 10"),
        ("Verbose logging", "python run_batch_processing.py --verbose"),
        ("Custom config", "python run_batch_processing.py --config my_config.json")
    ]
    
    for description, command in examples:
        print(f"  {description}:")
        print(f"    {command}")
        print()


def main():
    """Main demo function."""
    # Show usage examples first
    show_usage_examples()
    
    # Ask user if they want to run the demo
    response = input("Would you like to run the batch processing demo? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_full_demo()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("Check the outputs/ directory to see generated visualizations.")
        else:
            print("\n⚠️  Demo encountered issues. Check logs for details.")
    else:
        print("Demo skipped. Use the examples above to run batch processing.")


if __name__ == "__main__":
    main()
