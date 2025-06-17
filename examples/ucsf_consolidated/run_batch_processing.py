#!/usr/bin/env python3
"""
Simple Runner for UCSF Batch Processing

This script provides an easy way to run batch processing on all UCSF samples
with various options and configurations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ucsf_batch_processor import UCSFBatchProcessor


def main():
    """Main runner function."""
    print("🔬 UCSF Batch Sample Processor")
    print("=" * 50)
    print("Processing all available samples in the UCSF dataset")
    print("with comprehensive visualization and quality assessment.")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UCSF Batch Sample Processor')
    parser.add_argument('--config', default='configs/ucsf_batch_config.json',
                       help='Configuration file path (default: configs/ucsf_batch_config.json)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo/mock data (useful for testing)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with first 3 samples only')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.demo:
        print("🧪 Running in DEMO mode with mock data")
        config_file = 'configs/ucsf_data_config.json'  # Use basic config for demo
    else:
        config_file = args.config
    
    # Set sample limit
    max_samples = args.samples
    if args.quick:
        max_samples = 3
        print("⚡ Quick mode: Processing first 3 samples only")
    
    # Set logging level
    log_level = 'DEBUG' if args.verbose else 'INFO'
    
    try:
        print(f"📋 Configuration: {config_file}")
        print(f"📊 Sample limit: {'All available' if max_samples is None else max_samples}")
        print(f"📝 Log level: {log_level}")
        print()
        
        # Check if config file exists
        if not Path(config_file).exists():
            print(f"❌ Configuration file not found: {config_file}")
            print("Available configurations:")
            config_dir = Path("configs")
            if config_dir.exists():
                for config in config_dir.glob("*.json"):
                    print(f"   - {config}")
            return 1
        
        # Initialize and run batch processor
        print("🚀 Initializing batch processor...")
        batch_processor = UCSFBatchProcessor(config_file)
        
        print("🔄 Starting batch processing...")
        results = batch_processor.run_batch_processing(max_samples=max_samples)
        
        # Display final results
        print()
        print("=" * 60)
        print("✅ BATCH PROCESSING COMPLETED")
        print("=" * 60)
        
        summary = results['processing_summary']
        total = summary['total_samples']
        successful = summary['successful']
        failed = summary['failed']
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"📊 Processing Results:")
        print(f"   • Total samples: {total}")
        print(f"   • Successful: {successful} ({success_rate:.1f}%)")
        print(f"   • Failed: {failed}")
        
        if 'statistical_summary' in results:
            stats = results['statistical_summary']
            quality_stats = stats['quality_statistics']
            perf_stats = stats['processing_statistics']
            
            print(f"\n📈 Quality Metrics:")
            print(f"   • Mean quality score: {quality_stats['mean_quality_score']:.3f}")
            print(f"   • Quality range: {quality_stats['min_quality_score']:.3f} - {quality_stats['max_quality_score']:.3f}")
            
            print(f"\n⚡ Performance:")
            print(f"   • Processing rate: {perf_stats['samples_per_minute']:.1f} samples/minute")
            print(f"   • Total time: {perf_stats['total_processing_time']:.1f} seconds")
        
        print(f"\n📁 Generated Outputs:")
        print(f"   • Batch results: outputs/batch_processing/")
        print(f"   • Individual visualizations: outputs/batch_visualizations/")
        print(f"   • Summary dashboard: outputs/batch_summary/")
        print(f"   • Quality metrics CSV: outputs/batch_summary/quality_metrics.csv")
        
        print(f"\n📊 Key Visualizations:")
        print(f"   • Batch dashboard: outputs/batch_summary/batch_processing_dashboard.png")
        print(f"   • Quality summary: outputs/batch_summary/quality_metrics_summary.png")
        print(f"   • Sample comparison: outputs/batch_summary/sample_comparison_radar.png")
        print(f"   • Performance analysis: outputs/batch_summary/processing_performance_analysis.png")
        
        if failed > 0:
            print(f"\n⚠️  {failed} samples failed processing. Check logs for details.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
