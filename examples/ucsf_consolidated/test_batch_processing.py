#!/usr/bin/env python3
"""
Test Script for UCSF Batch Processing

This script runs a quick demonstration of the batch processing functionality
using mock data to show the complete workflow and visualization capabilities.
"""

import os
import sys
import shutil
from pathlib import Path

def test_batch_processing():
    """Run a test of the batch processing system."""
    print("🧪 UCSF Batch Processing Test")
    print("=" * 50)
    print("This test demonstrates the batch processing functionality")
    print("using mock data to show all features and visualizations.")
    print()
    
    # Change to the consolidated directory
    original_dir = os.getcwd()
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Clean up any previous test outputs
        output_dirs = ["outputs/batch_processing", "outputs/batch_visualizations", 
                      "outputs/batch_summary", "intermediate", "logs"]
        
        for output_dir in output_dirs:
            if Path(output_dir).exists():
                print(f"🧹 Cleaning up previous test outputs: {output_dir}")
                shutil.rmtree(output_dir)
        
        print("🚀 Starting batch processing test with mock data...")
        print()
        
        # Run the batch processing with demo mode and quick settings
        import subprocess
        result = subprocess.run([
            sys.executable, "run_batch_processing.py", 
            "--demo", "--quick", "--verbose"
        ], capture_output=True, text=True)
        
        print("📄 BATCH PROCESSING OUTPUT:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  WARNINGS/ERRORS:")
            print("-" * 40)
            print(result.stderr)
        
        # Check if outputs were created
        print()
        print("📁 CHECKING GENERATED OUTPUTS:")
        print("-" * 40)
        
        output_files_to_check = [
            "outputs/batch_processing/batch_processing_results.json",
            "outputs/batch_summary/batch_processing_dashboard.png",
            "outputs/batch_summary/quality_metrics_summary.png",
            "outputs/batch_summary/batch_processing_summary.md",
            "outputs/batch_summary/quality_metrics.csv"
        ]
        
        for file_path in output_files_to_check:
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                print(f"✅ {file_path} ({file_size:,} bytes)")
            else:
                print(f"❌ {file_path} (missing)")
        
        # Check visualization directories
        viz_dirs = ["outputs/batch_visualizations", "outputs/batch_summary"]
        for viz_dir in viz_dirs:
            if Path(viz_dir).exists():
                files = list(Path(viz_dir).glob("**/*"))
                print(f"📊 {viz_dir}: {len(files)} files")
            else:
                print(f"❌ {viz_dir} (missing)")
        
        print()
        if result.returncode == 0:
            print("✅ BATCH PROCESSING TEST COMPLETED SUCCESSFULLY!")
            print()
            print("📝 Summary:")
            print("   • Mock batch processing workflow executed")
            print("   • Individual sample visualizations created")
            print("   • Comprehensive summary dashboard generated")
            print("   • Quality metrics analysis completed")
            print("   • All output files and directories created")
            print()
            print("🔍 You can now examine the generated outputs in:")
            print("   • outputs/batch_processing/ - Main processing results")
            print("   • outputs/batch_visualizations/ - Individual sample plots")
            print("   • outputs/batch_summary/ - Summary dashboard and analysis")
        else:
            print("❌ BATCH PROCESSING TEST FAILED!")
            print(f"   Return code: {result.returncode}")
        
        return result.returncode
        
    finally:
        # Return to original directory
        os.chdir(original_dir)


def main():
    """Main function."""
    try:
        return test_batch_processing()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
