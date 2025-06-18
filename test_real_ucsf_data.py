#!/usr/bin/env python3
"""
Comprehensive test script for real UCSF data.
Tests all pipelines and workflows with actual UCSF iQID and H&E data.
"""

import os
import sys
import numpy as np
import traceback
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
try:
    from iqid_alphas.pipelines.simple import SimplePipeline
    from iqid_alphas.core.processor import IQIDProcessor
    from iqid_alphas.core.segmentation import ImageSegmenter
    from iqid_alphas.core.alignment import ImageAligner
    from iqid_alphas.visualization.plotter import Visualizer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class RealUCSFDataTester:
    """Test all components with real UCSF data."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_root = self.project_root / "data"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "data_found": {},
            "pipeline_results": {},
            "performance_metrics": {}
        }
        
        # Real data paths
        self.iqid_data_path = self.data_root / "DataPush1" / "iQID" / "Sequential" / "kidneys"
        self.he_data_path = self.data_root / "DataPush1" / "HE" / "Upper and Lower from 10um Sequential sections"
        self.reupload_path = self.data_root / "ReUpload" / "iQID_reupload"
        
        # Output directory for test results
        self.output_dir = self.project_root / "test_results_real_ucsf"
        self.output_dir.mkdir(exist_ok=True)
        
    def discover_real_data(self):
        """Discover available real UCSF data."""
        print("=== Discovering Real UCSF Data ===")
        
        # Check iQID data
        if self.iqid_data_path.exists():
            iqid_dirs = [d for d in self.iqid_data_path.iterdir() if d.is_dir()]
            iqid_files = []
            for dir_path in iqid_dirs:
                files = list(dir_path.glob("*.tif")) + list(dir_path.glob("*.tiff"))
                iqid_files.extend(files)
            
            self.results["data_found"]["iqid"] = {
                "directories": len(iqid_dirs),
                "files": len(iqid_files),
                "sample_dirs": [str(d.name) for d in iqid_dirs[:5]],
                "sample_files": [str(f.name) for f in iqid_files[:10]]
            }
            print(f"Found {len(iqid_dirs)} iQID directories with {len(iqid_files)} files")
        
        # Check H&E data
        if self.he_data_path.exists():
            he_files = list(self.he_data_path.glob("*.tif")) + list(self.he_data_path.glob("*.tiff"))
            self.results["data_found"]["he"] = {
                "files": len(he_files),
                "sample_files": [str(f.name) for f in he_files[:10]]
            }
            print(f"Found {len(he_files)} H&E files")
        
        # Check reupload data
        if self.reupload_path.exists():
            reup_files = list(self.reupload_path.rglob("*.tif")) + list(self.reupload_path.rglob("*.tiff"))
            self.results["data_found"]["reupload"] = {
                "files": len(reup_files),
                "sample_files": [str(f.name) for f in reup_files[:10]]
            }
            print(f"Found {len(reup_files)} reupload files")
        
        return self.results["data_found"]
    
    def load_and_validate_image(self, image_path):
        """Load and validate a real image file."""
        try:
            # Try with skimage first for scientific TIFFs
            try:
                from skimage import io
                img_array = io.imread(image_path)
            except ImportError:
                # Fallback to PIL for standard images
                img = Image.open(image_path)
                img_array = np.array(img)
            
            # Get basic info
            info = {
                "path": str(image_path),
                "shape": img_array.shape,
                "dtype": str(img_array.dtype),
                "min_val": float(np.min(img_array)),
                "max_val": float(np.max(img_array)),
                "mean_val": float(np.mean(img_array)),
                "file_size_mb": image_path.stat().st_size / (1024 * 1024)
            }
            
            return img_array, info
            
        except Exception as e:
            return None, {"error": str(e), "path": str(image_path)}
    
    def test_simple_pipeline_real_data(self):
        """Test SimplePipeline with real UCSF data."""
        print("\n=== Testing SimplePipeline with Real Data ===")
        
        # Get first available iQID directory
        if not self.iqid_data_path.exists():
            print("No real iQID data found")
            return
        
        iqid_dirs = [d for d in self.iqid_data_path.iterdir() if d.is_dir()]
        if not iqid_dirs:
            print("No iQID directories found")
            return
        
        test_dir = iqid_dirs[0]  # Use first directory
        iqid_files = list(test_dir.glob("*.tif"))
        
        if not iqid_files:
            print(f"No TIFF files in {test_dir}")
            return
        
        print(f"Testing with directory: {test_dir.name}")
        print(f"Found {len(iqid_files)} iQID files")
        
        # Test with first few files
        test_files = iqid_files[:3]  # Test with 3 files
        
        for i, iqid_file in enumerate(test_files):
            try:
                self.results["tests_run"] += 1
                print(f"\nTesting file {i+1}: {iqid_file.name}")
                
                # Load the image
                iqid_data, iqid_info = self.load_and_validate_image(iqid_file)
                if iqid_data is None:
                    print(f"Failed to load {iqid_file}")
                    self.results["tests_failed"] += 1
                    continue
                
                print(f"Loaded iQID: {iqid_info['shape']}, {iqid_info['dtype']}")
                
                # Try to find corresponding H&E data
                he_data = None
                he_info = None
                if self.he_data_path.exists():
                    he_files = list(self.he_data_path.glob("*.tif"))
                    if he_files:
                        he_file = he_files[0]  # Use first H&E file
                        he_data, he_info = self.load_and_validate_image(he_file)
                        if he_data is not None:
                            print(f"Loaded H&E: {he_info['shape']}, {he_info['dtype']}")
                
                # Initialize pipeline
                config = {
                    "processing": {
                        "normalize": True,
                        "remove_outliers": True,
                        "gaussian_filter": True,
                        "gaussian_sigma": 1.0
                    },
                    "segmentation": {
                        "method": "otsu",
                        "min_size": 100,
                        "fill_holes": True
                    },
                    "alignment": {
                        "method": "phase_correlation",
                        "max_shift": 50
                    },
                    "visualization": {
                        "save_plots": True,
                        "output_dir": str(self.output_dir),
                        "dpi": 150
                    }
                }
                
                pipeline = SimplePipeline(config)
                
                # Run pipeline
                start_time = datetime.now()
                
                if he_data is not None:
                    # Run full pipeline with both images
                    results = pipeline.run(iqid_data, he_data)
                else:
                    # Run with just iQID data
                    results = pipeline.run(iqid_data)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                # Store results
                test_key = f"real_data_test_{i+1}_{iqid_file.stem}"
                self.results["pipeline_results"][test_key] = {
                    "iqid_file": str(iqid_file),
                    "iqid_info": iqid_info,
                    "he_info": he_info,
                    "processing_time_seconds": processing_time,
                    "results_keys": list(results.keys()) if results else [],
                    "success": results is not None
                }
                
                if results:
                    print(f"✓ Pipeline succeeded in {processing_time:.2f}s")
                    print(f"  Results: {list(results.keys())}")
                    self.results["tests_passed"] += 1
                else:
                    print(f"✗ Pipeline failed")
                    self.results["tests_failed"] += 1
                
                # Performance metrics
                if test_key not in self.results["performance_metrics"]:
                    self.results["performance_metrics"][test_key] = {}
                
                self.results["performance_metrics"][test_key].update({
                    "image_size_mb": iqid_info.get("file_size_mb", 0),
                    "processing_time": processing_time,
                    "pixels_processed": int(np.prod(iqid_info["shape"])),
                    "pixels_per_second": int(np.prod(iqid_info["shape"]) / processing_time) if processing_time > 0 else 0
                })
                
            except Exception as e:
                print(f"✗ Error testing {iqid_file.name}: {e}")
                self.results["tests_failed"] += 1
                self.results["errors"].append({
                    "test": f"SimplePipeline_{iqid_file.name}",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
    
    def test_core_components_real_data(self):
        """Test individual core components with real data."""
        print("\n=== Testing Core Components with Real Data ===")
        
        # Get a sample real image
        if not self.iqid_data_path.exists():
            print("No real iQID data found")
            return
        
        iqid_dirs = [d for d in self.iqid_data_path.iterdir() if d.is_dir()]
        if not iqid_dirs:
            return
        
        test_dir = iqid_dirs[0]
        iqid_files = list(test_dir.glob("*.tif"))
        if not iqid_files:
            return
        
        iqid_file = iqid_files[0]
        iqid_data, iqid_info = self.load_and_validate_image(iqid_file)
        
        if iqid_data is None:
            print("Failed to load test image")
            return
        
        print(f"Testing with: {iqid_file.name}")
        print(f"Image shape: {iqid_data.shape}, dtype: {iqid_data.dtype}")
        
        # Test IQIDProcessor
        try:
            self.results["tests_run"] += 1
            processor = IQIDProcessor()
            processed = processor.process(iqid_data)
            print(f"✓ IQIDProcessor: {iqid_data.shape} -> {processed.shape}")
            self.results["tests_passed"] += 1
        except Exception as e:
            print(f"✗ IQIDProcessor failed: {e}")
            self.results["tests_failed"] += 1
            self.results["errors"].append({
                "test": "IQIDProcessor_real_data",
                "error": str(e)
            })
        
        # Test ImageSegmenter
        try:
            self.results["tests_run"] += 1
            segmenter = ImageSegmenter()
            segmented = segmenter.segment(iqid_data)
            print(f"✓ ImageSegmenter: unique labels = {len(np.unique(segmented))}")
            self.results["tests_passed"] += 1
        except Exception as e:
            print(f"✗ ImageSegmenter failed: {e}")
            self.results["tests_failed"] += 1
            self.results["errors"].append({
                "test": "ImageSegmenter_real_data",
                "error": str(e)
            })
        
        # Test ImageAligner (need two images)
        if len(iqid_files) > 1:
            try:
                self.results["tests_run"] += 1
                iqid_file2 = iqid_files[1]
                iqid_data2, _ = self.load_and_validate_image(iqid_file2)
                
                if iqid_data2 is not None:
                    aligner = ImageAligner()
                    aligned = aligner.align(iqid_data, iqid_data2)
                    print(f"✓ ImageAligner: aligned shape = {aligned.shape}")
                    self.results["tests_passed"] += 1
                else:
                    raise Exception("Failed to load second image")
                    
            except Exception as e:
                print(f"✗ ImageAligner failed: {e}")
                self.results["tests_failed"] += 1
                self.results["errors"].append({
                    "test": "ImageAligner_real_data",
                    "error": str(e)
                })
        
        # Test Visualizer
        try:
            self.results["tests_run"] += 1
            visualizer = Visualizer()
            viz_config = {"output_dir": str(self.output_dir), "save_plots": True}
            plot_path = visualizer.plot_image(iqid_data, "real_iqid_test", viz_config)
            print(f"✓ Visualizer: saved plot to {plot_path}")
            self.results["tests_passed"] += 1
        except Exception as e:
            print(f"✗ Visualizer failed: {e}")
            self.results["tests_failed"] += 1
            self.results["errors"].append({
                "test": "Visualizer_real_data",
                "error": str(e)
            })
    
    def test_data_formats_and_edge_cases(self):
        """Test various data formats and edge cases with real data."""
        print("\n=== Testing Data Formats and Edge Cases ===")
        
        # Get sample files of different types
        all_files = []
        
        # iQID files
        if self.iqid_data_path.exists():
            for dir_path in self.iqid_data_path.iterdir():
                if dir_path.is_dir():
                    all_files.extend(list(dir_path.glob("*.tif")))
        
        # H&E files
        if self.he_data_path.exists():
            all_files.extend(list(self.he_data_path.glob("*.tif")))
        
        # Reupload files
        if self.reupload_path.exists():
            all_files.extend(list(self.reupload_path.rglob("*.tif")))
        
        if not all_files:
            print("No files found for format testing")
            return
        
        # Test different file sizes
        file_sizes = [(f, f.stat().st_size) for f in all_files]
        file_sizes.sort(key=lambda x: x[1])
        
        # Test smallest, medium, and largest files
        test_files = []
        if file_sizes:
            test_files.append(file_sizes[0][0])  # Smallest
            if len(file_sizes) > 2:
                test_files.append(file_sizes[len(file_sizes)//2][0])  # Medium
            if len(file_sizes) > 1:
                test_files.append(file_sizes[-1][0])  # Largest
        
        for i, test_file in enumerate(test_files):
            try:
                self.results["tests_run"] += 1
                print(f"\nTesting file {i+1}: {test_file.name}")
                
                # Load and analyze
                img_data, img_info = self.load_and_validate_image(test_file)
                if img_data is None:
                    print(f"✗ Failed to load {test_file.name}")
                    self.results["tests_failed"] += 1
                    continue
                
                print(f"  Size: {img_info['file_size_mb']:.2f} MB")
                print(f"  Shape: {img_info['shape']}")
                print(f"  Range: {img_info['min_val']} - {img_info['max_val']}")
                
                # Try processing with SimplePipeline
                config = {
                    "processing": {"normalize": True},
                    "segmentation": {"method": "otsu"},
                    "visualization": {"save_plots": False}
                }
                
                pipeline = SimplePipeline(config)
                results = pipeline.run(img_data)
                
                if results:
                    print(f"✓ Successfully processed {test_file.name}")
                    self.results["tests_passed"] += 1
                else:
                    print(f"✗ Failed to process {test_file.name}")
                    self.results["tests_failed"] += 1
                
            except Exception as e:
                print(f"✗ Error with {test_file.name}: {e}")
                self.results["tests_failed"] += 1
                self.results["errors"].append({
                    "test": f"format_test_{test_file.name}",
                    "error": str(e)
                })
    
    def save_results(self):
        """Save test results to JSON file."""
        results_file = self.output_dir / "real_ucsf_data_test_results.json"
        
        # Calculate summary stats
        total_tests = self.results["tests_run"]
        passed_tests = self.results["tests_passed"]
        failed_tests = self.results["tests_failed"]
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate_percent": round(success_rate, 2),
            "data_directories_found": len(self.results["data_found"]),
            "total_errors": len(self.results["errors"])
        }
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n=== Test Results Saved ===")
        print(f"Results file: {results_file}")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        return results_file
    
    def run_all_tests(self):
        """Run all tests with real UCSF data."""
        print("🧪 Starting Real UCSF Data Testing")
        print("=" * 50)
        
        try:
            # Discover available data
            self.discover_real_data()
            
            # Run tests
            self.test_simple_pipeline_real_data()
            self.test_core_components_real_data()
            self.test_data_formats_and_edge_cases()
            
            # Save results
            self.save_results()
            
            print("\n✅ Real UCSF data testing completed!")
            
        except Exception as e:
            print(f"\n❌ Testing failed with error: {e}")
            traceback.print_exc()
            return False
        
        return True

def main():
    """Main function to run real UCSF data tests."""
    tester = RealUCSFDataTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All real UCSF data tests completed successfully!")
    else:
        print("\n💥 Some tests failed. Check the results for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
