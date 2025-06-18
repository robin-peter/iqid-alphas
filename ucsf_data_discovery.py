#!/usr/bin/env python3
"""
UCSF Data Discovery and Testing Script

This script searches for actual UCSF data in common locations and tests
workflows with any real data found.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def search_for_ucsf_data():
    """Search for actual UCSF data in common locations."""
    print("🔍 Searching for actual UCSF data...")
    
    # Common locations where UCSF data might be stored
    search_paths = [
        Path("/home/wxc151/iqid-alphas/examples/data"),
        Path("/home/wxc151/iqid-alphas/data"),
        Path("/data"),
        Path("/mnt/data"),
        Path("/opt/data"),
        Path.home() / "data",
        Path.home() / "Documents" / "data",
        Path("/tmp/ucsf_data"),
        # Add the examples directory structure
        Path("/home/wxc151/iqid-alphas/examples/ucsf_consolidated/Contains aligned iQID and H&E data"),
        Path("/home/wxc151/iqid-alphas/examples/ucsf_consolidated/Contains iQID raw, segmented, and manually aligned data"),
        Path("/home/wxc151/iqid-alphas/examples/ucsf_consolidated/Test aligned iQID and H&E data"),
        Path("/home/wxc151/iqid-alphas/examples/ucsf_consolidated/Test iQID raw data"),
    ]
    
    found_data = {}
    
    for search_path in search_paths:
        if search_path.exists():
            print(f"  📁 Checking: {search_path}")
            
            # Look for TIFF files
            tiff_files = list(search_path.rglob("*.tif")) + list(search_path.rglob("*.tiff"))
            if tiff_files:
                found_data[f"tiff_{search_path.name}"] = {
                    'path': str(search_path),
                    'files': [str(f) for f in tiff_files[:5]],  # First 5 files
                    'count': len(tiff_files),
                    'type': 'tiff'
                }
                print(f"    ✅ Found {len(tiff_files)} TIFF files")
            
            # Look for NumPy files
            npy_files = list(search_path.rglob("*.npy"))
            if npy_files:
                found_data[f"npy_{search_path.name}"] = {
                    'path': str(search_path),
                    'files': [str(f) for f in npy_files[:5]],
                    'count': len(npy_files),
                    'type': 'numpy'
                }
                print(f"    ✅ Found {len(npy_files)} NumPy files")
            
            # Look for other image formats
            other_files = (list(search_path.rglob("*.png")) + 
                          list(search_path.rglob("*.jpg")) + 
                          list(search_path.rglob("*.jpeg")))
            if other_files:
                found_data[f"images_{search_path.name}"] = {
                    'path': str(search_path),
                    'files': [str(f) for f in other_files[:5]],
                    'count': len(other_files),
                    'type': 'images'
                }
                print(f"    ✅ Found {len(other_files)} image files")
            
            # Look for specific UCSF naming patterns
            ucsf_patterns = ['*mBq*', '*iqid*', '*kidney*', '*tumor*', '*HE*', '*D1M*', '*D1T*']
            for pattern in ucsf_patterns:
                pattern_files = list(search_path.rglob(pattern))
                if pattern_files:
                    found_data[f"pattern_{pattern}_{search_path.name}"] = {
                        'path': str(search_path),
                        'files': [str(f) for f in pattern_files[:3]],
                        'count': len(pattern_files),
                        'type': 'pattern_match',
                        'pattern': pattern
                    }
                    print(f"    ✅ Found {len(pattern_files)} files matching '{pattern}'")
    
    return found_data

def analyze_found_data(found_data):
    """Analyze the structure and content of found data."""
    print(f"\n📊 Analyzing found data...")
    
    analysis = {
        'total_locations': len(found_data),
        'file_types': {},
        'potential_samples': [],
        'data_structure': {}
    }
    
    for data_key, data_info in found_data.items():
        file_type = data_info['type']
        if file_type not in analysis['file_types']:
            analysis['file_types'][file_type] = 0
        analysis['file_types'][file_type] += data_info['count']
        
        # Try to identify sample names and structure
        for file_path in data_info['files']:
            file_path_obj = Path(file_path)
            
            # Look for UCSF sample naming patterns
            filename = file_path_obj.name
            if any(pattern in filename.lower() for pattern in ['d1m', 'd1t', 'kidney', 'tumor']):
                analysis['potential_samples'].append({
                    'filename': filename,
                    'path': str(file_path_obj.parent),
                    'type': 'ucsf_sample'
                })
            
            # Analyze directory structure
            parts = file_path_obj.parts
            if len(parts) > 3:
                structure_key = ' > '.join(parts[-4:-1])  # Last few directory levels
                if structure_key not in analysis['data_structure']:
                    analysis['data_structure'][structure_key] = 0
                analysis['data_structure'][structure_key] += 1
    
    # Remove duplicates from potential samples
    unique_samples = []
    seen_names = set()
    for sample in analysis['potential_samples']:
        if sample['filename'] not in seen_names:
            unique_samples.append(sample)
            seen_names.add(sample['filename'])
    analysis['potential_samples'] = unique_samples[:10]  # Limit to first 10
    
    return analysis

def test_with_real_data(found_data):
    """Test workflows with any real data that was found."""
    print(f"\n🧪 Testing workflows with found real data...")
    
    test_results = {}
    
    if not found_data:
        print("  ⚠️ No real data found to test with")
        return {'status': 'no_data', 'message': 'No real UCSF data found'}
    
    # Test SimplePipeline with found data
    print("\n🔬 Testing SimplePipeline with real data...")
    try:
        from iqid_alphas.pipelines.simple import SimplePipeline
        pipeline = SimplePipeline()
        
        # Find the best data to test with
        for data_key, data_info in found_data.items():
            if data_info['type'] in ['tiff', 'numpy'] and data_info['files']:
                test_file = data_info['files'][0]
                
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_output:
                        result = pipeline.process_iqid_stack(test_file, temp_output)
                        test_results[f'simple_pipeline_{data_key}'] = {
                            'status': result['status'],
                            'file_tested': test_file,
                            'result': result
                        }
                        print(f"    ✅ Tested {Path(test_file).name}: {result['status']}")
                        break  # Test with first available file
                except Exception as e:
                    test_results[f'simple_pipeline_{data_key}'] = {
                        'status': 'failed',
                        'file_tested': test_file,
                        'error': str(e)
                    }
                    print(f"    ❌ Failed with {Path(test_file).name}: {str(e)[:40]}...")
                    continue  # Try next file
    except Exception as e:
        test_results['simple_pipeline'] = {'status': 'import_failed', 'error': str(e)}
        print(f"    ❌ SimplePipeline import failed: {e}")
    
    # Test core components with real data
    print("\n🔬 Testing core components with real data...")
    try:
        from iqid_alphas.core.processor import IQIDProcessor
        from iqid_alphas.core.segmentation import ImageSegmenter
        
        processor = IQIDProcessor()
        segmenter = ImageSegmenter()
        
        for data_key, data_info in found_data.items():
            if data_info['type'] in ['tiff', 'numpy'] and data_info['files']:
                test_file = data_info['files'][0]
                
                try:
                    # Load the data
                    if test_file.endswith('.npy'):
                        import numpy as np
                        test_data = np.load(test_file)
                    else:
                        try:
                            import tifffile
                            test_data = tifffile.imread(test_file)
                        except ImportError:
                            continue  # Skip if can't load
                    
                    # Test processor
                    if test_data.ndim > 2:
                        test_data = test_data[0]  # Use first slice if 3D
                    
                    analysis = processor.analyze_image(test_data)
                    tissue_mask = segmenter.segment_tissue(test_data)
                    
                    test_results[f'core_components_{data_key}'] = {
                        'status': 'success',
                        'file_tested': test_file,
                        'image_shape': test_data.shape,
                        'mean_intensity': float(analysis.get('mean_intensity', 0)),
                        'tissue_detected': bool(np.sum(tissue_mask) > 0)
                    }
                    print(f"    ✅ Analyzed {Path(test_file).name}: shape {test_data.shape}")
                    break  # Test with first successful file
                    
                except Exception as e:
                    test_results[f'core_components_{data_key}'] = {
                        'status': 'failed',
                        'file_tested': test_file,
                        'error': str(e)
                    }
                    print(f"    ❌ Failed analyzing {Path(test_file).name}: {str(e)[:40]}...")
                    continue
    except Exception as e:
        test_results['core_components'] = {'status': 'import_failed', 'error': str(e)}
        print(f"    ❌ Core components import failed: {e}")
    
    return test_results

def run_ucsf_data_discovery():
    """Run complete UCSF data discovery and testing."""
    print("🔍 UCSF DATA DISCOVERY AND TESTING")
    print("=" * 50)
    
    # Search for data
    found_data = search_for_ucsf_data()
    
    if found_data:
        print(f"\n✅ Found data in {len(found_data)} locations")
        
        # Analyze the data
        analysis = analyze_found_data(found_data)
        
        print(f"\n📊 DATA ANALYSIS:")
        print(f"  📁 Total locations: {analysis['total_locations']}")
        print(f"  📄 File types found:")
        for file_type, count in analysis['file_types'].items():
            print(f"    • {file_type}: {count} files")
        
        if analysis['potential_samples']:
            print(f"  🧬 Potential UCSF samples:")
            for sample in analysis['potential_samples'][:5]:  # Show first 5
                print(f"    • {sample['filename']} in {Path(sample['path']).name}")
        
        if analysis['data_structure']:
            print(f"  🏗️ Directory structures found:")
            for structure, count in list(analysis['data_structure'].items())[:3]:  # Show top 3
                print(f"    • {structure}: {count} files")
        
        # Test with real data
        test_results = test_with_real_data(found_data)
        
        # Summary
        print(f"\n📊 TESTING SUMMARY:")
        successful_tests = sum(1 for r in test_results.values() 
                             if isinstance(r, dict) and r.get('status') == 'success')
        total_tests = len(test_results)
        
        print(f"  🧪 Tests run: {total_tests}")
        print(f"  ✅ Successful: {successful_tests}")
        print(f"  ❌ Failed: {total_tests - successful_tests}")
        
        if successful_tests > 0:
            print(f"\n🎉 Successfully tested workflows with real UCSF data!")
        else:
            print(f"\n⚠️ No successful tests with real data (may need proper data format)")
    else:
        print(f"\n⚠️ No UCSF data found in searched locations")
        print(f"💡 Consider:")
        print(f"  • Adding actual UCSF data to the examples directory")
        print(f"  • Running the realistic data test instead")
        print(f"  • Checking if data is in external storage")
    
    # Save results
    try:
        results = {
            'found_data': found_data,
            'analysis': analysis if found_data else {},
            'test_results': test_results if found_data else {}
        }
        
        results_file = project_root / "ucsf_data_discovery_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")
    
    return found_data, test_results if found_data else {}

if __name__ == "__main__":
    found_data, test_results = run_ucsf_data_discovery()
    
    if not found_data:
        print(f"\n🚀 Running realistic UCSF data test instead...")
        # Import and run the realistic test
        try:
            from test_realistic_ucsf_data import run_realistic_ucsf_test
            realistic_results = run_realistic_ucsf_test()
        except Exception as e:
            print(f"❌ Could not run realistic test: {e}")
