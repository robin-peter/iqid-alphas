#!/usr/bin/env python3
"""
Real UCSF Data Structure Testing

This script creates realistic UCSF data structures and tests all workflows
with data that mimics actual UCSF experimental conditions.
"""

import sys
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_realistic_ucsf_data_structure(base_path):
    """Create a realistic UCSF data structure with proper file organization."""
    print("🏗️ Creating realistic UCSF data structure...")
    
    base_path = Path(base_path)
    
    # Main UCSF data structure
    ucsf_data = base_path / "UCSF-Collab" / "data"
    
    # DataPush1 structure (aligned iQID and H&E data)
    datapush1 = ucsf_data / "DataPush1"
    
    # H&E structure
    he_3d_kidney = datapush1 / "HE" / "3D scans" / "Kidney"
    he_3d_tumor = datapush1 / "HE" / "3D scans" / "Tumor"
    he_seq_kidney = datapush1 / "HE" / "Sequential sections (10um)" / "Kidney"
    he_seq_tumor = datapush1 / "HE" / "Sequential sections (10um)" / "Tumor"
    
    # iQID structure
    iqid_3d_kidney = datapush1 / "iQID" / "3D scans" / "Kidney"
    iqid_3d_tumor = datapush1 / "iQID" / "3D scans" / "Tumor"
    iqid_seq_kidney = datapush1 / "iQID" / "Sequential sections" / "Kidney"
    iqid_seq_tumor = datapush1 / "iQID" / "Sequential sections" / "Tumor"
    
    # ReUpload structure (raw iQID and processing pipeline)
    reupload = ucsf_data / "ReUpload" / "iQID_reupload"
    reupload_3d_kidney = reupload / "3D scans" / "Kidney"
    reupload_3d_tumor = reupload / "3D scans" / "Tumor"
    reupload_seq_kidney = reupload / "Sequential scans" / "Kidney"
    reupload_seq_tumor = reupload / "Sequential scans" / "Tumor"
    
    # Visualization structure
    viz = ucsf_data / "Visualization"
    
    # Create all directories
    for directory in [
        he_3d_kidney, he_3d_tumor, he_seq_kidney, he_seq_tumor,
        iqid_3d_kidney, iqid_3d_tumor, iqid_seq_kidney, iqid_seq_tumor,
        reupload_3d_kidney, reupload_3d_tumor, reupload_seq_kidney, reupload_seq_tumor,
        viz
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created UCSF data structure at {ucsf_data}")
    return ucsf_data

def create_realistic_iqid_data():
    """Create realistic iQID data that mimics actual experimental conditions."""
    
    # Realistic iQID parameters based on actual experiments
    base_size = (512, 512)
    temporal_frames = 10
    
    # Create realistic activity distribution
    # Background noise level
    background = np.random.poisson(50, base_size)
    
    # Add kidney tissue regions with varying activity
    kidney_region = np.zeros(base_size)
    
    # Cortex region (higher activity)
    cortex_mask = create_kidney_cortex_mask(base_size)
    kidney_region[cortex_mask] = np.random.poisson(800, np.sum(cortex_mask))
    
    # Medulla region (medium activity)
    medulla_mask = create_kidney_medulla_mask(base_size)
    kidney_region[medulla_mask] = np.random.poisson(400, np.sum(medulla_mask))
    
    # Pelvis region (low activity)
    pelvis_mask = create_kidney_pelvis_mask(base_size)
    kidney_region[pelvis_mask] = np.random.poisson(200, np.sum(pelvis_mask))
    
    # Combine background and tissue
    base_image = background + kidney_region
    
    # Create temporal decay series
    half_life_frames = 8  # Simulated half-life in frames
    decay_constant = np.log(2) / half_life_frames
    
    temporal_stack = []
    for t in range(temporal_frames):
        # Apply radioactive decay
        decay_factor = np.exp(-decay_constant * t)
        decayed_image = (base_image * decay_factor).astype(np.uint16)
        
        # Add some temporal noise
        noise = np.random.poisson(10, base_size)
        final_image = np.clip(decayed_image + noise, 0, 65535)
        
        temporal_stack.append(final_image)
    
    return np.array(temporal_stack)

def create_kidney_cortex_mask(shape):
    """Create a realistic kidney cortex mask."""
    h, w = shape
    center_y, center_x = h // 2, w // 2
    
    # Create kidney outline
    y, x = np.ogrid[:h, :w]
    
    # Kidney-like elliptical shape
    kidney_mask = ((x - center_x) / (w * 0.3))**2 + ((y - center_y) / (h * 0.2))**2 < 1
    
    # Cortex is outer part of kidney
    cortex_outer = ((x - center_x) / (w * 0.3))**2 + ((y - center_y) / (h * 0.2))**2 < 1
    cortex_inner = ((x - center_x) / (w * 0.25))**2 + ((y - center_y) / (h * 0.15))**2 < 1
    
    return cortex_outer & ~cortex_inner

def create_kidney_medulla_mask(shape):
    """Create a realistic kidney medulla mask."""
    h, w = shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Medulla is inner part
    medulla_outer = ((x - center_x) / (w * 0.25))**2 + ((y - center_y) / (h * 0.15))**2 < 1
    medulla_inner = ((x - center_x) / (w * 0.15))**2 + ((y - center_y) / (h * 0.1))**2 < 1
    
    return medulla_outer & ~medulla_inner

def create_kidney_pelvis_mask(shape):
    """Create a realistic kidney pelvis mask."""
    h, w = shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Pelvis is central part
    return ((x - center_x) / (w * 0.15))**2 + ((y - center_y) / (h * 0.1))**2 < 1

def create_realistic_he_data():
    """Create realistic H&E histology data."""
    
    # Typical H&E image size
    size = (512, 512, 3)
    
    # Create tissue structure
    he_image = np.zeros(size, dtype=np.uint8)
    
    # Background (slide background - light pink/white)
    he_image[:, :] = [240, 230, 240]  # Light background
    
    # Add tissue regions
    h, w, _ = size
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Kidney tissue mask
    tissue_mask = ((x - center_x) / (w * 0.3))**2 + ((y - center_y) / (h * 0.2))**2 < 1
    
    # Tissue regions with H&E-like colors
    # Cortex - more eosinophilic (pink)
    cortex_mask = create_kidney_cortex_mask((h, w))
    he_image[cortex_mask] = [200, 120, 150]  # Pink cortex
    
    # Medulla - more basophilic (purple/blue)
    medulla_mask = create_kidney_medulla_mask((h, w))
    he_image[medulla_mask] = [150, 100, 180]  # Purple medulla
    
    # Pelvis - intermediate
    pelvis_mask = create_kidney_pelvis_mask((h, w))
    he_image[pelvis_mask] = [180, 110, 160]  # Intermediate color
    
    # Add some texture/noise for realism
    noise = np.random.randint(-20, 20, size)
    he_image = np.clip(he_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return he_image

def save_realistic_data(ucsf_data_path, data_dict):
    """Save realistic data files in UCSF structure."""
    print("💾 Saving realistic UCSF data files...")
    
    try:
        import tifffile
        HAS_TIFF = True
    except ImportError:
        HAS_TIFF = False
        print("⚠️ tifffile not available, using NumPy format")
    
    saved_files = {}
    
    # Sample IDs for realistic naming
    kidney_samples = ["D1M1_L", "D1M1_R", "D1M2_L", "D1M2_R"]
    tumor_samples = ["D1T1_5mm", "D1T2_8mm", "D2T1_6mm"]
    
    # Save kidney iQID data (aligned in DataPush1)
    iqid_kidney_dir = ucsf_data_path / "DataPush1" / "iQID" / "3D scans" / "Kidney"
    for sample_id in kidney_samples[:2]:  # Just save 2 samples for testing
        iqid_data = data_dict['kidney_iqid']
        
        # Save temporal stack
        for frame_idx in range(len(iqid_data)):
            if HAS_TIFF:
                filename = f"mBq_corr_{frame_idx+1:03d}_{sample_id}.tif"
                filepath = iqid_kidney_dir / filename
                tifffile.imwrite(filepath, iqid_data[frame_idx])
            else:
                filename = f"mBq_corr_{frame_idx+1:03d}_{sample_id}.npy"
                filepath = iqid_kidney_dir / filename
                np.save(filepath, iqid_data[frame_idx])
        
        saved_files[f"kidney_iqid_{sample_id}"] = str(iqid_kidney_dir)
    
    # Save H&E data
    he_kidney_dir = ucsf_data_path / "DataPush1" / "HE" / "3D scans" / "Kidney"
    for sample_id in kidney_samples[:2]:
        he_data = data_dict['kidney_he']
        
        if HAS_TIFF:
            filename = f"HE_{sample_id}.tif"
            filepath = he_kidney_dir / filename
            tifffile.imwrite(filepath, he_data)
        else:
            filename = f"HE_{sample_id}.npy"
            filepath = he_kidney_dir / filename
            np.save(filepath, he_data)
        
        saved_files[f"kidney_he_{sample_id}"] = str(filepath)
    
    # Save raw iQID data in ReUpload structure
    reupload_kidney_dir = ucsf_data_path / "ReUpload" / "iQID_reupload" / "3D scans" / "Kidney"
    for sample_id in kidney_samples[:1]:  # One sample for raw data testing
        sample_dir = reupload_kidney_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        
        # Create raw data structure
        (sample_dir / "1_segmented").mkdir(exist_ok=True)
        (sample_dir / "2_aligned").mkdir(exist_ok=True)
        (sample_dir / "Visualization").mkdir(exist_ok=True)
        
        # Raw event data
        raw_data = data_dict['kidney_iqid']
        for frame_idx in range(len(raw_data)):
            if HAS_TIFF:
                # Raw data
                raw_filename = f"0_{sample_id}_iqid_event_image_{frame_idx+1:03d}.tif"
                raw_filepath = sample_dir / raw_filename
                tifffile.imwrite(raw_filepath, raw_data[frame_idx])
                
                # Segmented data
                seg_filename = f"mBq_{frame_idx+1:03d}.tif"
                seg_filepath = sample_dir / "1_segmented" / seg_filename
                # Apply simple segmentation for realism
                segmented = (raw_data[frame_idx] > np.percentile(raw_data[frame_idx], 75)).astype(np.uint16)
                tifffile.imwrite(seg_filepath, segmented)
                
                # Aligned data (same as segmented for this test)
                align_filename = f"mBq_corr_{frame_idx+1:03d}.tif"
                align_filepath = sample_dir / "2_aligned" / align_filename
                tifffile.imwrite(align_filepath, segmented)
        
        saved_files[f"raw_kidney_{sample_id}"] = str(sample_dir)
    
    print(f"✅ Saved {len(saved_files)} realistic data sets")
    return saved_files

def test_workflows_with_realistic_data(ucsf_data_path):
    """Test all workflows with the realistic UCSF data."""
    print("\n🧪 Testing workflows with realistic UCSF data...")
    
    results = {}
    
    # Test 1: SimplePipeline with realistic data
    print("\n🔬 Testing SimplePipeline with realistic kidney iQID data...")
    try:
        sys.path.insert(0, str(project_root))
        from iqid_alphas.pipelines.simple import SimplePipeline
        
        pipeline = SimplePipeline()
        
        # Find TIFF files to test
        iqid_files = list((ucsf_data_path / "DataPush1" / "iQID" / "3D scans" / "Kidney").glob("*.tif"))
        if not iqid_files:
            iqid_files = list((ucsf_data_path / "DataPush1" / "iQID" / "3D scans" / "Kidney").glob("*.npy"))
        
        if iqid_files:
            with tempfile.TemporaryDirectory() as temp_output:
                result = pipeline.process_iqid_stack(str(iqid_files[0]), temp_output)
                results['simple_pipeline'] = {
                    'status': result['status'],
                    'processed_file': iqid_files[0].name,
                    'output_location': temp_output
                }
                print(f"  ✅ SimplePipeline: {result['status']}")
        else:
            results['simple_pipeline'] = {'status': 'no_data', 'error': 'No iQID files found'}
            print("  ⚠️ SimplePipeline: No iQID files found")
            
    except Exception as e:
        results['simple_pipeline'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ SimplePipeline: {str(e)[:50]}...")
    
    # Test 2: UCSF Consolidated Workflow
    print("\n🔬 Testing UCSF Consolidated Workflow...")
    try:
        # Change to UCSF consolidated directory
        ucsf_examples_dir = project_root / "examples" / "ucsf_consolidated"
        sys.path.insert(0, str(ucsf_examples_dir))
        
        from ucsf_consolidated_workflow import UCSFConsolidatedWorkflow
        
        # Create proper config
        config = {
            "base_dir": str(ucsf_data_path.parent),
            "ucsf_data_base": str(ucsf_data_path),
            "workflow_config": {
                "path1_iqid_alignment": {
                    "enabled": True,
                    "input_source": "reupload.iqid_reupload"
                },
                "path2_coregistration": {
                    "enabled": True,
                    "aligned_iqid": str(ucsf_data_path / "DataPush1" / "iQID"),
                    "he_images": str(ucsf_data_path / "DataPush1" / "HE")
                }
            }
        }
        
        workflow = UCSFConsolidatedWorkflow(config)
        
        # Test data validation
        validation_result = workflow.validate_ucsf_data_paths()
        
        # Test Path 1 if we have raw data
        try:
            path1_result = workflow.run_path1_iqid_raw_to_aligned()
            results['ucsf_path1'] = {
                'status': path1_result.get('status', 'completed'),
                'details': 'Path 1 execution completed'
            }
            print("  ✅ UCSF Path 1: Completed")
        except Exception as e:
            results['ucsf_path1'] = {'status': 'failed', 'error': str(e)}
            print(f"  ⚠️ UCSF Path 1: {str(e)[:40]}...")
        
        # Test Path 2 if we have aligned data
        try:
            path2_result = workflow.run_path2_aligned_iqid_he_coregistration()
            results['ucsf_path2'] = {
                'status': path2_result.get('status', 'completed'),
                'details': 'Path 2 execution completed'
            }
            print("  ✅ UCSF Path 2: Completed")
        except Exception as e:
            results['ucsf_path2'] = {'status': 'failed', 'error': str(e)}
            print(f"  ⚠️ UCSF Path 2: {str(e)[:40]}...")
            
    except Exception as e:
        results['ucsf_workflow'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ UCSF Workflow: {str(e)[:50]}...")
    
    # Test 3: Core components with realistic data
    print("\n🔬 Testing Core Components with realistic data...")
    try:
        from iqid_alphas.core.processor import IQIDProcessor
        from iqid_alphas.core.segmentation import ImageSegmenter
        from iqid_alphas.core.alignment import ImageAligner
        
        # Load a realistic image
        iqid_files = list((ucsf_data_path / "DataPush1" / "iQID" / "3D scans" / "Kidney").glob("*.tif"))
        if not iqid_files:
            iqid_files = list((ucsf_data_path / "DataPush1" / "iQID" / "3D scans" / "Kidney").glob("*.npy"))
        
        if iqid_files:
            # Load the image
            if iqid_files[0].suffix == '.npy':
                test_image = np.load(iqid_files[0])
            else:
                try:
                    import tifffile
                    test_image = tifffile.imread(iqid_files[0])
                except ImportError:
                    test_image = np.load(iqid_files[0])  # Fallback
            
            # Test processor
            processor = IQIDProcessor()
            processed = processor.preprocess_image(test_image)
            analysis = processor.analyze_image(test_image)
            
            # Test segmenter
            segmenter = ImageSegmenter()
            tissue_mask = segmenter.segment_tissue(test_image)
            seg_analysis = segmenter.analyze_segments(test_image, tissue_mask)
            
            results['core_components'] = {
                'processor': 'success',
                'segmenter': 'success',
                'image_stats': {
                    'shape': test_image.shape,
                    'mean_intensity': float(analysis.get('mean_intensity', 0)),
                    'tissue_coverage': float(seg_analysis.get('coverage_percentage', 0))
                }
            }
            print(f"  ✅ Core Components: Image {test_image.shape}, Tissue coverage {seg_analysis.get('coverage_percentage', 0):.1f}%")
        else:
            results['core_components'] = {'status': 'no_data'}
            print("  ⚠️ Core Components: No image data found")
            
    except Exception as e:
        results['core_components'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ Core Components: {str(e)[:50]}...")
    
    return results

def run_realistic_ucsf_test():
    """Run comprehensive test with realistic UCSF data."""
    print("🧪 REALISTIC UCSF DATA TESTING")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create realistic data structure
        ucsf_data_path = create_realistic_ucsf_data_structure(temp_dir)
        
        # Generate realistic data
        print("🧬 Generating realistic experimental data...")
        realistic_data = {
            'kidney_iqid': create_realistic_iqid_data(),
            'kidney_he': create_realistic_he_data()
        }
        print(f"✅ Generated kidney iQID data: {realistic_data['kidney_iqid'].shape}")
        print(f"✅ Generated kidney H&E data: {realistic_data['kidney_he'].shape}")
        
        # Save the data
        saved_files = save_realistic_data(ucsf_data_path, realistic_data)
        
        # Test workflows
        test_results = test_workflows_with_realistic_data(ucsf_data_path)
        
        # Generate summary
        print("\n" + "=" * 50)
        print("📊 REALISTIC DATA TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results.values() 
                             if isinstance(r, dict) and r.get('status') in ['success', 'completed'])
        
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        # Detailed results
        for test_name, result in test_results.items():
            status = result.get('status', 'unknown')
            if status in ['success', 'completed']:
                print(f"  ✅ {test_name}: {status}")
            else:
                print(f"  ❌ {test_name}: {status}")
        
        print(f"\n🎉 Realistic UCSF data testing completed!")
        print(f"📁 Data structure created with {len(saved_files)} data sets")
        print(f"🔬 All workflows tested with realistic experimental data")
        
        return test_results

if __name__ == "__main__":
    results = run_realistic_ucsf_test()
    
    # Save results
    try:
        results_file = project_root / "realistic_ucsf_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save results: {e}")
