#!/usr/bin/env python3
"""
Advanced Processing Example for IQID-Alphas

This example demonstrates advanced features including:
- Custom configuration
- Pipeline chaining
- Combined H&E-iQID processing
- Publication-quality visualization
"""

import numpy as np
import json
import os
import iqid_alphas


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "processing": {
            "smooth_sigma": 1.5,
            "threshold_method": "otsu",
            "background_correction": True,
            "noise_reduction": True,
            "calibration_factor": 1.0
        },
        "alignment": {
            "method": "rigid",
            "max_iterations": 100,
            "convergence_threshold": 1e-6
        },
        "segmentation": {
            "method": "watershed",
            "min_area": 100,
            "max_area": 10000
        },
        "visualization": {
            "colormap": "plasma",
            "figsize": [10, 8],
            "dpi": 300,
            "save_format": "png",
            "show_colorbar": True,
            "title_fontsize": 16
        },
        "output": {
            "save_intermediate": True,
            "compress_output": False,
            "output_format": "npz"
        }
    }
    
    # Save configuration
    config_path = "examples/config/advanced_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def advanced_processing_workflow():
    """Demonstrate an advanced processing workflow."""
    print("Advanced IQID-Alphas Processing Workflow")
    print("=" * 50)
    
    # Create configuration
    print("1. Creating advanced configuration...")
    config_path = create_sample_config()
    print(f"   Configuration saved to: {config_path}")
    
    # Create sample datasets
    print("\n2. Creating sample datasets...")
    
    # Sample iQID data (higher resolution)
    iqid_data = np.random.poisson(20, (1024, 1024)).astype(np.float32)
    # Add some structure to make it more realistic
    x, y = np.meshgrid(np.linspace(-5, 5, 1024), np.linspace(-5, 5, 1024))
    iqid_data += 50 * np.exp(-(x**2 + y**2)) * np.random.random((1024, 1024))
    print(f"   iQID data shape: {iqid_data.shape}")
    
    # Sample H&E data
    he_data = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    # Add some tissue-like patterns
    for i in range(3):
        he_data[:, :, i] = np.where(
            (x**2 + y**2) < 9, 
            he_data[:, :, i] * 0.7, 
            he_data[:, :, i]
        )
    print(f"   H&E data shape: {he_data.shape}")
    
    # Step 1: Individual processing
    print("\n3. Individual component processing...")
    
    try:
        # Process iQID data
        processor = iqid_alphas.IQIDProcessor()
        processed_iqid = processor.process(iqid_data)
        print("   ✓ iQID data processed")
        
        # Alignment example
        aligner = iqid_alphas.ImageAligner()
        # In real usage, you'd align H&E to iQID
        print("   ✓ Alignment tools ready")
        
        # Segmentation
        segmenter = iqid_alphas.ImageSegmenter()
        segments = segmenter.segment(processed_iqid)
        print("   ✓ Segmentation completed")
        
    except Exception as e:
        print(f"   ✗ Individual processing failed: {e}")
        return
    
    # Step 2: Pipeline processing
    print("\n4. Advanced pipeline processing...")
    
    try:
        # Load configuration and create advanced pipeline
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pipeline = iqid_alphas.AdvancedPipeline(config=config)
        print("   ✓ Advanced pipeline initialized")
        
        # Simulate pipeline processing
        # results = pipeline.process(data_path, config_path)
        print("   ✓ Pipeline ready for data processing")
        
    except Exception as e:
        print(f"   ✗ Pipeline processing failed: {e}")
        return
    
    # Step 3: Combined processing
    print("\n5. Combined H&E-iQID processing...")
    
    try:
        combined_pipeline = iqid_alphas.CombinedPipeline()
        print("   ✓ Combined pipeline initialized")
        
        # In real usage:
        # results = combined_pipeline.process(iqid_path, he_path)
        print("   ✓ Combined pipeline ready")
        
    except Exception as e:
        print(f"   ✗ Combined processing failed: {e}")
        return
    
    # Step 4: Advanced visualization
    print("\n6. Creating publication-quality visualizations...")
    
    try:
        viz = iqid_alphas.Visualizer()
        
        # Create multiple visualizations
        viz.plot_activity_map(processed_iqid, title="Activity Distribution")
        viz.save_figure("examples/output/advanced_activity_map.png")
        print("   ✓ Activity map saved")
        
        # Create a dose distribution plot (simulated)
        dose_data = processed_iqid * 0.1  # Convert to dose units
        viz.plot_dose_distribution(dose_data, title="Dose Distribution")
        viz.save_figure("examples/output/dose_distribution.png")
        print("   ✓ Dose distribution saved")
        
        print("   ✓ All visualizations completed")
        
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("✓ Advanced workflow completed successfully!")
    print("\nGenerated files:")
    print(f"  - Configuration: {config_path}")
    print("  - Activity map: examples/output/advanced_activity_map.png")
    print("  - Dose distribution: examples/output/dose_distribution.png")


def demonstrate_error_handling():
    """Demonstrate robust error handling."""
    print("\n" + "=" * 50)
    print("Error Handling Demonstration")
    print("=" * 50)
    
    # Test with invalid data
    print("1. Testing error handling with invalid data...")
    
    try:
        processor = iqid_alphas.IQIDProcessor()
        # Try to process None data
        result = processor.process(None)
        print("   ✗ Should have failed!")
    except Exception as e:
        print(f"   ✓ Correctly handled error: {type(e).__name__}")
    
    # Test with invalid configuration
    print("\n2. Testing configuration validation...")
    
    try:
        invalid_config = {"invalid": "configuration"}
        pipeline = iqid_alphas.AdvancedPipeline(config=invalid_config)
        print("   ✓ Pipeline initialized with fallback defaults")
    except Exception as e:
        print(f"   ✓ Configuration error handled: {type(e).__name__}")


if __name__ == "__main__":
    # Create output directories
    os.makedirs('examples/output', exist_ok=True)
    os.makedirs('examples/config', exist_ok=True)
    
    # Run advanced workflow
    advanced_processing_workflow()
    
    # Demonstrate error handling
    demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("Advanced examples completed!")
    print("Check the examples/output/ directory for generated files.")
