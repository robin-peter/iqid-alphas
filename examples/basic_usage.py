#!/usr/bin/env python3
"""
Basic Usage Example for IQID-Alphas

This example demonstrates the basic usage of the iqid_alphas package
for processing iQID camera data.
"""

import numpy as np
import iqid_alphas


def basic_processing_example():
    """Demonstrate basic iQID data processing."""
    print("IQID-Alphas Basic Usage Example")
    print("=" * 40)
    
    # Create sample data (in real usage, you'd load actual iQID data)
    print("1. Creating sample data...")
    sample_data = np.random.poisson(10, (512, 512)).astype(np.float32)
    print(f"   Sample data shape: {sample_data.shape}")
    
    # Initialize the core processor
    print("\n2. Initializing IQID processor...")
    processor = iqid_alphas.IQIDProcessor()
    
    # Process the data
    print("3. Processing data...")
    try:
        processed_data = processor.process(sample_data)
        print(f"   Processed successfully!")
        print(f"   Output shape: {processed_data.shape}")
    except Exception as e:
        print(f"   Processing failed: {e}")
        return
    
    # Initialize visualizer
    print("\n4. Creating visualization...")
    try:
        viz = iqid_alphas.Visualizer()
        viz.plot_activity_map(processed_data)
        print("   Visualization created successfully!")
        
        # Save the plot
        viz.save_figure('examples/output/basic_activity_map.png')
        print("   Plot saved to: examples/output/basic_activity_map.png")
        
    except Exception as e:
        print(f"   Visualization failed: {e}")


def simple_pipeline_example():
    """Demonstrate simple pipeline usage."""
    print("\n" + "=" * 40)
    print("Simple Pipeline Example")
    print("=" * 40)
    
    try:
        # Initialize simple pipeline
        print("1. Initializing simple pipeline...")
        pipeline = iqid_alphas.SimplePipeline()
        
        # Create sample data directory structure (simulated)
        print("2. Processing sample data...")
        # In real usage: results = pipeline.process('/path/to/data')
        print("   (In real usage, provide path to actual data directory)")
        print("   pipeline.process('/path/to/iqid/data')")
        
        print("   Simple pipeline ready for use!")
        
    except Exception as e:
        print(f"   Pipeline initialization failed: {e}")


def advanced_pipeline_example():
    """Demonstrate advanced pipeline with configuration."""
    print("\n" + "=" * 40)
    print("Advanced Pipeline Example")
    print("=" * 40)
    
    try:
        # Create a sample configuration
        print("1. Creating custom configuration...")
        config = {
            'processing': {
                'smooth_sigma': 1.0,
                'threshold': 0.1,
                'background_correction': True
            },
            'visualization': {
                'colormap': 'viridis',
                'save_plots': True,
                'dpi': 300
            }
        }
        print("   Configuration created")
        
        # Initialize advanced pipeline
        print("2. Initializing advanced pipeline...")
        pipeline = iqid_alphas.AdvancedPipeline(config=config)
        
        print("3. Pipeline ready for advanced processing...")
        print("   Use: pipeline.process('/path/to/data', '/path/to/config.json')")
        
    except Exception as e:
        print(f"   Advanced pipeline failed: {e}")


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('examples/output', exist_ok=True)
    
    # Run examples
    basic_processing_example()
    simple_pipeline_example()
    advanced_pipeline_example()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("For more information, see the documentation in docs/")
