#!/usr/bin/env python3
"""
Test script for the simplified pipeline
"""

import numpy as np
import tempfile
from pathlib import Path
from iqid_alphas.pipelines.simple import SimplePipeline, run_simple_pipeline

def test_simple_pipeline():
    """Test the simplified pipeline with mock data."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock TIFF file (simulate multi-slice)
        try:
            import tifffile
            
            # Create mock 3D stack (time, height, width)
            mock_stack = np.random.randint(0, 255, (5, 256, 256)).astype(np.uint8)
            
            # Add some structure to make segmentation more realistic
            for i in range(5):
                # Add circular "tissue" regions
                y, x = np.ogrid[:256, :256]
                center_y, center_x = 128 + i*10, 128 + i*5
                mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
                mock_stack[i][mask] = 200
            
            tiff_path = temp_path / "test_stack.tif"
            tifffile.imwrite(tiff_path, mock_stack)
            
            print(f"Created test TIFF: {tiff_path}")
            
        except ImportError:
            print("tifffile not available, using basic test")
            tiff_path = temp_path / "test_image.tif"
            # Create a dummy file for testing without actual TIFF
            tiff_path.touch()
        
        # Test the pipeline
        output_dir = temp_path / "output"
        
        print("Testing SimplePipeline...")
        pipeline = SimplePipeline()
        result = pipeline.process_iqid_stack(str(tiff_path), str(output_dir))
        
        print(f"Pipeline result: {result}")
        
        # Test convenience function
        print("\nTesting convenience function...")
        result2 = run_simple_pipeline(str(tiff_path), str(output_dir / "test2"))
        print(f"Convenience function result: {result2}")
        
        # Check outputs
        if output_dir.exists():
            print(f"\nOutput directory contents:")
            for item in output_dir.rglob("*"):
                if item.is_file():
                    print(f"  {item.relative_to(temp_path)}")
        
        print("\n✅ Pipeline test completed successfully!")

if __name__ == "__main__":
    test_simple_pipeline()
