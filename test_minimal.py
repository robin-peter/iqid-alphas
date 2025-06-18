"""
Minimal Simple Pipeline Test - No external dependencies
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

class MockProcessor:
    def preprocess_image(self, image):
        return image * 0.8  # Simple preprocessing

class MockSegmenter:
    def segment_tissue(self, image):
        # Create a simple binary mask
        return (image > np.mean(image)).astype(np.uint8)

class MockAligner:
    def align_images(self, ref, moving):
        # Return the moving image as-is with mock info
        return moving, {'correlation': 0.8}

class MockVisualizer:
    def plot_activity_map(self, *args, **kwargs):
        pass
    current_figure = None

class MinimalPipeline:
    """Minimal pipeline for testing."""
    
    def __init__(self):
        self.processor = MockProcessor()
        self.segmenter = MockSegmenter()
        self.aligner = MockAligner()
        self.visualizer = MockVisualizer()
    
    def process_iqid_stack(self, tiff_path: str, output_dir: str) -> Dict[str, Any]:
        """Process mock iQID stack."""
        print(f"🔬 Processing: {Path(tiff_path).name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create mock stack
            stack = [np.random.randint(0, 255, (256, 256)) for _ in range(3)]
            
            # Process each slice
            processed = [self.processor.preprocess_image(img) for img in stack]
            
            # Segment tissues
            segmented = [self.segmenter.segment_tissue(img) for img in processed]
            
            # Align slices
            aligned = self._align_stack(segmented)
            
            # Save results
            self._save_stack(aligned, output_path / "segmented")
            
            result = {
                'status': 'success',
                'n_slices': len(stack),
                'output_dir': str(output_path)
            }
            print(f"✅ Processed {len(stack)} slices")
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _align_stack(self, stack: List[np.ndarray]) -> List[np.ndarray]:
        """Mock alignment."""
        if len(stack) <= 1:
            return stack
        
        reference = stack[0]
        aligned = []
        
        for img in stack:
            try:
                aligned_img, _ = self.aligner.align_images(reference, img)
                aligned.append(aligned_img)
            except:
                aligned.append(img)
        
        return aligned
    
    def _save_stack(self, stack: List[np.ndarray], output_dir: Path):
        """Save mock stack."""
        output_dir.mkdir(exist_ok=True)
        
        # Save summary
        with open(output_dir / "summary.json", 'w') as f:
            json.dump({'n_slices': len(stack), 'format': 'mock'}, f)

def test_minimal_pipeline():
    """Test the minimal pipeline."""
    print("Testing Minimal Pipeline...")
    
    pipeline = MinimalPipeline()
    result = pipeline.process_iqid_stack("test.tif", "/tmp/test_output")
    
    print(f"Result: {result}")
    print("✅ Minimal pipeline test completed!")

if __name__ == "__main__":
    test_minimal_pipeline()
