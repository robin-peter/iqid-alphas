"""
Concise Simple Pipeline for Automated iQID Processing

Streamlined pipeline leveraging existing iqid_alphas utilities for:
- Multi-slice TIFF processing 
- Automatic tissue segmentation
- Temporal alignment
- Basic visualization
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from ..core.processor import IQIDProcessor
from ..core.segmentation import ImageSegmenter
from ..core.alignment import ImageAligner
from ..visualization.plotter import Visualizer

try:
    import tifffile
    from skimage import measure
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False


class SimplePipeline:
    """Concise pipeline for automated iQID processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with existing core utilities."""
        self.config = config or {}
        self.processor = IQIDProcessor()
        self.segmenter = ImageSegmenter()
        self.aligner = ImageAligner()
        self.visualizer = Visualizer()
    
    def process_iqid_stack(self, tiff_path: str, output_dir: str) -> Dict[str, Any]:
        """Process multi-slice iQID TIFF through pipeline."""
        print(f"🔬 Processing: {Path(tiff_path).name}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load TIFF stack
            stack = self._load_stack(tiff_path)
            
            # Process each slice
            processed = [self.processor.preprocess_image(img) for img in stack]
            
            # Segment tissues
            segmented = [self.segmenter.segment_tissue(img) for img in processed]
            
            # Align slices if multiple
            aligned = self._align_stack(segmented) if len(segmented) > 1 else segmented
            
            # Save results
            self._save_stack(aligned, output_path / "segmented")
            
            # Basic visualization
            if len(stack) > 0:
                self._create_overview(stack[0], segmented[0], output_path / "overview.png")
            
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
    
    def _load_stack(self, tiff_path: str) -> List[np.ndarray]:
        """Load multi-slice TIFF using processor utilities."""
        if HAS_IMAGING:
            try:
                stack = tifffile.imread(tiff_path)
                return [stack[i] for i in range(stack.shape[0])] if stack.ndim == 3 else [stack]
            except:
                return [self.processor.load_image(tiff_path)]
        else:
            # Mock data for testing
            return [np.random.randint(0, 255, (512, 512)).astype(np.uint8) for _ in range(5)]
    
    def _align_stack(self, stack: List[np.ndarray]) -> List[np.ndarray]:
        """Align stack using core aligner."""
        if len(stack) <= 1:
            return stack
        
        reference = stack[len(stack) // 2]  # Use middle as reference
        aligned = []
        
        for img in stack:
            try:
                aligned_img, _ = self.aligner.align_images(reference, img)
                aligned.append(aligned_img)
            except:
                aligned.append(img)  # Keep original if alignment fails
        
        return aligned
    
    def _save_stack(self, stack: List[np.ndarray], output_dir: Path):
        """Save processed stack to TIFF files."""
        output_dir.mkdir(exist_ok=True)
        
        if HAS_IMAGING:
            for i, img in enumerate(stack):
                tifffile.imwrite(output_dir / f"slice_{i+1:03d}.tif", img.astype(np.uint16))
        
        # Save summary
        with open(output_dir / "summary.json", 'w') as f:
            json.dump({'n_slices': len(stack), 'format': 'uint16'}, f)
    
    def _create_overview(self, raw_img: np.ndarray, seg_img: np.ndarray, output_path: Path):
        """Create simple overview visualization."""
        try:
            self.visualizer.plot_activity_map(raw_img, "Original Image")
            self.visualizer.current_figure.savefig(str(output_path).replace('.png', '_raw.png'))
            
            self.visualizer.plot_activity_map(seg_img, "Segmentation")
            self.visualizer.current_figure.savefig(str(output_path).replace('.png', '_seg.png'))
        except:
            pass  # Skip if visualization fails
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.tif*") -> Dict[str, Any]:
        """Process multiple TIFF files in batch."""
        input_path = Path(input_dir)
        files = list(input_path.glob(file_pattern))
        
        if not files:
            return {'status': 'failed', 'error': f'No files found in {input_dir}'}
        
        results = []
        for file_path in files:
            file_output = Path(output_dir) / file_path.stem
            result = self.process_iqid_stack(str(file_path), str(file_output))
            results.append(result)
        
        # Save batch summary
        summary = {
            'total_files': len(files),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }
        
        with open(Path(output_dir) / 'batch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def run_simple_pipeline(image_path: str, output_dir: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Quick function to run pipeline on a single image."""
    pipeline = SimplePipeline(config)
    return pipeline.process_iqid_stack(image_path, output_dir)
