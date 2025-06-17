"""
Visualization module initialization file.

This package contains various visualization utilities for the IQID-Alphas project:
- Alignment visualization tools
- ROI segmentation displays  
- Dose kernel visualizations
- Cross-reference visualization tools
"""

__version__ = "1.0.0"
__author__ = "IQID-Alphas Team"

# Import main visualization functions
try:
    from .generate_enhanced_visualizations import create_enhanced_visualization
    from .create_meaningful_alignment_demo import create_alignment_demo
    from .create_true_alignment_visualization import create_true_alignment_viz
except ImportError:
    # Allow module to be imported even if specific visualization tools aren't available
    pass

__all__ = [
    'create_enhanced_visualization',
    'create_alignment_demo', 
    'create_true_alignment_viz'
]
