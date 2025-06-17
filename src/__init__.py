"""
IQID-Alphas source code package.

This package contains the core functionality for the IQID-Alphas project:
- core: Core IQID functionality (align, dpk, helper, process_object, spec)
- alignment: H&E and iQID image alignment modules
- processing: Enhanced processing and batch processing tools
- segmentation: Activity and tissue segmentation modules
- utils: Utility functions and analysis tools
- visualization: Visualization and plotting tools
"""

__version__ = "1.0.0"
__author__ = "IQID-Alphas Team"

# Core imports
try:
    from .core.iqid import align, dpk, helper, process_object, spec
except ImportError:
    pass

__all__ = [
    'align', 'dpk', 'helper', 'process_object', 'spec'
]
