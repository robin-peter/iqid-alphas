"""
IQID-Alphas: Quantitative Imaging Analysis Framework

A comprehensive Python package for processing iQID camera digital autoradiographs
and H&E histology images for quantitative analysis and dosimetry.

Authors: Robin Peter, Brian Miller
"""

__version__ = "1.0.0"
__author__ = "Robin Peter, Brian Miller"


from .core.processor import IQIDProcessor
from .core.alignment import ImageAligner
from .core.segmentation import ImageSegmenter
from .pipelines.simple import SimplePipeline
from .pipelines.advanced import AdvancedPipeline
from .pipelines.combined import CombinedPipeline
from .visualization.plotter import Visualizer

# CLI is available as a module but not imported by default
# Use: python -m iqid_alphas.cli

__all__ = [
    'IQIDProcessor',
    'ImageAligner', 
    'ImageSegmenter',
    'SimplePipeline',
    'AdvancedPipeline', 
    'CombinedPipeline',
    'Visualizer'
]
