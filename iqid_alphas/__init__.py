"""
IQID-Alphas: Quantitative Imaging Analysis Framework

A comprehensive Python package for processing iQID camera digital autoradiographs
and H&E histology images for quantitative analysis and dosimetry.

Authors: Robin Peter, Brian Miller
"""

__version__ = "1.0.0"
__author__ = "Robin Peter, Brian Miller"
__email__ = "contact@iqid-alphas.org"

from .core.processor import IQIDProcessor
from .core.alignment import ImageAligner
from .core.segmentation import ImageSegmenter
from .pipelines.simple import SimplePipeline
from .pipelines.advanced import AdvancedPipeline
from .pipelines.combined import CombinedPipeline
from .visualization.plotter import Visualizer

__all__ = [
    'IQIDProcessor',
    'ImageAligner', 
    'ImageSegmenter',
    'SimplePipeline',
    'AdvancedPipeline', 
    'CombinedPipeline',
    'Visualizer'
]
