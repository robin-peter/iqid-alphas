"""
Core module for IQID-Alphas package.
"""

from .processor import IQIDProcessor, quick_process, batch_process

__all__ = ['IQIDProcessor', 'quick_process', 'batch_process']
