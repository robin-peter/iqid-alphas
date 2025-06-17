"""
Visualization module for IQID-Alphas

Provides publication-quality plotting and visualization capabilities
for iQID data analysis and dose distribution visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from typing import Optional, Union, Tuple


class Visualizer:
    """
    Main visualization class for IQID-Alphas analysis.
    
    Provides methods for creating publication-quality plots of:
    - Activity maps
    - Dose distributions  
    - Segmentation results
    - Analysis overlays
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use ('default', 'seaborn', etc.)
            figsize: Default figure size as (width, height)
        """
        self.figsize = figsize
        self.current_figure = None
        self.current_axes = None
        
        # Set plotting style
        if style != 'default':
            try:
                plt.style.use(style)
            except OSError:
                print(f"Warning: Style '{style}' not available, using default")
    
    def plot_activity_map(self, data: np.ndarray, title: str = "Activity Distribution",
                         colormap: str = 'viridis', vmin: Optional[float] = None,
                         vmax: Optional[float] = None, show_colorbar: bool = True):
        """
        Plot an activity distribution map.
        
        Args:
            data: 2D numpy array of activity data
            title: Plot title
            colormap: Matplotlib colormap name
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            show_colorbar: Whether to show colorbar
        """
        if data is None or data.size == 0:
            raise ValueError("Data cannot be None or empty")
        
        if data.ndim != 2:
            raise ValueError("Data must be 2D array")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Handle NaN values
        data_clean = np.nan_to_num(data)
        
        # Set color limits if not provided
        if vmin is None:
            vmin = np.percentile(data_clean[data_clean > 0], 1) if np.any(data_clean > 0) else 0
        if vmax is None:
            vmax = np.percentile(data_clean, 99)
        
        # Create the plot
        im = ax.imshow(data_clean, cmap=colormap, vmin=vmin, vmax=vmax,
                      aspect='equal', origin='lower')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Activity (counts)', fontsize=12)
        
        plt.tight_layout()
        
        self.current_figure = fig
        self.current_axes = ax
        
        return fig, ax
    
    def plot_dose_distribution(self, dose_data: np.ndarray, title: str = "Dose Distribution",
                              colormap: str = 'plasma', units: str = 'Gy',
                              show_colorbar: bool = True):
        """
        Plot a dose distribution map.
        
        Args:
            dose_data: 2D numpy array of dose data
            title: Plot title
            colormap: Matplotlib colormap name
            units: Units for the colorbar label
            show_colorbar: Whether to show colorbar
        """
        if dose_data is None or dose_data.size == 0:
            raise ValueError("Dose data cannot be None or empty")
        
        if dose_data.ndim != 2:
            raise ValueError("Dose data must be 2D array")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Handle NaN values and ensure positive values
        dose_clean = np.nan_to_num(dose_data)
        dose_clean = np.maximum(dose_clean, 0)
        
        # Use log scale for dose if dynamic range is large
        if np.max(dose_clean) / np.min(dose_clean[dose_clean > 0]) > 100:
            # Log scale visualization
            dose_clean = np.log10(dose_clean + 1e-10)
            label_suffix = f'log({units})'
        else:
            label_suffix = units
        
        im = ax.imshow(dose_clean, cmap=colormap, aspect='equal', origin='lower')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f'Dose ({label_suffix})', fontsize=12)
        
        plt.tight_layout()
        
        self.current_figure = fig
        self.current_axes = ax
        
        return fig, ax
    
    def plot_segmentation_overlay(self, image: np.ndarray, segments: np.ndarray,
                                 alpha: float = 0.5, title: str = "Segmentation Overlay"):
        """
        Plot segmentation results overlaid on original image.
        
        Args:
            image: Original 2D image
            segments: 2D segmentation mask
            alpha: Transparency of overlay
            title: Plot title
        """
        if image is None or segments is None:
            raise ValueError("Image and segments cannot be None")
        
        if image.shape != segments.shape:
            raise ValueError("Image and segments must have the same shape")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Show original image in grayscale
        ax.imshow(image, cmap='gray', alpha=1.0, aspect='equal', origin='lower')
        
        # Create colored overlay for segments
        masked_segments = np.ma.masked_where(segments == 0, segments)
        ax.imshow(masked_segments, cmap='jet', alpha=alpha, aspect='equal', origin='lower')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        
        plt.tight_layout()
        
        self.current_figure = fig
        self.current_axes = ax
        
        return fig, ax
    
    def plot_histogram(self, data: np.ndarray, bins: int = 50, title: str = "Data Histogram",
                      xlabel: str = "Value", ylabel: str = "Frequency", log_scale: bool = False):
        """
        Plot histogram of data values.
        
        Args:
            data: 1D or 2D numpy array
            bins: Number of histogram bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
        """
        if data is None or data.size == 0:
            raise ValueError("Data cannot be None or empty")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Flatten data if 2D
        data_flat = data.flatten() if data.ndim > 1 else data
        
        # Remove NaN values
        data_clean = data_flat[~np.isnan(data_flat)]
        
        ax.hist(data_clean, bins=bins, alpha=0.7, edgecolor='black')
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self.current_figure = fig
        self.current_axes = ax
        
        return fig, ax
    
    def save_figure(self, filename: str, dpi: int = 300, format: str = 'png',
                   bbox_inches: str = 'tight', transparent: bool = False):
        """
        Save the current figure to file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            format: File format ('png', 'pdf', 'svg', etc.)
            bbox_inches: Bounding box ('tight' or None)
            transparent: Whether to save with transparent background
        """
        if self.current_figure is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.current_figure.savefig(filename, dpi=dpi, format=format,
                                   bbox_inches=bbox_inches, transparent=transparent)
        print(f"Figure saved to: {filename}")
    
    def show(self):
        """Display the current figure."""
        if self.current_figure is None:
            raise ValueError("No figure to show. Create a plot first.")
        
        plt.show()
    
    def close(self):
        """Close the current figure."""
        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None
            self.current_axes = None
    
    def create_subplot_grid(self, rows: int, cols: int, figsize: Optional[Tuple[int, int]] = None):
        """
        Create a subplot grid for multiple plots.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size, uses default if None
            
        Returns:
            fig, axes: Figure and axes objects
        """
        if figsize is None:
            figsize = (self.figsize[0] * cols, self.figsize[1] * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        self.current_figure = fig
        self.current_axes = axes
        
        return fig, axes
