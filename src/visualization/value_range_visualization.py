#!/usr/bin/env python3
"""
Value Range Visualization with Outlier Clipping

This module provides comprehensive visualization of image value ranges
with intelligent outlier detection and clipping for better display.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ValueRangeVisualizer:
    """
    Comprehensive value range visualization with outlier handling.
    """
    
    def __init__(self, figsize=(15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.outlier_percentiles = (1, 99)  # Default outlier clipping percentiles
        
    def analyze_value_ranges(self, images: Dict[str, np.ndarray], 
                           outlier_percentiles: Tuple[float, float] = (1, 99)):
        """
        Analyze value ranges for multiple images with outlier detection.
        
        Args:
            images: Dictionary of image_name -> image_array
            outlier_percentiles: Percentiles for outlier clipping (lower, upper)
            
        Returns:
            Dictionary with analysis results
        """
        self.outlier_percentiles = outlier_percentiles
        results = {}
        
        for name, image in images.items():
            # Flatten image for analysis
            flat_values = image.flatten()
            
            # Basic statistics
            stats = {
                'min': float(np.min(flat_values)),
                'max': float(np.max(flat_values)),
                'mean': float(np.mean(flat_values)),
                'median': float(np.median(flat_values)),
                'std': float(np.std(flat_values)),
                'count': len(flat_values)
            }
            
            # Percentile analysis for outlier detection
            percentiles = np.percentile(flat_values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            outlier_bounds = np.percentile(flat_values, outlier_percentiles)
            
            # Outlier analysis
            outliers_low = np.sum(flat_values < outlier_bounds[0])
            outliers_high = np.sum(flat_values > outlier_bounds[1])
            outlier_ratio = (outliers_low + outliers_high) / len(flat_values)
            
            # Clipped statistics
            clipped_values = flat_values[
                (flat_values >= outlier_bounds[0]) & 
                (flat_values <= outlier_bounds[1])
            ]
            
            clipped_stats = {
                'min': float(np.min(clipped_values)) if len(clipped_values) > 0 else stats['min'],
                'max': float(np.max(clipped_values)) if len(clipped_values) > 0 else stats['max'],
                'mean': float(np.mean(clipped_values)) if len(clipped_values) > 0 else stats['mean'],
                'std': float(np.std(clipped_values)) if len(clipped_values) > 0 else stats['std']
            }
            
            results[name] = {
                'raw_stats': stats,
                'clipped_stats': clipped_stats,
                'percentiles': {
                    'p01': percentiles[0], 'p05': percentiles[1], 'p10': percentiles[2],
                    'p25': percentiles[3], 'p50': percentiles[4], 'p75': percentiles[5],
                    'p90': percentiles[6], 'p95': percentiles[7], 'p99': percentiles[8]
                },
                'outliers': {
                    'low_count': int(outliers_low),
                    'high_count': int(outliers_high),
                    'total_ratio': float(outlier_ratio),
                    'bounds': outlier_bounds.tolist()
                },
                'flat_values': flat_values
            }
        
        return results
    
    def create_comprehensive_visualization(self, images: Dict[str, np.ndarray], 
                                         output_path: Optional[str] = None,
                                         outlier_percentiles: Tuple[float, float] = (0.5, 99.5),
                                         is_16bit: bool = True):
        """
        Create a comprehensive visualization showing all value ranges with outlier clipping.
        
        Args:
            images: Dictionary of image_name -> image_array
            output_path: Optional path to save the visualization
            outlier_percentiles: Percentiles for outlier clipping
            is_16bit: Whether to use 16-bit optimized analysis
        """
        if is_16bit:
            analysis = self.analyze_16bit_images(images, outlier_percentiles)
        else:
            analysis = self.analyze_value_ranges(images, outlier_percentiles)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
        
        # Color palette for different images
        colors = plt.cm.Set3(np.linspace(0, 1, len(images)))
        
        # 1. Raw vs Clipped Histograms
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_raw_vs_clipped_histograms(ax1, analysis, colors)
        
        # 2. Box plots comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_boxplot_comparison(ax2, analysis, colors)
        
        # 3. Percentile analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_percentile_analysis(ax3, analysis, colors)
        
        # 4. Outlier analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_outlier_analysis(ax4, analysis, colors)
        
        # 5. Value distribution heatmap
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_value_distribution_heatmap(ax5, analysis)
        
        # 6. Range comparison bar chart
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_range_comparison(ax6, analysis, colors)
        
        # 7. Summary statistics table
        ax7 = fig.add_subplot(gs[3, :])
        self._create_summary_table(ax7, analysis)
        
        # Add main title
        fig.suptitle(f'Comprehensive Value Range Analysis with Outlier Clipping\n'
                    f'Outlier percentiles: {self.outlier_percentiles[0]}% - {self.outlier_percentiles[1]}%',
                    fontsize=16, fontweight='bold')
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comprehensive visualization: {output_path}")
        
        return fig, analysis
    
    def _plot_raw_vs_clipped_histograms(self, ax, analysis, colors):
        """Plot raw vs clipped histograms for each image."""
        ax.set_title('Raw vs Clipped Value Distributions', fontweight='bold')
        
        for i, (name, data) in enumerate(analysis.items()):
            flat_values = data['flat_values']
            bounds = data['outliers']['bounds']
            
            # Plot raw histogram (transparent)
            ax.hist(flat_values, bins=100, alpha=0.3, color=colors[i], 
                   label=f'{name} (raw)', density=True)
            
            # Plot clipped histogram (solid)
            clipped_values = flat_values[
                (flat_values >= bounds[0]) & (flat_values <= bounds[1])
            ]
            ax.hist(clipped_values, bins=50, alpha=0.8, color=colors[i],
                   label=f'{name} (clipped)', density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel('Pixel Values')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_boxplot_comparison(self, ax, analysis, colors):
        """Plot box plots comparing raw and clipped statistics."""
        ax.set_title('Raw vs Clipped Statistics Comparison', fontweight='bold')
        
        image_names = list(analysis.keys())
        raw_data = [analysis[name]['flat_values'] for name in image_names]
        
        # Create clipped data
        clipped_data = []
        for name in image_names:
            flat_values = analysis[name]['flat_values']
            bounds = analysis[name]['outliers']['bounds']
            clipped = flat_values[(flat_values >= bounds[0]) & (flat_values <= bounds[1])]
            clipped_data.append(clipped)
        
        # Plot box plots
        positions_raw = np.arange(len(image_names)) * 2;
        positions_clipped = positions_raw + 0.8;
        
        bp1 = ax.boxplot(raw_data, positions=positions_raw, widths=0.6, 
                        patch_artist=True, showfliers=True)
        bp2 = ax.boxplot(clipped_data, positions=positions_clipped, widths=0.6,
                        patch_artist=True, showfliers=False)
        
        # Color the boxes
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # Customize
        ax.set_xticks(positions_raw + 0.4)
        ax.set_xticklabels([f'{name}\n(raw/clipped)' for name in image_names])
        ax.set_ylabel('Pixel Values')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.plot([], [], 's', color='gray', alpha=0.5, label='Raw (with outliers)')
        ax.plot([], [], 's', color='gray', alpha=0.8, label='Clipped (outliers removed)')
        ax.legend()
    
    def _plot_percentile_analysis(self, ax, analysis, colors):
        """Plot percentile analysis showing the distribution shape."""
        ax.set_title('Percentile Distribution Analysis', fontweight='bold')
        
        percentile_labels = ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%']
        percentile_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        
        for i, (name, data) in enumerate(analysis.items()):
            p_data = data['percentiles']
            values = [p_data[f'p{p:02d}'] for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
            
            ax.plot(percentile_values, values, 'o-', color=colors[i], 
                   label=name, linewidth=2, markersize=6)
            
            # Highlight outlier boundaries
            outlier_bounds = data['outliers']['bounds']
            ax.axhline(outlier_bounds[0], color=colors[i], linestyle='--', alpha=0.5)
            ax.axhline(outlier_bounds[1], color=colors[i], linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Pixel Value')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add percentile labels
        ax.set_xticks(percentile_values)
        ax.set_xticklabels(percentile_labels)
    
    def _plot_outlier_analysis(self, ax, analysis, colors):
        """Plot outlier analysis showing counts and ratios."""
        ax.set_title('Outlier Analysis', fontweight='bold')
        
        names = list(analysis.keys())
        low_counts = [analysis[name]['outliers']['low_count'] for name in names]
        high_counts = [analysis[name]['outliers']['high_count'] for name in names]
        total_pixels = [analysis[name]['raw_stats']['count'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        # Plot stacked bars
        bars1 = ax.bar(x - width/2, low_counts, width, label='Low outliers', 
                      color='red', alpha=0.7)
        bars2 = ax.bar(x - width/2, high_counts, width, bottom=low_counts,
                      label='High outliers', color='orange', alpha=0.7)
        
        # Plot total pixels (reference)
        ax2 = ax.twinx()
        bars3 = ax2.bar(x + width/2, total_pixels, width, label='Total pixels',
                       color='lightblue', alpha=0.5)
        
        # Annotations
        for i, (name, data) in enumerate(analysis.items()):
            ratio = data['outliers']['total_ratio'] * 100
            ax.text(i, low_counts[i] + high_counts[i] + max(high_counts) * 0.05,
                   f'{ratio:.1f}%', ha='center', fontweight='bold')
        
        ax.set_xlabel('Images')
        ax.set_ylabel('Outlier Count')
        ax2.set_ylabel('Total Pixels')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_value_distribution_heatmap(self, ax, analysis):
        """Plot a heatmap showing value distribution across images."""
        ax.set_title('Value Distribution Heatmap (Clipped)', fontweight='bold')
        
        # Create bins for the heatmap
        all_values = []
        for data in analysis.values():
            bounds = data['outliers']['bounds']
            flat_values = data['flat_values']
            clipped = flat_values[(flat_values >= bounds[0]) & (flat_values <= bounds[1])]
            all_values.extend(clipped)
        
        if not all_values:
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
            return
        
        global_min, global_max = np.min(all_values), np.max(all_values)
        bins = np.linspace(global_min, global_max, 50)
        
        # Create heatmap data
        heatmap_data = []
        image_names = []
        
        for name, data in analysis.items():
            bounds = data['outliers']['bounds']
            flat_values = data['flat_values']
            clipped = flat_values[(flat_values >= bounds[0]) & (flat_values <= bounds[1])]
            
            if len(clipped) > 0:
                hist, _ = np.histogram(clipped, bins=bins, density=True)
                heatmap_data.append(hist)
                image_names.append(name)
        
        if heatmap_data:
            heatmap_data = np.array(heatmap_data)
            
            # Plot heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                          extent=[global_min, global_max, 0, len(image_names)])
            
            # Colorbar
            plt.colorbar(im, ax=ax, label='Density')
            
            # Labels
            ax.set_xlabel('Pixel Value (Clipped)')
            ax.set_ylabel('Images')
            ax.set_yticks(np.arange(len(image_names)) + 0.5)
            ax.set_yticklabels(image_names)
    
    def _plot_range_comparison(self, ax, analysis, colors):
        """Plot range comparison showing raw vs clipped ranges."""
        ax.set_title('Value Range Comparison: Raw vs Clipped', fontweight='bold')
        
        names = list(analysis.keys())
        raw_mins = [analysis[name]['raw_stats']['min'] for name in names]
        raw_maxs = [analysis[name]['raw_stats']['max'] for name in names]
        clipped_mins = [analysis[name]['clipped_stats']['min'] for name in names]
        clipped_maxs = [analysis[name]['clipped_stats']['max'] for name in names]
        
        y_pos = np.arange(len(names))
        
        # Plot ranges as horizontal bars
        for i, name in enumerate(names):
            # Raw range (background)
            ax.barh(y_pos[i] - 0.2, raw_maxs[i] - raw_mins[i], left=raw_mins[i],
                   height=0.4, color=colors[i], alpha=0.3, label='Raw' if i == 0 else "")
            
            # Clipped range (foreground)
            ax.barh(y_pos[i] + 0.2, clipped_maxs[i] - clipped_mins[i], left=clipped_mins[i],
                   height=0.4, color=colors[i], alpha=0.8, label='Clipped' if i == 0 else "")
            
            # Add range annotations
            raw_range = raw_maxs[i] - raw_mins[i]
            clipped_range = clipped_maxs[i] - clipped_mins[i]
            reduction = (1 - clipped_range/raw_range) * 100 if raw_range > 0 else 0
            
            ax.text(raw_maxs[i] + (max(raw_maxs) - min(raw_mins)) * 0.02, y_pos[i],
                   f'{reduction:.1f}% reduced', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Pixel Value Range')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_summary_table(self, ax, analysis):
        """Create a summary statistics table."""
        ax.set_title('Summary Statistics Table', fontweight='bold')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Image', 'Raw Min', 'Raw Max', 'Raw Mean±STD', 
                  'Clipped Min', 'Clipped Max', 'Clipped Mean±STD', 
                  'Outliers (%)', 'Range Reduction (%)']
        
        table_data = []
        for name, data in analysis.items():
            raw = data['raw_stats']
            clipped = data['clipped_stats']
            outlier_ratio = data['outliers']['total_ratio'] * 100
            
            raw_range = raw['max'] - raw['min']
            clipped_range = clipped['max'] - clipped['min']
            range_reduction = (1 - clipped_range/raw_range) * 100 if raw_range > 0 else 0
            
            row = [
                name,
                f"{raw['min']:.1f}",
                f"{raw['max']:.1f}",
                f"{raw['mean']:.1f}±{raw['std']:.1f}",
                f"{clipped['min']:.1f}",
                f"{clipped['max']:.1f}",
                f"{clipped['mean']:.1f}±{clipped['std']:.1f}",
                f"{outlier_ratio:.1f}%",
                f"{range_reduction:.1f}%"
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternate rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')


    def analyze_16bit_images(self, images: Dict[str, np.ndarray], 
                           outlier_percentiles: Tuple[float, float] = (0.5, 99.5)):
        """
        Analyze value ranges specifically for 16-bit images (like iQID data).
        Uses more conservative outlier clipping to preserve dynamic range.
        
        Args:
            images: Dictionary of image_name -> image_array
            outlier_percentiles: Percentiles for outlier clipping (default 0.5, 99.5 for 16-bit)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"📊 Analyzing 16-bit images with {outlier_percentiles[0]}-{outlier_percentiles[1]} percentile clipping...")
        return self.analyze_value_ranges(images, outlier_percentiles)
    

def create_sample_visualization(output_dir: str = "./outputs/visualization"):
    """
    Create a sample visualization with synthetic data demonstrating outlier clipping.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate sample images with different characteristics
    np.random.seed(42)
    
    # Image 1: Normal distribution with some outliers
    img1_base = np.random.normal(100, 20, (200, 200))
    img1_outliers_high = np.random.exponential(500, (200, 200)) * (np.random.random((200, 200)) < 0.02)
    img1_outliers_low = -np.random.exponential(50, (200, 200)) * (np.random.random((200, 200)) < 0.01)
    img1 = img1_base + img1_outliers_high + img1_outliers_low
    img1 = np.clip(img1, 0, None)  # Ensure non-negative
    
    # Image 2: Log-normal distribution (typical for biological images)
    img2 = np.random.lognormal(4, 1, (200, 200))
    # Add some extreme outliers
    outlier_mask = np.random.random((200, 200)) < 0.005
    img2[outlier_mask] = np.random.uniform(5000, 10000, np.sum(outlier_mask))
    
    # Image 3: Bimodal distribution
    img3_mode1 = np.random.normal(50, 10, (200, 200)) * (np.random.random((200, 200)) < 0.6)
    img3_mode2 = np.random.normal(200, 30, (200, 200)) * (np.random.random((200, 200)) >= 0.6)
    img3 = img3_mode1 + img3_mode2
    # Add scattered high outliers
    outlier_mask = np.random.random((200, 200)) < 0.01
    img3[outlier_mask] = np.random.uniform(1000, 2000, np.sum(outlier_mask))
    
    # Create sample images dictionary
    sample_images = {
        'H&E Tissue': img1.astype(np.float32),
        'iQID Activity': img2.astype(np.float32),
        'Combined Signal': img3.astype(np.float32)
    }
    
    # Create visualizer and generate plots
    visualizer = ValueRangeVisualizer(figsize=(20, 16))
    
    # Create comprehensive visualization
    fig, analysis = visualizer.create_comprehensive_visualization(
        sample_images, 
        output_path=f"{output_dir}/comprehensive_value_range_analysis.png"
    )
    
    # Also create individual image visualizations
    for name, image in sample_images.items():
        fig_individual = plt.figure(figsize=(15, 10))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(image, cmap='viridis')
        ax1.set_title(f'{name} - Original')
        plt.colorbar(im1, ax=ax1)
        
        # Clipped image
        ax2 = plt.subplot(2, 3, 2)
        bounds = analysis[name]['outliers']['bounds']
        clipped_image = np.clip(image, bounds[0], bounds[1])
        im2 = ax2.imshow(clipped_image, cmap='viridis')
        ax2.set_title(f'{name} - Clipped ({bounds[0]:.1f}-{bounds[1]:.1f})')
        plt.colorbar(im2, ax=ax2)
        
        # Outlier mask
        ax3 = plt.subplot(2, 3, 3)
        outlier_mask = (image < bounds[0]) | (image > bounds[1])
        ax3.imshow(outlier_mask, cmap='Reds')
        ax3.set_title(f'{name} - Outliers ({np.sum(outlier_mask)} pixels)')
        
        # Histogram comparison
        ax4 = plt.subplot(2, 3, (4, 6))
        flat_vals = image.flatten()
        ax4.hist(flat_vals, bins=100, alpha=0.5, label='Original', density=True)
        clipped_vals = flat_vals[(flat_vals >= bounds[0]) & (flat_vals <= bounds[1])]
        ax4.hist(clipped_vals, bins=50, alpha=0.7, label='Clipped', density=True)
        ax4.axvline(bounds[0], color='red', linestyle='--', label='Lower bound')
        ax4.axvline(bounds[1], color='red', linestyle='--', label='Upper bound')
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        individual_path = f"{output_dir}/{name.replace('&', 'and').replace(' ', '_').lower()}_analysis.png"
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved individual analysis: {individual_path}")
    
    plt.show()
    
    return analysis


def visualize_pipeline_data(data_directory: str, output_dir: str = "./outputs/visualization",
                           file_patterns: Dict[str, str] = None):
    """
    Visualize value ranges from real pipeline data.
    
    Args:
        data_directory: Directory containing image files
        output_dir: Output directory for visualizations
        file_patterns: Dictionary of data_type -> file_pattern (e.g., {'HE': '*.tif', 'iQID': '*_iqid.tif'})
    """
    from tifffile import imread
    import glob
    
    if file_patterns is None:
        file_patterns = {
            'HE': '*he*.tif',
            'iQID': '*iqid*.tif', 
            'Raw': '*raw*.tif',
            'Segmented': '*seg*.tif',
            'Aligned': '*align*.tif'
        }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load images based on patterns
    images = {}
    data_path = Path(data_directory)
    
    for data_type, pattern in file_patterns.items():
        matching_files = list(data_path.glob(pattern))
        if matching_files:
            # Load first matching file as sample
            try:
                image = imread(str(matching_files[0]))
                # Handle multi-channel images by taking first channel or converting to grayscale
                if len(image.shape) > 2:
                    if image.shape[-1] == 3:  # RGB
                        image = np.mean(image, axis=-1)  # Convert to grayscale
                    else:
                        image = image[..., 0]  # Take first channel
                
                images[f'{data_type} ({matching_files[0].name})'] = image.astype(np.float32)
                print(f"✓ Loaded {data_type}: {matching_files[0].name} - Shape: {image.shape}")
            except Exception as e:
                print(f"⚠ Failed to load {data_type} from {matching_files[0]}: {e}")
    
    if not images:
        print("⚠ No images found matching the patterns")
        return None
    
    # Create visualizer and generate analysis
    visualizer = ValueRangeVisualizer(figsize=(20, 16))
    
    # Create comprehensive visualization
    output_path = f"{output_dir}/pipeline_data_value_analysis.png"
    fig, analysis = visualizer.create_comprehensive_visualization(images, output_path)
    
    # Create individual visualizations for each image type
    for name, image in images.items():
        safe_name = name.replace('(', '').replace(')', '').replace(' ', '_').replace('.', '_').lower()
        individual_path = f"{output_dir}/individual_{safe_name}_analysis.png"
        
        fig_individual = plt.figure(figsize=(15, 10))
        
        # Original image with outliers highlighted
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(image, cmap='viridis')
        ax1.set_title(f'{name} - Original')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Clipped image
        ax2 = plt.subplot(2, 3, 2)
        bounds = analysis[name]['outliers']['bounds']
        clipped_image = np.clip(image, bounds[0], bounds[1])
        im2 = ax2.imshow(clipped_image, cmap='viridis')
        ax2.set_title(f'Clipped (1%-99%)\n[{bounds[0]:.1f}, {bounds[1]:.1f}]')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Outlier locations
        ax3 = plt.subplot(2, 3, 3)
        outlier_mask = (image < bounds[0]) | (image > bounds[1])
        outlier_display = np.zeros_like(image)
        outlier_display[image < bounds[0]] = -1  # Low outliers in blue
        outlier_display[image > bounds[1]] = 1   # High outliers in red
        im3 = ax3.imshow(outlier_display, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_title(f'Outlier Map\n{np.sum(outlier_mask)} pixels ({np.sum(outlier_mask)/image.size*100:.1f}%)')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.set_ticks([-1, 0, 1])
        cbar3.set_ticklabels(['Low', 'Normal', 'High'])
        
        # Value distribution histogram
        ax4 = plt.subplot(2, 3, (4, 6))
        flat_vals = image.flatten()
        
        # Plot full histogram
        ax4.hist(flat_vals, bins=100, alpha=0.5, label='All values', density=True, color='lightblue')
        
        # Plot clipped histogram
        clipped_vals = flat_vals[(flat_vals >= bounds[0]) & (flat_vals <= bounds[1])]
        ax4.hist(clipped_vals, bins=50, alpha=0.8, label='Clipped (1%-99%)', density=True, color='darkblue')
        
        # Mark percentile boundaries
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        p_values = np.percentile(flat_vals, percentiles)
        colors_p = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']
        
        for p, val, color in zip(percentiles, p_values, colors_p):
            ax4.axvline(val, color=color, linestyle='--', alpha=0.7, linewidth=1)
            if p in [1, 99]:  # Only label the clipping boundaries
                ax4.axvline(val, color='red', linestyle='-', linewidth=2, 
                           label=f'{p}% bound ({val:.1f})')
        
        ax4.set_xlabel('Pixel Value')
        ax4.set_ylabel('Density')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Value Distribution with Percentile Boundaries')
        
        plt.tight_layout()
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved individual analysis: {individual_path}")
    
    return analysis


def analyze_iqid_reupload_data(reupload_path: str = "../data/UCSF-Collab/data/ReUpload/iQID_reupload/iQID/Sequential/kidneys/D1M1_L"):
    """
    Analyze value ranges in the iQID ReUpload data specifically.
    """
    print(f"🔍 Analyzing iQID ReUpload data from: {reupload_path}")
    
    # Check if path exists
    data_path = Path(reupload_path)
    if not data_path.exists():
        print(f"⚠ Path does not exist: {reupload_path}")
        return None
    
    # Define specific patterns for ReUpload data
    file_patterns = {
        'Raw': 'raw_events*.tif',
        'Segmented': '1_segmented/*seg*.tif',
        'Aligned': '2_aligned/*align*.tif'
    }
    
    return visualize_pipeline_data(str(data_path), "./outputs/reupload_analysis", file_patterns)


def analyze_datapush_data(datapush_path: str = "../data/UCSF-Collab/data/DataPush1"):
    """
    Analyze value ranges in the DataPush1 data.
    """
    print(f"🔍 Analyzing DataPush1 data from: {datapush_path}")
    
    # Check if path exists
    data_path = Path(datapush_path)
    if not data_path.exists():
        print(f"⚠ Path does not exist: {datapush_path}")
        return None
    
    # Look for H&E and iQID subdirectories
    he_path = data_path / "HE"
    iqid_path = data_path / "iQID"
    
    all_images = {}
    
    # Load H&E images
    if he_path.exists():
        he_files = list(he_path.rglob("*.tif")) + list(he_path.rglob("*.tiff"))
        if he_files:
            try:
                he_image = imread(str(he_files[0]))
                if len(he_image.shape) > 2:
                    he_image = np.mean(he_image, axis=-1)  # Convert to grayscale if needed
                all_images[f'H&E ({he_files[0].name})'] = he_image.astype(np.float32)
                print(f"✓ Loaded H&E: {he_files[0].name}")
            except Exception as e:
                print(f"⚠ Failed to load H&E image: {e}")
    
    # Load iQID images
    if iqid_path.exists():
        iqid_files = list(iqid_path.rglob("*.tif")) + list(iqid_path.rglob("*.tiff"))
        for i, iqid_file in enumerate(iqid_files[:3]):  # Limit to first 3 files
            try:
                iqid_image = imread(str(iqid_file))
                if len(iqid_image.shape) > 2:
                    iqid_image = iqid_image[..., 0] if iqid_image.shape[-1] > 1 else iqid_image.squeeze()
                all_images[f'iQID_{i+1} ({iqid_file.name})'] = iqid_image.astype(np.float32)
                print(f"✓ Loaded iQID: {iqid_file.name}")
            except Exception as e:
                print(f"⚠ Failed to load iQID image {iqid_file}: {e}")
    
    if not all_images:
        print("⚠ No images found in DataPush1 directory")
        return None
    
    # Create visualizer and analyze
    visualizer = ValueRangeVisualizer(figsize=(20, 16))
    output_path = "./outputs/datapush_analysis/comprehensive_datapush_analysis.png"
    Path("./outputs/datapush_analysis").mkdir(parents=True, exist_ok=True)
    
    fig, analysis = visualizer.create_comprehensive_visualization(all_images, output_path)
    
    return analysis


def analyze_iqid_data_optimized(data_directory: str, output_dir: str = "./outputs/visualization",
                              outlier_percentiles: Tuple[float, float] = (0.1, 99.9)):
    """
    Analyze iQID data with optimized settings for 16-bit images.
    Uses very conservative outlier clipping to preserve the full dynamic range.
    
    Args:
        data_directory: Path to directory containing iQID images
        output_dir: Output directory for visualizations
        outlier_percentiles: Conservative percentiles for 16-bit data (default 0.1, 99.9)
    """
    print(f"🔬 Analyzing iQID data with optimized 16-bit settings...")
    print(f"   📊 Using {outlier_percentiles[0]}-{outlier_percentiles[1]} percentile clipping")
    
    # Define file patterns for different types of iQID data
    file_patterns = {
        'Raw Event': '*raw*.tif',
        'Segmented': '*seg*.tif',
        'Aligned': '*align*.tif',
        'Event Data': '*event*.tif',
        'Frame Data': '*frame*.tif'
    }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load images based on patterns
    images = {}
    data_path = Path(data_directory)
    
    for data_type, pattern in file_patterns.items():
        matching_files = list(data_path.glob(pattern))
        if matching_files:
            # Load first matching file as sample
            try:
                image = imread(str(matching_files[0]))
                # Handle multi-channel images by taking first channel
                if len(image.shape) > 2:
                    image = image[:, :, 0] if image.shape[-1] <= image.shape[0] else image[0]
                
                images[f"{data_type} ({image.dtype}, range: {image.min()}-{image.max()})"] = image
                print(f"   ✓ Loaded {data_type}: {image.shape}, {image.dtype}, range: {image.min()}-{image.max()}")
            except Exception as e:
                print(f"   ⚠ Could not load {data_type}: {e}")
    
    if not images:
        print("⚠ No images found matching the patterns")
        return None
    
    # Create visualizer optimized for 16-bit data
    visualizer = ValueRangeVisualizer(figsize=(20, 16))
    
    # Create comprehensive visualization with 16-bit optimization
    output_path = f"{output_dir}/iqid_optimized_value_analysis.png"
    fig, analysis = visualizer.create_comprehensive_visualization(
        images, output_path, outlier_percentiles=outlier_percentiles, is_16bit=True
    )
    
    # Print detailed statistics for 16-bit analysis
    print(f"\n📈 16-bit iQID Data Analysis Summary:")
    for name, stats in analysis.items():
        print(f"   🔬 {name}:")
        print(f"      📊 Full Range: {stats['original_min']:.1f} - {stats['original_max']:.1f}")
        print(f"      ✂️  Clipped Range: {stats['clipped_min']:.1f} - {stats['clipped_max']:.1f}")
        print(f"      📈 Dynamic Range: {stats['original_max'] - stats['original_min']:.1f}")
        print(f"      🎯 Outliers: {stats['outlier_count']:.0f} ({stats['outlier_percentage']:.2f}%)")
    
    print(f"\n✅ Analysis saved to: {output_path}")
    return fig, analysis

# ...existing code...
