"""
Data processing utilities for iQID analysis.
"""

import os
import logging
import numpy as np
from ..core.iqid.process_object import ClusterData

def load_and_process_listmode_data(file_name, c_area_thresh=15, makedir=False, ftype='processed_lm'):
    """
    Load and process listmode data.
    
    Parameters:
    -----------
    file_name : str
        Path to the listmode data file
    c_area_thresh : int
        Cluster area threshold (default: 15)
    makedir : bool
        Whether to create directory (default: False)
    ftype : str
        File type identifier (default: 'processed_lm')
        
    Returns:
    --------
    tuple
        (cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num)
    """
    try:
        cluster_data = ClusterData(file_name, c_area_thresh, makedir, ftype)
        cluster_data.init_header()
        data = cluster_data.load_cluster_data()
        time_ms, cluster_area, xC_global, yC_global, frame_num = cluster_data.init_metadata(data)
        return cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num
    except Exception as e:
        logging.error("Failed to load and process listmode data: %s", str(e))
        raise

def filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half):
    """
    Filter, correct, and analyze cluster data.
    
    Parameters:
    -----------
    cluster_data : ClusterData
        The cluster data object
    binfac : int
        Binning factor
    ROI_area_thresh : float
        ROI area threshold
    t_binsize : float
        Time bin size
    t_half : float
        Half-life for decay correction
    """
    try:
        cluster_data.set_process_params(binfac, ROI_area_thresh, t_binsize, t_half)
        cluster_data.get_mean_n()
        cluster_data.estimate_missed_timestamps()
        return cluster_data
    except Exception as e:
        logging.error("Failed to filter, correct, and analyze data: %s", str(e))
        raise

def generate_spatial_images(cluster_data, subpx=1):
    """Generate spatial images from cluster data."""
    try:
        cluster_image = cluster_data.image_from_listmode(subpx)
        return cluster_image
    except Exception as e:
        logging.error("Failed to generate spatial images: %s", str(e))
        raise

def generate_temporal_information(cluster_data, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
    """Generate temporal information from cluster data."""
    try:
        cluster_image = cluster_data.image_from_big_listmode(event_fx, xlim, ylim)
        return cluster_image
    except Exception as e:
        logging.error("Failed to generate temporal information: %s", str(e))
        raise

def detect_contours_extract_ROIs(cluster_data, im, gauss=15, thresh=0):
    """Detect contours and extract ROIs from image data."""
    try:
        cluster_data.set_contour_params(gauss, thresh)
        contours = cluster_data.detect_contours(im)
        rois = cluster_data.extract_ROIs(contours)
        return contours, rois
    except Exception as e:
        logging.error("Failed to detect contours and extract ROIs: %s", str(e))
        raise

def process_full_pipeline(file_name, config):
    """
    Process full iQID pipeline from listmode data to final results.
    
    Parameters:
    -----------
    file_name : str
        Path to listmode data file
    config : dict
        Configuration parameters
        
    Returns:
    --------
    dict
        Processing results including images and metadata
    """
    results = {}
    
    try:
        # Load and process listmode data
        cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num = load_and_process_listmode_data(
            file_name, 
            c_area_thresh=config.get('c_area_thresh', 15)
        )
        
        # Filter and analyze
        cluster_data = filter_correct_analyze_data(
            cluster_data,
            config.get('binfac', 1),
            config.get('ROI_area_thresh', 100),
            config.get('t_binsize', 1000),
            config.get('t_half', 9.9 * 24 * 3600)  # Ac-225 half-life in seconds
        )
        
        # Generate spatial images
        spatial_image = generate_spatial_images(cluster_data, subpx=config.get('subpx', 1))
        
        # Generate temporal information
        temporal_image = generate_temporal_information(
            cluster_data,
            event_fx=config.get('event_fx', 0.1)
        )
        
        # Detect contours and extract ROIs
        contours, rois = detect_contours_extract_ROIs(
            cluster_data,
            spatial_image,
            gauss=config.get('gauss', 15),
            thresh=config.get('thresh', 0)
        )
        
        results = {
            'cluster_data': cluster_data,
            'spatial_image': spatial_image,
            'temporal_image': temporal_image,
            'contours': contours,
            'rois': rois,
            'metadata': {
                'time_ms': time_ms,
                'cluster_area': cluster_area,
                'xC_global': xC_global,
                'yC_global': yC_global,
                'frame_num': frame_num
            }
        }
        
        logging.info(f"Successfully processed {file_name}")
        return results
        
    except Exception as e:
        logging.error(f"Failed to process full pipeline for {file_name}: {str(e)}")
        raise
