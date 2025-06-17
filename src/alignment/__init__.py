"""
Image alignment processing module.
"""

import os
import logging
import numpy as np
from skimage import io
from ..core.iqid.align import assemble_stack, assemble_stack_hne, coarse_stack, pad_stack_he, crop_down

def align_and_register_images(image_dir, output_dir, fformat='tif', deg=2, avg_over=1, 
                            subpx=1, color=(0, 0, 0), convert_to_grayscale_for_ssd=True):
    """
    Align and register images from a directory.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing images to align
    output_dir : str
        Directory to save registered images
    fformat : str
        Image format (default: 'tif')
    deg : int
        Polynomial degree for alignment (default: 2)
    avg_over : int
        Averaging parameter (default: 1)
    subpx : int
        Subpixel factor (default: 1)
    color : tuple
        Color for padding (default: (0, 0, 0))
    convert_to_grayscale_for_ssd : bool
        Convert to grayscale for SSD calculation (default: True)
    """
    try:
        logging.info(f"Assembling H&E stack from: {image_dir}")
        image_stack = assemble_stack_hne(imdir=image_dir, fformat=fformat, color=color, pad=True)

        if image_stack is None or len(image_stack) == 0:
            logging.warning(f"No images found or assembled from {image_dir}. Skipping alignment.")
            return
        
        logging.info(f"Coarsely aligning stack with {len(image_stack)} images.")
        aligned_stack = coarse_stack(image_stack, deg=deg, avg_over=avg_over, 
                                   convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)
        
        logging.info(f"Saving registered stack to: {output_dir}")
        save_registered_images(aligned_stack, output_dir, fformat)
        
        return aligned_stack
        
    except Exception as e:
        logging.error(f"Failed to align and register images for {image_dir}: {str(e)}", exc_info=True)
        raise

def save_registered_images(image_stack, output_dir, fformat='tif'):
    """Save a stack of registered images to disk."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, image in enumerate(image_stack):
            io.imsave(os.path.join(output_dir, f'registered_image_{i}.{fformat}'), image)
    except Exception as e:
        logging.error("Failed to save registered images: %s", str(e))
        raise

def align_iqid_only(image_dir, output_dir, **kwargs):
    """Align iQID-only images using boundary-based methods."""
    try:
        logging.info(f"Assembling iQID stack from: {image_dir}")
        image_stack = assemble_stack(imdir=image_dir, **kwargs)
        
        if image_stack is None or len(image_stack) == 0:
            logging.warning(f"No images found or assembled from {image_dir}. Skipping alignment.")
            return
            
        # Use coarse alignment for iQID
        aligned_stack = coarse_stack(image_stack, **kwargs)
        
        logging.info(f"Saving aligned iQID stack to: {output_dir}")
        save_registered_images(aligned_stack, output_dir)
        
        return aligned_stack
        
    except Exception as e:
        logging.error(f"Failed to align iQID images for {image_dir}: {str(e)}", exc_info=True)
        raise
