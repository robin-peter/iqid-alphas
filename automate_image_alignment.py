import os
import json
import logging
import argparse
import numpy as np
from skimage import io
from iqid.align import assemble_stack, assemble_stack_hne, coarse_stack, pad_stack_he, crop_down

# Configure logging
logging.basicConfig(filename='automate_image_alignment.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def align_and_register_images(image_dir, output_dir, fformat='tif', deg=2, avg_over=1, subpx=1, color=(0, 0, 0), convert_to_grayscale_for_ssd=True):
    try:
        logging.info(f"Assembling H&E stack from: {image_dir}")
        image_stack = assemble_stack_hne(imdir=image_dir, fformat=fformat, color=color, pad=True)

        if image_stack is None or len(image_stack) == 0:
            logging.warning(f"No images found or assembled from {image_dir}. Skipping alignment.")
            return
        
        logging.info(f"Coarsely aligning stack with {len(image_stack)} images.")
        aligned_stack = coarse_stack(image_stack, deg=deg, avg_over=avg_over, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)
        
        # Pad image stack
        # padded_stack = pad_stack_he(data_path=image_dir, fformat=fformat, color=color, savedir=output_dir) # Removed
        
        # Crop down image stack
        # cropped_stack = crop_down(padded_stack, aligned_stack) # Removed or commented out
        
        logging.info(f"Saving registered stack to: {output_dir}")
        save_registered_images(aligned_stack, output_dir, fformat)
    except Exception as e:
        logging.error(f"Failed to align and register images for {image_dir}: {str(e)}", exc_info=True)
        raise

def save_registered_images(image_stack, output_dir, fformat='tif'):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, image in enumerate(image_stack):
            io.imsave(os.path.join(output_dir, f'registered_image_{i}.{fformat}'), image)
    except Exception as e:
        logging.error("Failed to save registered images: %s", str(e))
        raise

def main(image_dir, output_dir): # fformat, pad, deg etc. are loaded from config inside main
    try:
        with open('config.json', 'r') as f:
            config_params = json.load(f)['automate_image_alignment']

        # image_dir and output_dir are from args
        fformat = config_params.get('fformat', 'tif')
        # pad = config_params.get('pad', False) # Still read but not explicitly used for H&E assembly padding
        deg = config_params.get('deg', 2)
        avg_over = config_params.get('avg_over', 1)
        subpx = config_params.get('subpx', 1) # Kept for signature, though not used by assemble_stack_hne
        color = tuple(config_params.get('color', [0, 0, 0]))


        align_and_register_images(image_dir, output_dir, fformat, deg, avg_over, subpx, color, convert_to_grayscale_for_ssd=True)
    except Exception as e:
        logging.error(f"Failed to complete main image alignment for {image_dir}: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and register image stacks.")
    parser.add_argument("image_dir", help="Directory containing images to align.")
    parser.add_argument("output_dir", help="Directory to save registered images.")
    # You can add other arguments here later if needed, e.g., for config file path
    # parser.add_argument("--config", default="config.json", help="Path to the configuration file.")
    args = parser.parse_args()

    # Call main with the parsed command-line arguments for image_dir and output_dir
    # Other parameters will still be loaded from config.json within main() for now
    main(args.image_dir, args.output_dir) # Parameters like fformat, deg, etc., will be loaded from config inside main
