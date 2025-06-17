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

def align_and_register_images(image_dir, output_dir, fformat='tif', pad=False, deg=2, avg_over=1, subpx=1, color=(0, 0, 0)):
    try:
        # Assemble image stack
        image_stack = assemble_stack_hne(imdir=image_dir, fformat=fformat, color=color, pad=True)
        
        # Coarse align image stack
        aligned_stack = coarse_stack(image_stack, deg=deg, avg_over=avg_over)
        
        # Pad image stack
        padded_stack = pad_stack_he(data_path=image_dir, fformat=fformat, color=color, savedir=output_dir)
        
        # Crop down image stack
        cropped_stack = crop_down(padded_stack, aligned_stack)
        
        # Save registered image stack
        save_registered_images(cropped_stack, output_dir, fformat)
    except Exception as e:
        logging.error("Failed to align and register images: %s", str(e))
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

def main(image_dir, output_dir, fformat='tif', pad=False, deg=2, avg_over=1, subpx=1, color=(0, 0, 0)):
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        # image_dir and output_dir are now passed as arguments
        fformat = config['automate_image_alignment']['fformat']
        pad = config['automate_image_alignment']['pad']
        deg = config['automate_image_alignment']['deg']
        avg_over = config['automate_image_alignment']['avg_over']
        subpx = config['automate_image_alignment']['subpx']
        color = config['automate_image_alignment']['color']

        align_and_register_images(image_dir, output_dir, fformat, pad, deg, avg_over, subpx, color)
    except Exception as e:
        logging.error("Failed to complete main image alignment: %s", str(e))
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
    main(args.image_dir, args.output_dir)
