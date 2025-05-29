import os
import numpy as np
from skimage import io
from iqid.align import assemble_stack, coarse_stack, pad_stack_he, crop_down

def align_and_register_images(image_dir, output_dir, fformat='tif', pad=False, deg=2, avg_over=1, subpx=1, color=(0, 0, 0)):
    # Assemble image stack
    image_stack = assemble_stack(imdir=image_dir, fformat=fformat, pad=pad)
    
    # Coarse align image stack
    aligned_stack = coarse_stack(image_stack, deg=deg, avg_over=avg_over)
    
    # Pad image stack
    padded_stack = pad_stack_he(data_path=image_dir, fformat=fformat, color=color, savedir=output_dir)
    
    # Crop down image stack
    cropped_stack = crop_down(padded_stack, aligned_stack)
    
    # Save registered image stack
    save_registered_images(cropped_stack, output_dir, fformat)

def save_registered_images(image_stack, output_dir, fformat='tif'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(image_stack):
        io.imsave(os.path.join(output_dir, f'registered_image_{i}.{fformat}'), image)

def main(image_dir, output_dir, fformat='tif', pad=False, deg=2, avg_over=1, subpx=1, color=(0, 0, 0)):
    align_and_register_images(image_dir, output_dir, fformat, pad, deg, avg_over, subpx, color)

if __name__ == "__main__":
    image_dir = "path/to/your/image_directory"
    output_dir = "path/to/save/registered_images"
    main(image_dir, output_dir)
