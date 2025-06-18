#!/usr/bin/env python3
"""
Test script to debug UCSF data loading issues.
"""

import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Try loading a real file with different methods
test_file = Path("data/DataPush1/iQID/Sequential/kidneys/D1M1_L/mBq_corr_11.tif")

print(f"Testing file: {test_file}")
print(f"File exists: {test_file.exists()}")
print(f"File size: {test_file.stat().st_size if test_file.exists() else 'N/A'}")

if test_file.exists():
    # Try with PIL
    try:
        print("\n=== Testing with PIL ===")
        img = Image.open(test_file)
        print(f"PIL - Size: {img.size}, Mode: {img.mode}")
        img_array = np.array(img)
        print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"Value range: {img_array.min()} - {img_array.max()}")
    except Exception as e:
        print(f"PIL failed: {e}")
    
    # Try with skimage
    try:
        print("\n=== Testing with skimage ===")
        from skimage import io
        img = io.imread(test_file)
        print(f"skimage - Shape: {img.shape}, dtype: {img.dtype}")
        print(f"Value range: {img.min()} - {img.max()}")
    except Exception as e:
        print(f"skimage failed: {e}")
    
    # Try with tifffile
    try:
        print("\n=== Testing with tifffile ===")
        import tifffile
        img = tifffile.imread(test_file)
        print(f"tifffile - Shape: {img.shape}, dtype: {img.dtype}")
        print(f"Value range: {img.min()} - {img.max()}")
    except Exception as e:
        print(f"tifffile failed: {e}")

# Test H&E file too
he_test_file = Path("data/DataPush1/HE/Upper and Lower from 10um Sequential sections/D1-M1-Seq-Kidney-Upper_c820cf62-844f-485e-bcdd-7c12af7eef9d_scene_1of2.tif")

print(f"\n\nTesting H&E file: {he_test_file}")
print(f"File exists: {he_test_file.exists()}")
print(f"File size: {he_test_file.stat().st_size / (1024*1024):.2f} MB" if he_test_file.exists() else "N/A")

if he_test_file.exists():
    try:
        print("\n=== Testing H&E with PIL ===")
        img = Image.open(he_test_file)
        print(f"PIL - Size: {img.size}, Mode: {img.mode}")
        img_array = np.array(img)
        print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"Value range: {img_array.min()} - {img_array.max()}")
    except Exception as e:
        print(f"H&E PIL failed: {e}")
