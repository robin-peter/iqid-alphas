# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the IQID-Alphas pipeline.

## Quick Diagnostics

### 1. Run System Validation

First, run the built-in validation to identify issues:

```bash
# Production readiness check
python evaluation/scripts/production_validation.py

# Comprehensive pipeline evaluation
python evaluation/scripts/comprehensive_pipeline_evaluator.py
```

### 2. Check Dependencies

Verify all required packages are installed:

```python
import sys
required_packages = ['numpy', 'matplotlib', 'tifffile', 'scipy', 'pandas', 'skimage']

for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package}: OK")
    except ImportError:
        print(f"❌ {package}: MISSING")
```

## Common Issues and Solutions

### Import and Module Errors

#### Issue: `No module named 'pipelines'`

**Symptoms:**
```
ImportError: No module named 'pipelines'
ModuleNotFoundError: No module named 'pipelines.simplified_iqid_pipeline'
```

**Solutions:**

1. **Ensure correct working directory**:
   ```bash
   cd /path/to/iqid-alphas  # Must be in root directory
   python your_script.py
   ```

2. **Add to Python path**:
   ```python
   import sys
   sys.path.insert(0, '.')
   sys.path.insert(0, './pipelines')
   from simplified_iqid_pipeline import SimpleiQIDPipeline
   ```

3. **Use absolute imports**:
   ```python
   import os
   os.chdir('/path/to/iqid-alphas')
   from pipelines.simplified_iqid_pipeline import SimpleiQIDPipeline
   ```

#### Issue: `No module named 'src'`

**Symptoms:**
```
ImportError: No module named 'src'
ImportError: No module named 'src.core.iqid'
```

**Solutions:**

1. **Check directory structure**:
   ```bash
   ls -la src/core/iqid/  # Should contain align.py, helper.py, etc.
   ```

2. **Verify __init__.py files**:
   ```bash
   find src -name "__init__.py"  # Should exist in all directories
   ```

3. **Add src to path**:
   ```python
   import sys
   sys.path.insert(0, './src')
   ```

### Dependency Issues

#### Issue: Missing scikit-image

**Symptoms:**
```
ImportError: No module named 'skimage'
AttributeError: module 'skimage' has no attribute 'filters'
```

**Solutions:**

1. **Install scikit-image**:
   ```bash
   pip install scikit-image
   # Or for conda users:
   conda install scikit-image
   ```

2. **Verify installation**:
   ```python
   import skimage
   print(skimage.__version__)
   ```

#### Issue: TIFF reading errors

**Symptoms:**
```
ValueError: Cannot read TIFF file
OSError: cannot identify image file
```

**Solutions:**

1. **Install tifffile**:
   ```bash
   pip install tifffile
   ```

2. **Check file format**:
   ```python
   from tifffile import imread, TiffFile
   
   # Check if file is valid TIFF
   try:
       with TiffFile('your_file.tif') as tif:
           print(f"Valid TIFF: {tif.pages}")
   except Exception as e:
       print(f"Invalid TIFF: {e}")
   ```

3. **Convert file format**:
   ```python
   from PIL import Image
   import numpy as np
   from tifffile import imwrite
   
   # Convert other formats to TIFF
   img = Image.open('your_image.png')
   img_array = np.array(img)
   imwrite('converted_image.tif', img_array)
   ```

### Memory and Performance Issues

#### Issue: Out of Memory

**Symptoms:**
```
MemoryError: Unable to allocate array
numpy.core._exceptions.MemoryError
```

**Solutions:**

1. **Process smaller batches**:
   ```python
   # Instead of processing all at once
   large_dataset = get_all_samples()
   
   # Process in smaller batches
   batch_size = 5
   for i in range(0, len(large_dataset), batch_size):
       batch = large_dataset[i:i+batch_size]
       process_batch(batch)
   ```

2. **Reduce image resolution**:
   ```python
   from skimage.transform import rescale
   
   # Reduce image size by 50%
   small_image = rescale(original_image, 0.5, 
                        preserve_range=True, anti_aliasing=True)
   ```

3. **Use tile processing for large images**:
   ```python
   def process_large_image_in_tiles(image, tile_size=1024, overlap=128):
       height, width = image.shape[:2]
       results = []
       
       for y in range(0, height, tile_size - overlap):
           for x in range(0, width, tile_size - overlap):
               tile = image[y:y+tile_size, x:x+tile_size]
               result = process_tile(tile)
               results.append((y, x, result))
       
       return merge_tiles(results, height, width)
   ```

4. **Configure memory limits**:
   ```json
   {
     "performance": {
       "memory_management": {
         "max_image_size": 2048,
         "tile_processing": true,
         "tile_size": 512
       }
     }
   }
   ```

#### Issue: Slow Processing

**Symptoms:**
- Processing takes very long time
- CPU usage is low
- Single-threaded execution

**Solutions:**

1. **Enable multiprocessing**:
   ```json
   {
     "performance": {
       "multiprocessing": {
         "enabled": true,
         "n_jobs": -1
       }
     }
   }
   ```

2. **Optimize parameters**:
   ```json
   {
     "processing": {
       "gaussian_blur_sigma": 2.0,    // Higher = faster but less precise
       "median_filter_size": 5        // Larger = faster but more smoothing
     },
     "alignment": {
       "pyramid_levels": 2,           // Fewer levels = faster
       "correlation_threshold": 0.6   // Lower threshold = faster convergence
     }
   }
   ```

3. **Use caching**:
   ```json
   {
     "performance": {
       "caching": {
         "enabled": true,
         "cache_directory": "./cache"
       }
     }
   }
   ```

### Data and File Issues

#### Issue: File Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory
IOError: cannot read file
```

**Solutions:**

1. **Check file paths**:
   ```python
   import os
   
   file_path = 'path/to/your/file.tif'
   if os.path.exists(file_path):
       print(f"✅ File exists: {file_path}")
   else:
       print(f"❌ File not found: {file_path}")
       # List directory contents
       dir_path = os.path.dirname(file_path)
       if os.path.exists(dir_path):
           print(f"Directory contents: {os.listdir(dir_path)}")
   ```

2. **Use absolute paths**:
   ```python
   import os
   
   # Convert relative to absolute path
   relative_path = 'test_data/sample_001/iqid/image.tif'
   absolute_path = os.path.abspath(relative_path)
   ```

3. **Check file permissions**:
   ```bash
   ls -la your_file.tif  # Check file permissions
   chmod 644 your_file.tif  # Fix permissions if needed
   ```

#### Issue: Invalid Image Data

**Symptoms:**
```
ValueError: Input image must be 2D or 3D
TypeError: Input image dtype must be numeric
```

**Solutions:**

1. **Check image properties**:
   ```python
   from tifffile import imread
   
   image = imread('your_image.tif')
   print(f"Shape: {image.shape}")
   print(f"Dtype: {image.dtype}")
   print(f"Min/Max: {image.min()}/{image.max()}")
   ```

2. **Fix image format**:
   ```python
   import numpy as np
   
   # Convert to appropriate format
   if image.dtype == np.bool_:
       image = image.astype(np.uint8) * 255
   elif image.dtype == np.float64:
       image = (image * 255).astype(np.uint8)
   
   # Ensure 2D for grayscale
   if len(image.shape) == 3 and image.shape[2] == 1:
       image = image.squeeze()
   ```

3. **Handle different color formats**:
   ```python
   def standardize_image(image):
       # Handle different input formats
       if len(image.shape) == 3:
           if image.shape[2] == 3:  # RGB
               # Convert to grayscale if needed
               from skimage.color import rgb2gray
               image = rgb2gray(image)
           elif image.shape[2] == 4:  # RGBA
               image = image[:, :, :3]  # Remove alpha channel
       
       # Ensure proper data type
       if image.dtype == np.bool_:
           image = image.astype(np.uint8) * 255
       elif image.dtype in [np.float32, np.float64]:
           if image.max() <= 1.0:
               image = (image * 255).astype(np.uint8)
           else:
               image = image.astype(np.uint8)
       
       return image
   ```

### Processing and Algorithm Issues

#### Issue: Segmentation Failures

**Symptoms:**
- No objects detected
- Over-segmentation (too many small objects)
- Under-segmentation (objects merged together)

**Solutions:**

1. **Adjust thresholding parameters**:
   ```json
   {
     "segmentation": {
       "tissue": {
         "method": "li",              // Try different methods
         "morphology": {
           "opening_size": 3,         // Increase for noise reduction
           "closing_size": 5,         // Increase to fill gaps
           "remove_small_objects": 100 // Adjust size threshold
         }
       }
     }
   }
   ```

2. **Debug segmentation visually**:
   ```python
   import matplotlib.pyplot as plt
   from skimage import filters
   
   # Visualize intermediate steps
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # Original image
   axes[0, 0].imshow(original_image, cmap='gray')
   axes[0, 0].set_title('Original')
   
   # After preprocessing
   processed = filters.gaussian(original_image, sigma=1.0)
   axes[0, 1].imshow(processed, cmap='gray')
   axes[0, 1].set_title('Preprocessed')
   
   # Threshold
   threshold = filters.threshold_otsu(processed)
   binary = processed > threshold
   axes[1, 0].imshow(binary, cmap='gray')
   axes[1, 0].set_title('Thresholded')
   
   # Final segmentation
   axes[1, 1].imshow(final_segmentation, cmap='gray')
   axes[1, 1].set_title('Final Segmentation')
   
   plt.tight_layout()
   plt.show()
   ```

3. **Try different segmentation methods**:
   ```python
   from skimage import filters
   
   # Test different thresholding methods
   methods = ['otsu', 'li', 'triangle', 'yen']
   
   for method in methods:
       if method == 'otsu':
           threshold = filters.threshold_otsu(image)
       elif method == 'li':
           threshold = filters.threshold_li(image)
       elif method == 'triangle':
           threshold = filters.threshold_triangle(image)
       elif method == 'yen':
           threshold = filters.threshold_yen(image)
       
       binary = image > threshold
       print(f"{method}: {binary.sum()} pixels segmented")
   ```

#### Issue: Alignment Failures

**Symptoms:**
```
Warning: Alignment correlation below threshold
ValueError: Alignment failed - insufficient overlap
```

**Solutions:**

1. **Check image overlap**:
   ```python
   def check_image_overlap(img1, img2):
       # Simple correlation check
       from scipy.signal import correlate2d
       
       correlation = correlate2d(img1, img2, mode='valid')
       max_corr = correlation.max()
       print(f"Maximum correlation: {max_corr}")
       
       return max_corr > 0.5  # Threshold for sufficient overlap
   ```

2. **Adjust alignment parameters**:
   ```json
   {
     "alignment": {
       "max_translation": 100,        // Increase search range
       "correlation_threshold": 0.5,  // Lower threshold
       "pyramid_levels": 4,           // More pyramid levels
       "method": "feature_based"      // Try different method
     }
   }
   ```

3. **Preprocess images for alignment**:
   ```python
   from skimage import filters, feature
   
   def prepare_for_alignment(image):
       # Enhance edges for better alignment
       edges = filters.sobel(image)
       
       # Or use Harris corners
       corners = feature.corner_harris(image)
       
       return edges  # or corners
   ```

4. **Manual alignment check**:
   ```python
   import matplotlib.pyplot as plt
   
   def visualize_alignment(img1, img2, shift):
       fig, axes = plt.subplots(1, 3, figsize=(15, 5))
       
       axes[0].imshow(img1, cmap='gray')
       axes[0].set_title('Reference Image')
       
       axes[1].imshow(img2, cmap='gray')
       axes[1].set_title('Moving Image')
       
       # Apply shift and overlay
       from scipy.ndimage import shift
       aligned = shift(img2, shift)
       overlay = 0.5 * img1 + 0.5 * aligned
       axes[2].imshow(overlay, cmap='gray')
       axes[2].set_title('Alignment Overlay')
       
       plt.show()
   ```

### Configuration Issues

#### Issue: Invalid Configuration

**Symptoms:**
```
KeyError: 'processing'
ValueError: Invalid configuration parameter
```

**Solutions:**

1. **Validate configuration format**:
   ```python
   import json
   
   def validate_config(config_path):
       try:
           with open(config_path, 'r') as f:
               config = json.load(f)
           print("✅ Valid JSON format")
           return config
       except json.JSONDecodeError as e:
           print(f"❌ Invalid JSON: {e}")
           return None
   ```

2. **Check required fields**:
   ```python
   required_fields = ['processing', 'segmentation', 'alignment', 'output']
   
   for field in required_fields:
       if field not in config:
           print(f"❌ Missing required field: {field}")
       else:
           print(f"✅ Found field: {field}")
   ```

3. **Use default configuration**:
   ```python
   def get_default_config():
       return {
           "processing": {
               "gaussian_blur_sigma": 1.0,
               "threshold_method": "otsu"
           },
           "segmentation": {
               "min_object_size": 100
           },
           "alignment": {
               "max_translation": 50,
               "correlation_threshold": 0.7
           },
           "output": {
               "save_intermediate": True
           }
       }
   ```

## Advanced Troubleshooting

### Enable Debug Mode

1. **Set debug configuration**:
   ```json
   {
     "debug": {
       "save_debug_images": true,
       "verbose_logging": true,
       "profile_performance": true
     }
   }
   ```

2. **Check debug outputs**:
   ```bash
   ls -la outputs/debug/  # Look for debug images and logs
   ```

### Performance Profiling

```python
import cProfile
import pstats

def profile_pipeline():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your pipeline
    pipeline = SimpleiQIDPipeline()
    results = pipeline.process_sample('test_image.tif')
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Log Analysis

Check log files for detailed error information:

```bash
# Find recent log files
find . -name "*.log" -mtime -1

# View error messages
grep -i "error\|exception\|failed" *.log

# Check warning messages
grep -i "warning" *.log
```

## Getting Additional Help

### 1. Create a Minimal Reproducible Example

```python
# Minimal example that reproduces the issue
from pipelines.simplified_iqid_pipeline import SimpleiQIDPipeline
import numpy as np

# Create test data
test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

# Try to reproduce the issue
pipeline = SimpleiQIDPipeline()
try:
    result = pipeline.process_sample(test_image)
    print("✅ No issue with test data")
except Exception as e:
    print(f"❌ Issue reproduced: {e}")
```

### 2. Collect System Information

```python
import sys
import platform
import numpy as np

print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"NumPy version: {np.__version__}")

# Check available memory
import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")
```

### 3. Run Comprehensive Validation

```bash
# Full system validation
python evaluation/scripts/production_validation.py > validation_report.txt 2>&1

# Include the validation_report.txt when seeking help
```

If you're still experiencing issues after trying these solutions, please provide:
1. The complete error message and stack trace
2. Your configuration file
3. System information (Python version, OS, available memory)
4. Validation report output
5. A minimal example that reproduces the issue
