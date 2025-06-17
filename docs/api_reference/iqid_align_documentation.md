# `iqid.align` Module Documentation

This module provides functions for alignment and registration of iQID activity image stacks. It includes utilities for assembling image stacks from files, padding images, performing coarse alignment based on rotation and SSD minimization, cropping, and various image transformation and visualization helpers. It relies on libraries such as NumPy, scikit-image, OpenCV, and Matplotlib.

---

## Functions

### `get_maxdim(fileList)`
- **Purpose:** Gets the maximum image height and width from a list of image files without reading the entire image data into memory. Uses the `imagesize` package.
- **Parameters:**
    - `fileList` (list[str]): List of file paths for the images.
- **Returns:**
    - `tuple[int, int]`: `(maxh, maxw)` - Maximum height and maximum width found among the images.
- **Function Signature:**
  ```python
  def get_maxdim(fileList: list[str]) -> tuple[int, int]:
  ```

---

### `assemble_stack(imdir=None, fformat='tif', pad=False)`
- **Purpose:** Loads a sequence of images from a directory, sorts them using natural sort order, and assembles them into a NumPy stack. Optionally pads images to the maximum dimensions found in the stack.
- **Parameters:**
    - `imdir` (str, optional): The directory containing the image files. Defaults to `None` (implies current directory or that path construction happens before call).
    - `fformat` (str, optional): File extension of the images (e.g., 'tif', 'png'). Defaults to `'tif'`.
    - `pad` (bool, optional): If `True`, images are padded with zeros to match the dimensions of the largest image in the stack. Defaults to `False`.
- **Returns:**
    - `np.ndarray`: A 3D NumPy array representing the image stack.
- **Dependencies:** `glob`, `os`, `numpy`, `skimage.io`, `cv2` (for padding), `iqid.helper.natural_keys`.
- **Function Signature:**
  ```python
  def assemble_stack(imdir: str | None = None, fformat: str = 'tif', pad: bool = False) -> np.ndarray:
  ```

---

### `assemble_stack_hne(imdir=None, fformat='tif', color=(0, 0, 0), pad=True)`
- **Purpose:** Assembles a stack of H&E (color) images from a directory. Similar to `assemble_stack` but specifically handles 3-channel (color) images and pads with a specified color if `pad` is True.
- **Parameters:**
    - `imdir` (str, optional): Directory containing the images.
    - `fformat` (str, optional): File extension. Defaults to `'tif'`.
    - `color` (tuple[int, int, int], optional): RGB color tuple to use for padding. Defaults to `(0, 0, 0)` (black).
    - `pad` (bool, optional): If `True` (default), images are padded to maximum dimensions.
- **Returns:**
    - `np.ndarray`: A 4D NumPy array (num_images, height, width, channels) of type `int`.
- **Dependencies:** `glob`, `os`, `numpy`, `skimage.io`, `iqid.helper.natural_keys`.
- **Function Signature:**
  ```python
  def assemble_stack_hne(imdir: str | None = None, fformat: str = 'tif', color: tuple[int, int, int] = (0, 0, 0), pad: bool = True) -> np.ndarray:
  ```

---

### `pad_stack_he(data_path, fformat='png', color=(0, 0, 0), savedir=None, verbose=False)`
- **Purpose:** Pads a stack of images (grayscale or color) to the maximum dimensions found in the set and saves them to a new directory.
- **Parameters:**
    - `data_path` (str): Directory containing the images to pad.
    - `fformat` (str, optional): File extension. Defaults to `'png'`.
    - `color` (tuple[int, int, int] | int, optional): Color for padding. For RGB, a 3-tuple. For grayscale, the first element of the tuple or an int. Defaults to `(0, 0, 0)`.
    - `savedir` (str, optional): Directory where the padded images will be saved (in a subdirectory named 'padded').
    - `verbose` (bool, optional): If `True`, prints the list of files. Defaults to `False`.
- **Returns:** None. Saves padded images to disk.
- **Dependencies:** `glob`, `os`, `numpy`, `skimage.io`, `pathlib.Path`, `tqdm.trange`, `imagesize.get` (via `get_maxdim`).
- **Function Signature:**
  ```python
  def pad_stack_he(data_path: str, fformat: str = 'png', color: tuple[int, int, int] | int = (0, 0, 0), savedir: str | None = None, verbose: bool = False) -> None:
  ```

---

### `organize_onedir(imdir=None, include_idx=[], order_idx=[], fformat='png')`
- **Purpose:** Organizes images within a single directory by optionally deleting some images and renaming others based on provided indices and naming conventions. **Caution: This function can modify files in place.**
- **Parameters:**
    - `imdir` (str, optional): The directory to organize.
    - `include_idx` (list[bool] | np.ndarray[bool], optional): A boolean list/array indicating which files to keep. If empty, keeps all.
    - `order_idx` (list[int] | np.ndarray[int], optional): An array specifying the new numerical suffix for renaming files. If empty, uses original order.
    - `fformat` (str, optional): File extension. Defaults to `'png'`.
- **Returns:** None. Modifies files in `imdir`.
- **Dependencies:** `glob`, `os`, `numpy`, `iqid.helper.natural_keys`, `iqid.helper.get_yn`.
- **Notes:** Uses a `nameDict` for renaming prefixes. Prompts user for confirmation before deleting or renaming. The variable `onepath` seems to be a typo and likely should be `imdir`.

---

### `preprocess_topdir(topdir, include_idx=[], order_idx=[])`
- **Purpose:** Preprocesses images across multiple subdirectories within a top directory. Allows for selective deletion and renaming of images based on predefined dictionaries for names and formats. **Caution: This function can modify files in place.**
- **Parameters:**
    - `topdir` (str): The top directory containing subdirectories of images.
    - `include_idx` (list[bool] | np.ndarray[bool], optional): Boolean list/array for including files.
    - `order_idx` (list[int] | np.ndarray[int], optional): Array for ordering/renaming files.
- **Returns:** None. Modifies files within subdirectories of `topdir`.
- **Dependencies:** `glob`, `os`, `numpy`, `iqid.helper.list_studies`, `iqid.helper.natural_keys`, `iqid.helper.get_yn`.
- **Notes:** Uses `nameDict` and `fformatDict`. Prompts user for confirmation.

---

### `ignore_images(data, exclude_list, pad='backwards')`
- **Purpose:** Excludes specified images from a stack. Can either remove them or replace them by padding with adjacent images.
- **Parameters:**
    - `data` (np.ndarray): The input 3D image stack.
    - `exclude_list` (list[int] | np.ndarray[int]): List of indices of images to exclude.
    - `pad` (str | bool, optional):
        - `'backwards'` (default): Replace excluded image with the previous one.
        - `'forwards'`: Replace excluded image with the next one.
        - If `False` (or any other string not 'backwards'/'forwards' when `pad` is True, though the code only explicitly checks these two if `pad` is True): Remove the image from the stack.
- **Returns:**
    - `np.ndarray`: The modified image stack.
- **Function Signature:**
  ```python
  def ignore_images(data: np.ndarray, exclude_list: list[int] | np.ndarray[int], pad: str | bool = 'backwards') -> np.ndarray:
  ```
- **Note:** Docstring mentions `pad: bool` but code implies `pad` can be a string. The note about handling only 2 bad images in a row applies to the padding logic.

---

### `get_SSD(im1, im2)`
- **Purpose:** Computes the Sum of Squared Differences (SSD), equivalent to Mean Squared Error (MSE) if divided by N, between two images of the same shape.
- **Parameters:**
    - `im1` (np.ndarray): First image.
    - `im2` (np.ndarray): Second image.
- **Returns:**
    - `float`: The SSD value.
- **Function Signature:**
  ```python
  def get_SSD(im1: np.ndarray, im2: np.ndarray) -> float:
  ```

---

### `coarse_rotation(mov, ref, deg=2, interpolation=0, gauss=5, preserve_range=True, recenter=False)`
- **Purpose:** Performs coarse rotation of a "moving" image (`mov`) to align it with a "reference" image (`ref`) by minimizing SSD. It tests rotations in increments of `deg`.
- **Parameters:**
    - `mov` (np.ndarray): The image to be rotated.
    - `ref` (np.ndarray): The reference image.
    - `deg` (float, optional): Angular increment for rotation tests (degrees). Defaults to `2`.
    - `interpolation` (int, optional): Interpolation order for `skimage.transform.rotate` (0-5). 0 is nearest-neighbor. Defaults to `0`.
    - `gauss` (int, optional): Size of Gaussian blur kernel applied to images before SSD comparison. If 0 or None, no blur. Defaults to `5`.
    - `preserve_range` (bool, optional): Passed to `skimage.transform.rotate`. Defaults to `True`.
    - `recenter` (bool, optional): If `True`, recenters both images using `recenter_im` before processing. Defaults to `False`.
- **Returns:**
    - `tuple[np.ndarray, float]`: `(rot, outdeg)`
        - `rot` (np.ndarray): The rotated version of `mov`.
        - `outdeg` (float): The optimal rotation angle found.
- **Dependencies:** `numpy`, `cv2.GaussianBlur`, `skimage.transform.rotate`, `get_SSD`, `recenter_im`.
- **Function Signature:**
  ```python
  def coarse_rotation(mov: np.ndarray, ref: np.ndarray, deg: float = 2, interpolation: int = 0, gauss: int = 5, preserve_range: bool = True, recenter: bool = False) -> tuple[np.ndarray, float]:
  ```

---

### `coarse_stack(unreg, deg=2, avg_over=1, preserve_range=True, return_deg=False)`
- **Purpose:** Aligns an entire stack of images using `coarse_rotation`. Each image is aligned to the previous one or an average of previously aligned images.
- **Parameters:**
    - `unreg` (np.ndarray): The 3D stack of unregistered images.
    - `deg` (float, optional): Angular increment for `coarse_rotation`. Defaults to `2`.
    - `avg_over` (int, optional): Number of prior aligned images to average for the reference. `1` means align to the immediate previous. Defaults to `1`.
    - `preserve_range` (bool, optional): Passed to `coarse_rotation`. Defaults to `True`.
    - `return_deg` (bool, optional): If `True`, also returns the array of rotation degrees applied to each slice. Defaults to `False`.
- **Returns:**
    - `np.ndarray` or `tuple[np.ndarray, np.ndarray]`:
        - `reg` (np.ndarray): The (coarsely) registered stack.
        - `degs` (np.ndarray, optional): If `return_deg` is `True`, the rotation degrees.
- **Dependencies:** `numpy`, `coarse_rotation`.
- **Function Signature:**
  ```python
  def coarse_stack(unreg: np.ndarray, deg: float = 2, avg_over: int = 1, preserve_range: bool = True, return_deg: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  ```

---

### `format_circle(h, k, r, numpoints=400)`
- **Purpose:** Generates coordinates for a circle, often used as an initial contour for active contour algorithms.
- **Parameters:**
    - `h` (float): Row-coordinate of the circle center.
    - `k` (float): Column-coordinate of the circle center.
    - `r` (float): Radius of the circle.
    - `numpoints` (int, optional): Number of points to define the circle. Defaults to `400`.
- **Returns:**
    - `np.ndarray`: A (`numpoints`, 2) array of (row, col) coordinates.
- **Function Signature:**
  ```python
  def format_circle(h: float, k: float, r: float, numpoints: int = 400) -> np.ndarray:
  ```

---

### `binary_mask(img, finagle=1)`
- **Purpose:** Creates a binary mask from an image using Otsu's thresholding method, with an optional "finagle-factor" to adjust the threshold.
- **Parameters:**
    - `img` (np.ndarray): Input grayscale image.
    - `finagle` (float, optional): Multiplier for the Otsu threshold. Defaults to `1`.
- **Returns:**
    - `np.ndarray`: The binary mask (boolean or 0/1).
- **Dependencies:** `skimage.filters.threshold_otsu`.
- **Function Signature:**
  ```python
  def binary_mask(img: np.ndarray, finagle: float = 1) -> np.ndarray:
  ```

---

### `mask_from_contour(img, snake)`
- **Purpose:** Creates a binary mask from a contour (snake) by filling the polygon defined by the contour points.
- **Parameters:**
    - `img` (np.ndarray): The reference image (used for shape of the mask).
    - `snake` (np.ndarray): An array of (row, col) coordinates defining the contour.
- **Returns:**
    - `np.ndarray`: The binary mask with 1s inside the contour and 0s outside.
- **Dependencies:** `numpy`, `skimage.draw.polygon`.
- **Function Signature:**
  ```python
  def mask_from_contour(img: np.ndarray, snake: np.ndarray) -> np.ndarray:
  ```

---

### `to_shape(mov, x_, y_, vals=(0, 0))`
- **Purpose:** Pads a 2D image (`mov`) with constant values to reach a target shape (`x_`, `y_`). Padding is distributed equally on sides.
- **Parameters:**
    - `mov` (np.ndarray): The 2D image to pad.
    - `x_` (int): Target width.
    - `y_` (int): Target height.
    - `vals` (tuple, optional): Values to use for padding. If `mov` is grayscale, `vals[0]` might be used or it could be a single value. Defaults to `(0,0)`.
- **Returns:**
    - `np.ndarray`: The padded 2D image.
- **Function Signature:**
  ```python
  def to_shape(mov: np.ndarray, x_: int, y_: int, vals: tuple = (0, 0)) -> np.ndarray:
  ```

---

### `pad_2d_masks(movmask, refmask, func=to_shape)`
- **Purpose:** Pads two 2D masks (`movmask`, `refmask`) to the maximum dimensions found between them using a specified padding function (`func`).
- **Parameters:**
    - `movmask` (np.ndarray): First 2D mask.
    - `refmask` (np.ndarray): Second 2D mask.
    - `func` (callable, optional): The padding function to use. Defaults to `to_shape`.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray]`: `(movpad, refpad)` - The two padded masks.
- **Function Signature:**
  ```python
  def pad_2d_masks(movmask: np.ndarray, refmask: np.ndarray, func: callable = to_shape) -> tuple[np.ndarray, np.ndarray]:
  ```

---

### `to_shape_rgb(mov, x_, y_)`
- **Purpose:** Pads an RGB image (`mov`) to a target shape (`x_`, `y_`) with constant values (255, 255) - likely white.
- **Parameters:**
    - `mov` (np.ndarray): The 3D RGB image (height, width, channels) to pad.
    - `x_` (int): Target width.
    - `y_` (int): Target height.
- **Returns:**
    - `np.ndarray`: The padded RGB image.
- **Function Signature:**
  ```python
  def to_shape_rgb(mov: np.ndarray, x_: int, y_: int) -> np.ndarray:
  ```

---

### `pad_rgb_im(im_2d, im_rgb)`
- **Purpose:** Pads a 2D grayscale/binary image and an RGB image to the maximum dimensions found between them.
- **Parameters:**
    - `im_2d` (np.ndarray): The 2D grayscale or binary image.
    - `im_rgb` (np.ndarray): The 3D RGB image.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray]`: `(movpad, refpad)` - The padded 2D image and padded RGB image.
- **Dependencies:** `to_shape`, `to_shape_rgb`.
- **Function Signature:**
  ```python
  def pad_rgb_im(im_2d: np.ndarray, im_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  ```

---

### `to_shape_center(im, x_, y_)`
- **Purpose:** Pads a 2D image (`im`) to a target shape (`x_`, `y_`) by centering the original image within a new array filled with zeros.
- **Parameters:**
    - `im` (np.ndarray): The 2D image to pad and center.
    - `x_` (int): Target width.
    - `y_` (int): Target height.
- **Returns:**
    - `np.ndarray`: The padded and centered image.
- **Function Signature:**
  ```python
  def to_shape_center(im: np.ndarray, x_: int, y_: int) -> np.ndarray:
  ```

---

### `crop_down(rgb_overlay, rgb_ref, axis='both')`
- **Purpose:** Crops an overlay image (`rgb_overlay`) to match the size of a reference image (`rgb_ref`) by removing pixels from the edges.
- **Parameters:**
    - `rgb_overlay` (np.ndarray): The image to be cropped (e.g., H, W, C).
    - `rgb_ref` (np.ndarray): The reference image defining the target size (e.g., H', W', C).
    - `axis` (str, optional): Specifies which axes to crop. `'both'` (default), `'x'`, or `'y'`.
- **Returns:**
    - `np.ndarray`: The cropped overlay image.
- **Dependencies:** `skimage.util.crop`.
- **Function Signature:**
  ```python
  def crop_down(rgb_overlay: np.ndarray, rgb_ref: np.ndarray, axis: str = 'both') -> np.ndarray:
  ```

---

### `crop_to(im, x_, y_)`
- **Purpose:** Crops an image (`im`) to a target size (`x_` width, `y_` height) by removing pixels symmetrically from its borders.
- **Parameters:**
    - `im` (np.ndarray): The image to crop (2D or 3D for RGB).
    - `x_` (int): Target width.
    - `y_` (int): Target height.
- **Returns:**
    - `np.ndarray`: The cropped image. Issues a warning if target dimensions are larger than image dimensions.
- **Dependencies:** `skimage.util.crop`, `warnings`.
- **Function Signature:**
  ```python
  def crop_to(im: np.ndarray, x_: int, y_: int) -> np.ndarray:
  ```

---

### `overlay_images(imgs, equalize=False, aggregator=np.mean)`
- **Purpose:** Overlays a list of images into a single image, typically by averaging. Optionally equalizes histograms first. (From PyStackReg documentation).
- **Parameters:**
    - `imgs` (list[np.ndarray]): List of 2D images to overlay.
    - `equalize` (bool, optional): If `True`, applies histogram equalization to each image. Defaults to `False`.
    - `aggregator` (callable, optional): Function to combine images (e.g., `np.mean`, `np.sum`). Defaults to `np.mean`.
- **Returns:**
    - `np.ndarray`: The resulting overlay image.
- **Dependencies:** `numpy`, `skimage.exposure.equalize_hist`.
- **Function Signature:**
  ```python
  def overlay_images(imgs: list[np.ndarray], equalize: bool = False, aggregator: callable = np.mean) -> np.ndarray:
  ```

---

### `composite_images(imgs, equalize=False, aggregator=np.mean)`
- **Purpose:** Creates a color composite image from a list of up to 3 grayscale images, assigning each to a color channel (R, G, B). (From PyStackReg documentation). The `aggregator` parameter seems unused given the dstack operation.
- **Parameters:**
    - `imgs` (list[np.ndarray]): List of 2D images (1 to 3 images).
    - `equalize` (bool, optional): If `True`, applies histogram equalization. Defaults to `False`.
    - `aggregator` (callable, optional): Docstring mentions it, but it's not used in the code for the final dstacking. Defaults to `np.mean`.
- **Returns:**
    - `np.ndarray`: The RGB composite image.
- **Dependencies:** `numpy`, `skimage.exposure.equalize_hist`.
- **Function Signature:**
  ```python
  def composite_images(imgs: list[np.ndarray], equalize: bool = False, aggregator: callable = np.mean) -> np.ndarray:
  ```

---

### `save_imbatch(imstack, newdir, prefix, fformat='tif')`
- **Purpose:** Saves each image in a 3D stack to a specified directory with a given prefix and format.
- **Parameters:**
    - `imstack` (np.ndarray): 3D image stack (num_images, height, width).
    - `newdir` (str): Directory to save the images.
    - `prefix` (str): Prefix for the filenames (e.g., "image" -> "image_0.tif", "image_1.tif").
    - `fformat` (str, optional): File format. Defaults to `'tif'`.
- **Returns:** None. Saves images to disk.
- **Dependencies:** `pathlib.Path`, `tqdm.trange`, `skimage.io.imsave`.
- **Function Signature:**
  ```python
  def save_imbatch(imstack: np.ndarray, newdir: str, prefix: str, fformat: str = 'tif') -> None:
  ```

---

### `concatenate_dsets(astack_1, astack_2)`
- **Purpose:** Concatenates two 3D image stacks (`astack_1`, `astack_2`) along the first axis (number of images). Before concatenation, it pads all images in both stacks to the maximum height and width found across both stacks.
- **Parameters:**
    - `astack_1` (np.ndarray): First image stack.
    - `astack_2` (np.ndarray): Second image stack.
- **Returns:**
    - `np.ndarray`: The concatenated image stack.
- **Dependencies:** `numpy`, `pad_2d_masks` (and its dependency `to_shape`).
- **Function Signature:**
  ```python
  def concatenate_dsets(astack_1: np.ndarray, astack_2: np.ndarray) -> np.ndarray:
  ```

---

### `quantify_err(imstack, reg, tmat, vis=True)`
- **Purpose:** Quantifies errors introduced by registration, including shear, zoom, and percentage difference in summed activity. Optionally visualizes unregistered vs. registered overlays.
- **Parameters:**
    - `imstack` (np.ndarray): Original (unregistered) image stack.
    - `reg` (np.ndarray): Registered image stack.
    - `tmat` (np.ndarray): Array of transformation matrices (likely N x 3 x 3 or N x 2 x 3).
    - `vis` (bool, optional): If `True`, displays comparison plots. Defaults to `True`.
- **Returns:**
    - `float`: Percentage difference in summed activity between `reg` and `imstack`.
- **Dependencies:** `numpy`, `matplotlib.pyplot`, `iqid.helper.decompose_affine`, `overlay_images`.
- **Function Signature:**
  ```python
  def quantify_err(imstack: np.ndarray, reg: np.ndarray, tmat: np.ndarray, vis: bool = True) -> float:
  ```

---

### `simple_slice(arr, inds, axis)`
- **Purpose:** Helper function to slice a NumPy array along a specified axis using given indices. (From PyStackReg).
- **Parameters:**
    - `arr` (np.ndarray): The array to slice.
    - `inds` (int | slice | list[int]): Indices or slice object for the specified axis.
    - `axis` (int): The axis along which to slice.
- **Returns:**
    - `np.ndarray`: The sliced view of the array.
- **Function Signature:**
  ```python
  def simple_slice(arr: np.ndarray, inds: int | slice | list[int], axis: int) -> np.ndarray:
  ```

---

### `transform_stack(img, tmat, axis=0, order=0)`
- **Purpose:** Applies a series of affine transformations (`tmat`) to each slice of an image stack (`img`) along a specified `axis`. This is a modified version of a PyStackReg function to allow selection of interpolation order.
- **Parameters:**
    - `img` (np.ndarray): Input image stack.
    - `tmat` (np.ndarray): Array of transformation matrices, one for each slice to be transformed. Shape typically (num_slices, 2, 3) or (num_slices, 3, 3).
    - `axis` (int, optional): Axis along which slices are taken. Defaults to `0`.
    - `order` (int, optional): Interpolation order for `skimage.transform.warp`. Defaults to `0` (nearest-neighbor).
- **Returns:**
    - `np.ndarray`: The transformed image stack.
- **Dependencies:** `numpy`, `skimage.transform.warp`, `simple_slice`.
- **Function Signature:**
  ```python
  def transform_stack(img: np.ndarray, tmat: np.ndarray, axis: int = 0, order: int = 0) -> np.ndarray:
  ```

---

### `recenter_im(im)`
- **Purpose:** Recenters the content of an image by cropping to the bounding box of non-zero pixels and then padding it back to its original dimensions.
- **Parameters:**
    - `im` (np.ndarray): The 2D input image.
- **Returns:**
    - `np.ndarray`: The recentered image.
- **Dependencies:** `numpy`, `to_shape`.
- **Function Signature:**
  ```python
  def recenter_im(im: np.ndarray) -> np.ndarray:
  ```

---

### `downsamp(ref, factor)`
- **Purpose:** Downsamples an RGB image by a given factor using block reduction (averaging). Assumes input is normalized to 255.
- **Parameters:**
    - `ref` (np.ndarray): The RGB image (H, W, C) to downsample. Values are expected to be in a range that makes sense for division by 255 (e.g. 0-255).
    - `factor` (int): The downsampling factor.
- **Returns:**
    - `np.ndarray`: The downsampled RGB image.
- **Dependencies:** `numpy`, `skimage.measure.block_reduce`.
- **Function Signature:**
  ```python
  def downsamp(ref: np.ndarray, factor: int) -> np.ndarray:
  ```

---

### `shape_colorise(dr, ref, cmap=plt.cm.inferno)`
- **Purpose:** Colorizes a grayscale image (`dr`), crops it to the shape of a reference RGB image (`ref`), pads it to match `ref`'s dimensions, and converts to uint8.
- **Parameters:**
    - `dr` (np.ndarray): Grayscale image to colorize.
    - `ref` (np.ndarray): Reference RGB image (H, W, 3) for shaping.
    - `cmap` (matplotlib.colors.Colormap, optional): Colormap to use. Defaults to `plt.cm.inferno`.
- **Returns:**
    - `np.ndarray`: The shaped, colorized, and uint8-converted image.
- **Dependencies:** `numpy`, `matplotlib.pyplot.cm.inferno`, `matplotlib.colors.Normalize`, `crop_down`, `to_shape_rgb`.
- **Function Signature:**
  ```python
  def shape_colorise(dr: np.ndarray, ref: np.ndarray, cmap: plt.cm.colors.Colormap = plt.cm.inferno) -> np.ndarray:
  ```

---

### `do_transform(mov, fac, deg, tf)`
- **Purpose:** Applies a sequence of transformations: rescale, rotate, and warp (affine transform). Clips values to be non-negative and rounds them.
- **Parameters:**
    - `mov` (np.ndarray): The image to transform.
    - `fac` (float): Rescaling factor.
    - `deg` (float): Rotation angle in degrees.
    - `tf` (skimage.transform.ProjectiveTransform | skimage.transform.AffineTransform): Transformation object for `warp`.
- **Returns:**
    - `np.ndarray`: The transformed image.
- **Dependencies:** `skimage.transform.rescale`, `skimage.transform.rotate`, `skimage.transform.warp`, `numpy`.
- **Function Signature:**
  ```python
  def do_transform(mov: np.ndarray, fac: float, deg: float, tf) -> np.ndarray: # tf type is complex
  ```

---

### `rescale_tmat(tmat, s)`
- **Purpose:** Rescales the translation components (last column) of an affine transformation matrix.
- **Parameters:**
    - `tmat` (np.ndarray): A 2x3 or 3x3 affine transformation matrix.
    - `s` (float): Scaling factor for translation.
- **Returns:**
    - `np.ndarray`: The transformation matrix with rescaled translation.
- **Function Signature:**
  ```python
  def rescale_tmat(tmat: np.ndarray, s: float) -> np.ndarray:
  ```

---

### `tmat_3to2(tmat)`
- **Purpose:** Converts a 3D affine transformation matrix (presumably 3x4 or 4x4, though indexing suggests 3xN or Nx3) from BigWarp into a 2D affine matrix (3x3).
- **Parameters:**
    - `tmat` (np.ndarray): The input 3D transformation matrix. The indexing `tmat[:2, :2]`, `tmat[:2, -1]`, `tmat[2, :3]` implies specific structure.
- **Returns:**
    - `np.ndarray`: The 2D (3x3) affine transformation matrix.
- **Function Signature:**
  ```python
  def tmat_3to2(tmat: np.ndarray) -> np.ndarray:
  ```

---

### `do_transform_noscale(mov, ref, deg, tf)`
- **Purpose:** Applies rotation and warp (affine transformation) to an image, then recenters and crops it to match a reference image's shape. Uses nearest-neighbor interpolation and preserves range.
- **Parameters:**
    - `mov` (np.ndarray): Image to transform.
    - `ref` (np.ndarray): Reference image for final cropping.
    - `deg` (float): Rotation angle in degrees.
    - `tf` (skimage.transform.ProjectiveTransform | skimage.transform.AffineTransform): Transformation for `warp`.
- **Returns:**
    - `np.ndarray`: The transformed, recentered, and cropped image.
- **Dependencies:** `skimage.transform.rotate`, `skimage.transform.warp`, `recenter_im`, `crop_to`.
- **Function Signature:**
  ```python
  def do_transform_noscale(mov: np.ndarray, ref: np.ndarray, deg: float, tf) -> np.ndarray: # tf type is complex
  ```

---

### `do_transform_noPSR(mov, ref, deg)`
- **Purpose:** Applies only coarse rotation, then recenters and crops to match a reference image. (PSR likely refers to PyStackReg, implying this avoids its more complex registration).
- **Parameters:**
    - `mov` (np.ndarray): Image to transform.
    - `ref` (np.ndarray): Reference image for final cropping.
    - `deg` (float): Rotation angle in degrees.
- **Returns:**
    - `np.ndarray`: The transformed image.
- **Dependencies:** `skimage.transform.rotate`, `recenter_im`, `crop_to`.
- **Function Signature:**
  ```python
  def do_transform_noPSR(mov: np.ndarray, ref: np.ndarray, deg: float) -> np.ndarray:
  ```

---

### `plot_compare(mov, ref, lab1=None, lab2=None, cmap1='inferno', cmap2=False, axis='off')`
- **Purpose:** Plots two images side-by-side for comparison.
- **Parameters:**
    - `mov` (np.ndarray): First image.
    - `ref` (np.ndarray): Second image.
    - `lab1` (str, optional): Title for the first image. Defaults to `None`.
    - `lab2` (str, optional): Title for the second image. Defaults to `None`.
    - `cmap1` (str, optional): Colormap for the first image. Defaults to `'inferno'`.
    - `cmap2` (str | bool, optional): Colormap for the second image. If `False`, uses Matplotlib default. Defaults to `False`.
    - `axis` (str, optional): Matplotlib axis state ('on', 'off'). Defaults to `'off'`.
- **Returns:** None. Displays a plot.
- **Dependencies:** `matplotlib.pyplot`.
- **Function Signature:**
  ```python
  def plot_compare(mov: np.ndarray, ref: np.ndarray, lab1: str | None = None, lab2: str | None = None, cmap1: str = 'inferno', cmap2: str | bool = False, axis: str = 'off') -> None:
  ```

---

### `norm_im(im)`
- **Purpose:** Normalizes an image to the 0-255 range and converts to `np.uint8`.
- **Parameters:**
    - `im` (np.ndarray): Input image.
- **Returns:**
    - `np.ndarray`: Normalized uint8 image.
- **Function Signature:**
  ```python
  def norm_im(im: np.ndarray) -> np.ndarray:
  ```

---

### `colorise_out(im, cmap='inferno')`
- **Purpose:** Applies a colormap to a normalized version of an image and converts it to uint8 RGB.
- **Parameters:**
    - `im` (np.ndarray): Input grayscale image.
    - `cmap` (str, optional): Name of the Matplotlib colormap. Defaults to `'inferno'`.
- **Returns:**
    - `np.ndarray`: Colorized uint8 RGB image.
- **Dependencies:** `matplotlib.pyplot.get_cmap`, `norm_im`.
- **Function Signature:**
  ```python
  def colorise_out(im: np.ndarray, cmap: str = 'inferno') -> np.ndarray:
  ```

---

### `myround(x, base=5)`
- **Purpose:** Rounds a number `x` to the nearest multiple of `base` by flooring.
- **Parameters:**
    - `x` (float | np.ndarray): Number(s) to round.
    - `base` (int, optional): The base to round to. Defaults to `5`.
- **Returns:**
    - `float | np.ndarray`: The rounded number(s).
- **Function Signature:**
  ```python
  def myround(x: float | np.ndarray, base: int = 5) -> float | np.ndarray:
  ```

---

### `bin_bin(binary, he_um_px, iq_um_px, method='ndarray', op='sum')`
- **Purpose:** Re-bins a binary image (presumably from H&E resolution) to iQID resolution. It calculates target dimensions based on pixel sizes, crops the input binary image, and then re-bins using either NumPy reshaping (`ndarray`) or OpenCV resizing (`cv2_nn`).
- **Parameters:**
    - `binary` (np.ndarray): The input binary image (H&E scale).
    - `he_um_px` (float): Pixel size of the H&E image (microns per pixel).
    - `iq_um_px` (float): Pixel size of the iQID image (microns per pixel).
    - `method` (str, optional): Re-binning method. `'ndarray'` for `helper.bin_ndarray`, `'cv2_nn'` for `cv2.resize` with nearest-neighbor. Defaults to `'ndarray'`.
    - `op` (str, optional): Operation for `helper.bin_ndarray` if `method='ndarray'`. Defaults to `'sum'`.
- **Returns:**
    - `np.ndarray`: The re-binned image at iQID resolution.
- **Dependencies:** `numpy`, `crop_to`, `myround`, `iqid.helper.bin_ndarray`, `cv2.resize`.
- **Function Signature:**
  ```python
  def bin_bin(binary: np.ndarray, he_um_px: float, iq_um_px: float, method: str = 'ndarray', op: str = 'sum') -> np.ndarray:
  ```

---

### `bin_centroids(fileName, imsize, he_um_px, iq_um_px, minA=1)`
- **Purpose:** Reads centroid coordinates (X, Y) and area (A) from a CSV file, filters them by minimum area, scales them based on H&E to iQID pixel size ratio, and generates a 2D histogram (binned image) of these centroids.
- **Parameters:**
    - `fileName` (str): Path to the CSV file containing centroid data (expected columns: Area, X, Y).
    - `imsize` (tuple[int, int]): `(height, width)` of the original H&E image space.
    - `he_um_px` (float): Pixel size of H&E image (microns/pixel).
    - `iq_um_px` (float): Pixel size of iQID image (microns/pixel).
    - `minA` (float, optional): Minimum area threshold for centroids. Defaults to `1`.
- **Returns:**
    - `np.ndarray`: A 2D histogram representing the binned centroids at iQID resolution.
- **Dependencies:** `numpy`, `matplotlib.pyplot.hist2d`.
- **Function Signature:**
  ```python
  def bin_centroids(fileName: str, imsize: tuple[int, int], he_um_px: float, iq_um_px: float, minA: float = 1) -> np.ndarray:
  ```

```
