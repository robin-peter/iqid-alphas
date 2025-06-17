# `iqid.helper` Module Documentation

This module provides various helper functions for the iQID processing pipeline, including file system operations, string manipulation for natural sorting, numerical calculations, date/time utilities, array manipulations, and plotting utilities.

---

## Functions

### `list_studies(rootdir)`
- **Purpose:** Gets a list of directory paths for all folders directly within the specified `rootdir`.
- **Parameters:**
    - `rootdir` (str): The path to the root directory to scan.
- **Returns:**
    - `study_list` (list[str]): A list of full directory paths for folders in `rootdir`.
- **Function Signature:**
  ```python
  def list_studies(rootdir: str) -> list[str]:
  ```

---

### `list_substudies(rootdir)`
- **Purpose:** Gets a list of directory paths for all subfolders one level down from the `rootdir` (i.e., subfolders of folders within `rootdir`).
- **Parameters:**
    - `rootdir` (str): The path to the root directory.
- **Returns:**
    - `substudy_list` (list[str]): A list of full directory paths for subfolders.
- **Function Signature:**
  ```python
  def list_substudies(rootdir: str) -> list[str]:
  ```

---

### `organize_dirs(rootdir, study_list, copy_files=True, copy_meta=True)`
- **Purpose:** Organizes a set of subdirectories by creating a new directory structure under `../analysis/` relative to `rootdir`. It then copies files (e.g., `.dat` listmode files) and metadata files (e.g., `Acquisition_Info*.txt`) into appropriately named new subdirectories.
- **Parameters:**
    - `rootdir` (str): The root directory from which the analysis structure will be created (one level up).
    - `study_list` (list[str]): List of identified directory paths containing the original data.
    - `copy_files` (bool, optional): If `True` (default), copies the primary data files (e.g., `.dat`).
    - `copy_meta` (bool, optional): If `True` (default), copies metadata files.
- **Returns:**
    - None. Modifies the file system.
- **Function Signature:**
  ```python
  def organize_dirs(rootdir: str, study_list: list[str], copy_files: bool = True, copy_meta: bool = True) -> None:
  ```
- **Note:** This function uses `shutil.copy2` for copying. The new directory names are constructed from parts of the original directory paths. It relies on a specific structure like `dayname/seqname/name/Listmode/*.dat`.

---

### `atoi(text)`
- **Purpose:** Helper function for `natural_keys`. Converts text to an integer if it's a digit, otherwise returns the text as is.
- **Parameters:**
    - `text` (str): A segment of a string.
- **Returns:**
    - `int | str`: The integer representation if `text` is a digit, otherwise `text`.
- **Function Signature:**
  ```python
  def atoi(text: str) -> int | str:
  ```

---

### `natural_keys(text)`
- **Purpose:** Generates a list of string and number parts from a given text string for natural sort order. Used as a key for sorting functions.
- **Parameters:**
    - `text` (str): The string to be split into parts for natural sorting.
- **Returns:**
    - `list[int | str]`: A list where numbers are converted to integers and text parts remain strings.
- **Function Signature:**
  ```python
  def natural_keys(text: str) -> list[int | str]:
  ```

---

### `natural_sort(l)`
- **Purpose:** Sorts a list of strings in natural order (e.g., "item2" comes before "item10").
- **Parameters:**
    - `l` (list[str]): The list of strings to sort.
- **Returns:**
    - `list[str]`: The naturally sorted list.
- **Function Signature:**
  ```python
  def natural_sort(l: list[str]) -> list[str]:
  ```

---

### `mean_stdev(array, return_vals=False)`
- **Purpose:** Prints the mean and standard deviation of a NumPy array, formatted to two significant figures. Optionally returns these values.
- **Parameters:**
    - `array` (np.ndarray): The input NumPy array.
    - `return_vals` (bool, optional): If `True`, returns the mean and standard deviation. Defaults to `False`.
- **Returns:**
    - `tuple[float, float]` | `str`: If `return_vals` is `True`, returns `(mean, std_dev)`. Otherwise, returns an empty string.
- **Function Signature:**
  ```python
  def mean_stdev(array: np.ndarray, return_vals: bool = False) -> tuple[float, float] | str:
  ```

---

### `pct_err(ref, exp)`
- **Purpose:** Calculates the percentage error between a reference value and an experimental value.
- **Parameters:**
    - `ref` (float | np.ndarray): The reference value(s).
    - `exp` (float | np.ndarray): The experimental value(s).
- **Returns:**
    - `float | np.ndarray`: The percentage error: `100 * (ref - exp) / ref`.
- **Function Signature:**
  ```python
  def pct_err(ref: float | np.ndarray, exp: float | np.ndarray) -> float | np.ndarray:
  ```

---

### `get_yn(string)`
- **Purpose:** Converts a user input string (yes/no, y/n) to a boolean value.
- **Parameters:**
    - `string` (str): The user input string.
- **Returns:**
    - `bool` | `None`: `True` for "yes" or "y", `False` for "no" or "n". Prints an error and implicitly returns `None` for invalid input.
- **Function Signature:**
  ```python
  def get_yn(string: str) -> bool | None:
  ```

---

### `get_dt(t1, t2, grain='us', verbose=False)`
- **Purpose:** Calculates the absolute time difference between two `datetime.datetime` objects.
- **Parameters:**
    - `t1` (datetime.datetime): The first datetime object.
    - `t2` (datetime.datetime): The second datetime object.
    - `grain` (str, optional): The temporal resolution of the result, either 'us' (microseconds, default) or 's' (total seconds).
    - `verbose` (bool, optional): Unused parameter (as of current code). Defaults to `False`.
- **Returns:**
    - `float`: The time difference in the specified `grain`. Returns `None` implicitly if `grain` is invalid.
- **Function Signature:**
  ```python
  def get_dt(t1: datetime.datetime, t2: datetime.datetime, grain: str = 'us', verbose: bool = False) -> float | None:
  ```

---

### `check_mono(arr)`
- **Purpose:** Checks if a NumPy array is monotonically increasing (i.e., each element is greater than or equal to the previous one).
- **Parameters:**
    - `arr` (np.ndarray): The input NumPy array.
- **Returns:**
    - `bool`: `True` if the array is monotonically increasing, `False` otherwise.
- **Function Signature:**
  ```python
  def check_mono(arr: np.ndarray) -> bool:
  ```

---

### `bin_ndarray(ndarray, new_shape, operation='sum')`
- **Purpose:** Bins/resamples an ndarray to a `new_shape` by summing or averaging values in the bins. The dimensions of `new_shape` must be factors of the original dimensions.
- **Parameters:**
    - `ndarray` (np.ndarray): The input NumPy array.
    - `new_shape` (tuple[int, ...]): The target shape for the binned array.
    - `operation` (str, optional): The operation to perform ('sum' or 'mean'). Defaults to 'sum'.
- **Returns:**
    - `np.ndarray`: The binned/resampled array.
- **Raises:**
    - `ValueError`: If `operation` is not 'sum' or 'mean', or if the number of dimensions in `ndarray.shape` and `new_shape` do not match.
- **Function Signature:**
  ```python
  def bin_ndarray(ndarray: np.ndarray, new_shape: tuple[int, ...], operation: str = 'sum') -> np.ndarray:
  ```

---

### `decompose_affine(A)`
- **Purpose:** Decomposes a 3x3 affine transformation matrix `A` into its constituent parts: translation (T), rotation (R), scaling/zoom (Z), and shear (S).
- **Parameters:**
    - `A` (np.ndarray): A 3x3 affine transformation matrix.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: A tuple containing (T, R, Z, S).
        - T (np.ndarray): Translation vector (2 elements).
        - R (np.ndarray): Rotation matrix (2x2).
        - Z (np.ndarray): Scaling/zoom factors (2 elements).
        - S (np.ndarray): Shear factor(s) (1 element for 2D shear from 3x3 matrix context).
- **Function Signature:**
  ```python
  def decompose_affine(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  ```

---

### `set_plot_parms()`
- **Purpose:** Sets standard Matplotlib parameters for plots to ensure a consistent style (font, line width, grid, legend).
- **Parameters:**
    - None.
- **Returns:**
    - `matplotlib.colors.Colormap`: The 'tab10' colormap object.
- **Function Signature:**
  ```python
  def set_plot_parms() -> plt.cm.ScalarMappable: # Actually returns a Colormap, but ScalarMappable is a common parent.
  ```

---

### `draw_lines(es, label=None, ls='--', **kwargs)`
- **Purpose:** Draws vertical dashed lines on the current Matplotlib plot axis at specified x-values.
- **Parameters:**
    - `es` (list[float]): A list of x-values (e.g., energies) where lines should be drawn.
    - `label` (str, optional): A label for the set of lines (will create a legend entry with no visible line if used). Defaults to `None`.
    - `ls` (str, optional): Linestyle for the lines. Defaults to '--'.
    - `**kwargs`: Additional keyword arguments passed to `plt.axvline` and `plt.plot` (for the label).
- **Returns:**
    - None. Modifies the current Matplotlib plot.
- **Function Signature:**
  ```python
  def draw_lines(es: list[float], label: str | None = None, ls: str = '--', **kwargs) -> None:
  ```

---

### `add_scalebar(scalebarlen, ax, px_mm=1/(80/2048), loc='lower right', **kwargs)`
- **Purpose:** Adds a formatted scale bar to a Matplotlib axes object.
- **Parameters:**
    - `scalebarlen` (float): The physical length of the scale bar in millimeters.
    - `ax` (matplotlib.axes.Axes): The Matplotlib axes object to add the scale bar to.
    - `px_mm` (float, optional): Pixels per millimeter for the image space. Defaults to `1/(80/2048)` which is `2048/80 = 25.6`.
    - `loc` (str, optional): Location of the scale bar (e.g., 'lower right'). Defaults to 'lower right'.
    - `**kwargs`: Additional keyword arguments passed to `AnchoredSizeBar`.
- **Returns:**
    - None. Modifies the provided `ax` object.
- **Function Signature:**
  ```python
  def add_scalebar(scalebarlen: float, ax: plt.Axes, px_mm: float = 25.6, loc: str = 'lower right', **kwargs) -> None:
  ```

---

### `nice_colorbar(mappable, cbar_location="right", orientation="vertical", pad=0.05)`
- **Purpose:** Creates a nicely formatted colorbar for a Matplotlib mappable object (e.g., an image).
- **Parameters:**
    - `mappable` (matplotlib.cm.ScalarMappable): The artist (e.g., output of `imshow`) to which the colorbar applies.
    - `cbar_location` (str, optional): Position of the colorbar relative to the axes. Defaults to "right".
    - `orientation` (str, optional): Orientation of the colorbar. Defaults to "vertical".
    - `pad` (float, optional): Padding between the axes and the colorbar. Defaults to 0.05.
- **Returns:**
    - `matplotlib.colorbar.Colorbar`: The created colorbar object.
- **Function Signature:**
  ```python
  def nice_colorbar(mappable: plt.cm.ScalarMappable, cbar_location: str = "right", orientation: str = "vertical", pad: float = 0.05) -> plt.colorbar.Colorbar:
  ```

---

### `compare_metrics(ref, mov)`
- **Purpose:** Compares two images (`ref` and `mov`) by printing mean and max percentage differences, and showing visual comparisons using `plot_compare()` and histogram comparisons using `plot_hists()`.
- **Parameters:**
    - `ref` (np.ndarray): The reference image.
    - `mov` (np.ndarray): The moving/comparison image.
- **Returns:**
    - None. Prints output and displays plots.
- **Function Signature:**
  ```python
  def compare_metrics(ref: np.ndarray, mov: np.ndarray) -> None:
  ```

---

### `plot_hists(slice1, slice2)`
- **Purpose:** Plots and compares the pixel intensity histograms of two image slices.
- **Parameters:**
    - `slice1` (np.ndarray): The first image slice.
    - `slice2` (np.ndarray): The second image slice.
- **Returns:**
    - None. Displays a Matplotlib plot.
- **Function Signature:**
  ```python
  def plot_hists(slice1: np.ndarray, slice2: np.ndarray) -> None:
  ```

---

### `plot_compare(slice1, slice2)`
- **Purpose:** Displays two image slices side-by-side for visual comparison, with a shared colorbar.
- **Parameters:**
    - `slice1` (np.ndarray): The first image slice (reference).
    - `slice2` (np.ndarray): The second image slice (comparison).
- **Returns:**
    - None. Displays a Matplotlib plot.
- **Function Signature:**
  ```python
  def plot_compare(slice1: np.ndarray, slice2: np.ndarray) -> None:
  ```

---

### `plot_sequence(imstack, nrows=2, figsize=(18, 10), balance=False)`
- **Purpose:** Plots a sequence of images from an image stack in a grid.
- **Parameters:**
    - `imstack` (np.ndarray): A 3D NumPy array representing an image stack (num_images, height, width).
    - `nrows` (int, optional): Number of rows in the plot grid. Defaults to 2.
    - `figsize` (tuple[int, int], optional): Figure size. Defaults to (18, 10).
    - `balance` (bool, optional): If `True`, uses a shared `vmax` for all images based on the global maximum of the stack. Defaults to `False`.
- **Returns:**
    - None. Displays a Matplotlib plot.
- **Function Signature:**
  ```python
  def plot_sequence(imstack: np.ndarray, nrows: int = 2, figsize: tuple[int, int] = (18, 10), balance: bool = False) -> None:
  ```

---
### `string_underscores(string_list)`
- **Purpose:** Concatenates a list of strings with underscores. (Note: This function was commented out as "DEPRECATED" in the provided source but is documented here as it was parsed.)
- **Parameters:**
    - `string_list` (list[str]): A list of strings.
- **Returns:**
    - `str`: A single string with elements joined by underscores.
- **Function Signature (if it were active):**
  ```python
  # def string_underscores(string_list: list[str]) -> str:
  # newstring = ''
  # for i in range(len(string_list)):
  # newstring = newstring + string_list[i] + '_'
  # newstring = newstring[:-1] # remove last _
  # return (newstring)
  ```
- **Status:** Marked as DEPRECATED in source. The code for this function was commented out. The documentation reflects its original intent.

*(Note: Functions `plot_all` and `subplot_label` were also commented out in the source and are not included in this documentation as per typical practice for deprecated code unless specifically requested.)*

```
