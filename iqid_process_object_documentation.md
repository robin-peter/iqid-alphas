# `iqid.process_object` Module Documentation

This module defines classes and functions for importing, processing, and analyzing listmode data from the iQID camera. The primary class, `ClusterData`, provides a comprehensive suite of tools for handling various data formats, performing image reconstruction, ROI analysis, and temporal fitting.

---

## Standalone Functions

### `exponential(x, a, thalf)`
- **Purpose:** Defines an exponential decay function.
- **Parameters:**
    - `x` (float | np.ndarray): The independent variable (e.g., time).
    - `a` (float): The initial amplitude or activity.
    - `thalf` (float): The half-life of the decay.
- **Returns:**
    - `float | np.ndarray`: The value of the exponential function at `x`.
- **Function Signature:**
  ```python
  def exponential(x, a, thalf):
  ```

---

## Class: `ClusterData`

- **Overall Purpose:** This class is designed to load, process, and analyze iQID camera data. It handles different listmode file formats (`processed_lm`, `offset_lm`, `clusters`), extracts metadata, reconstructs images, allows for ROI definition and analysis, and performs temporal fitting of activity data.

### `__init__(self, file_name, c_area_thresh=15, makedir=False, ftype='processed_lm')`
- **Purpose:** Initializes a `ClusterData` object.
- **Parameters:**
    - `file_name` (str): Path to the iQID data file.
    - `c_area_thresh` (int, optional): Cluster area threshold. Defaults to `15`.
    - `makedir` (bool, optional): If `True`, creates an analysis subdirectory. Defaults to `False`.
    - `ftype` (str, optional): File type of the input data. Must be one of "processed_lm", "offset_lm", or "clusters". Defaults to `'processed_lm'`.
- **Instance Attributes Initialized:**
    - `self.file_name` (str): Stores `file_name`.
    - `self.ftype` (str): Stores `ftype`.
    - `self.savedir` (str): Path to a directory for saving analysis results, derived from `file_name`.
    - `self.c_area_thresh` (int): Stores `c_area_thresh`.
- **Raises:**
    - `TypeError`: If `ftype` is not one of the recognized values.

---

### Methods

#### `init_header(self)`
- **Purpose:** Reads and parses the header of the iQID data file to extract metadata.
- **Parameters:** None.
- **Returns:**
    - `tuple[int, int, int]`: A tuple containing `(HEADER_SIZE, XDIM, YDIM)`.
- **Side Effects:** Populates the following instance attributes:
    - `self.HEADER_SIZE` (int): Size of the header in bytes (from file, multiplied by 4).
    - `self.XDIM` (int): X dimension of the detector/image.
    - `self.YDIM` (int): Y dimension of the detector/image.
    - `self.NUM_DATA_ELEMENTS` (int): Number of data elements per cluster/event, depends on `self.ftype`.
    - If `self.ftype == 'clusters'`:
        - `self.cluster_radius` (int): Radius of the cluster images.
        - `self.cluster_imsize` (int): Dimension of the square cluster images (2 * radius + 1).
- **Raises:**
    - `TypeError`: If `self.ftype` is not recognized (though this should be caught in `__init__`).

---

#### `set_process_params(self, binfac, ROI_area_thresh, t_binsize, t_half)`
- **Purpose:** Sets parameters for subsequent processing steps.
- **Parameters:**
    - `binfac` (int): Binning factor for image processing.
    - `ROI_area_thresh` (int): Area threshold for ROI detection.
    - `t_binsize` (float | int): Time bin size for temporal analysis (in seconds).
    - `t_half` (float | int): Half-life for decay correction (in seconds).
- **Returns:**
    - `tuple`: An empty tuple.
- **Side Effects:** Populates instance attributes:
    - `self.binfac`
    - `self.ROI_area_thresh`
    - `self.t_binsize`
    - `self.t_half`

---

#### `load_cluster_data(self, event_fx=1, dtype=np.float64)`
- **Purpose:** Loads cluster data from the file specified in `self.file_name`.
- **Parameters:**
    - `event_fx` (float, optional): Fraction of total events/clusters to load. Defaults to `1` (load all).
    - `dtype` (np.dtype, optional): NumPy data type to load data as. Defaults to `np.float64`.
- **Returns:**
    - `np.ndarray`: A 2D NumPy array where columns are individual events/clusters and rows are data elements. Shape: (`self.NUM_DATA_ELEMENTS`, `NUM_LOAD`).
- **Side Effects:** Calls `self.init_header()` if not already called.

---

#### `load_raws(self, cluster_size=10)`
- **Purpose:** Loads raw listmode data from an associated "*Cropped_Raw_Listmode.dat" file, which is expected to be in the same directory as `self.file_name`.
- **Parameters:**
    - `cluster_size` (int, optional): This parameter seems unused in the current implementation of the method but might be intended for future use or is a remnant. Defaults to `10`.
- **Returns:**
    - `np.ndarray`: A 2D NumPy array containing the raw listmode data, similar in structure to `load_cluster_data` output but loaded as `np.int32`.
- **Raises:**
    - `Exception`: If the associated raw listmode file is not found.
- **Side Effects:** Uses `self.HEADER_SIZE` and `self.NUM_DATA_ELEMENTS` (which should be initialized by `init_header` typically called via `load_cluster_data` or directly).

---

#### `init_metadata(self, data)`
- **Purpose:** Parses the loaded raw data array into meaningful instance attributes based on `self.ftype`.
- **Parameters:**
    - `data` (np.ndarray): The raw data array loaded by `load_cluster_data` or `load_raws`.
- **Returns:**
    - `tuple`: A tuple of NumPy arrays containing key metadata. The content of the tuple varies based on `self.ftype`:
        - `processed_lm`: `(time_ms, cluster_area, xC_global, yC_global, frame_num)`
        - `offset_lm`: `(frame_num, time_ms, n_miss, n_cluster, pix, cam_temp_10K)`
        - `clusters`: `(frame_num, time_ms, xC, yC, raw_imgs, cim_sum, cim_px)`
- **Side Effects:** Populates many instance attributes depending on `self.ftype`, including:
    - Common: `self.f` (frame numbers), `self.time_ms` (timestamps).
    - `processed_lm`: `self.cluster_area`, `self.xC`, `self.yC`, `self.miss = None`, `self.nevents_per_frame = None`, `self.offset_frame_time = None`.
    - `offset_lm`: `self.miss` (missed events), `self.nevents_per_frame`, `self.offset_frame_time`.
    - `clusters`: `self.xC`, `self.yC`, `self.cim_sum` (cluster sum), `self.cim_px` (cluster pixels/area), `self.raws` (raw cluster images).

---

#### `image_from_xy(self, x, y)`
- **Purpose:** Creates a 2D histogram (image) from x and y coordinates.
- **Parameters:**
    - `x` (np.ndarray): Array of x-coordinates.
    - `y` (np.ndarray): Array of y-coordinates.
- **Returns:**
    - `np.ndarray`: A 2D NumPy array representing the image, with dimensions `(self.YDIM, self.XDIM)`.
- **Side Effects:** Requires `self.XDIM` and `self.YDIM` to be initialized.

---

#### `image_from_listmode(self, subpx=1)`
- **Purpose:** Reconstructs a spatial image from listmode data, applying area threshold and optional subpixel positioning.
- **Parameters:**
    - `subpx` (int, optional): Subpixel factor. If `>1`, coordinates are scaled by this factor before flooring. Defaults to `1`.
- **Returns:**
    - `np.ndarray`: The reconstructed cluster image. Dimensions depend on `subpx`, `self.XDIM`, `self.YDIM`.
- **Side Effects:**
    - Calls `self.init_header()`, `self.load_cluster_data()`, `self.init_metadata()`.
    - Updates `self.xC`, `self.yC`, and `self.t_s` with the filtered, (optionally) subpixelated coordinates and corresponding timestamps (in seconds) of events used in the image.

---

#### `image_from_big_listmode(self, event_fx=0.1, xlim=(0, None), ylim=(0, None))`
- **Purpose:** Efficiently reconstructs an image from a potentially large listmode dataset, allowing for loading a fraction of events and spatial cropping. Only supports `subpx=1`.
- **Parameters:**
    - `event_fx` (float, optional): Fraction of events to load. Defaults to `0.1`.
    - `xlim` (tuple[int, int | None], optional): (min_x, max_x) crop limits. `None` for max_x means `self.XDIM`. Defaults to `(0, None)`.
    - `ylim` (tuple[int, int | None], optional): (min_y, max_y) crop limits. `None` for max_y means `self.YDIM`. Defaults to `(0, None)`.
- **Returns:**
    - `np.ndarray`: The reconstructed cluster image of dimensions `(self.YDIM, self.XDIM)`.
- **Side Effects:**
    - Calls `self.init_header()`, `self.load_cluster_data()`, `self.init_metadata()`.
    - Updates `self.xC`, `self.yC`, and `self.t_s` with the filtered and cropped coordinates and corresponding timestamps (in seconds).
- **Raises:**
    - `ValueError`: If `xlim` or `ylim` exceed acquisition dimensions.

---

#### `apply_selection(self, selection_bool)`
- **Purpose:** Filters instance attributes `self.xC`, `self.yC`, and `self.f` in place based on a boolean array.
- **Parameters:**
    - `selection_bool` (np.ndarray[bool]): Boolean array of the same length as `self.xC`, `self.yC`, `self.f`.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray]`: The filtered `self.xC`, `self.yC`, `self.f`.
- **Side Effects:** Modifies `self.xC`, `self.yC`, `self.f` in place.

---

#### `get_subset(self, selection_bool)`
- **Purpose:** Creates a new `Subset` object containing a subset of the data based on `selection_bool`.
- **Parameters:**
    - `selection_bool` (np.ndarray[bool]): Boolean array to filter the data.
- **Returns:**
    - `Subset`: A new `Subset` object with filtered data.
- **Side Effects:** Initializes attributes of the `Subset` object (`HEADER_SIZE`, `XDIM`, `YDIM`, `xC`, `yC`, `f`, `time_ms`, and `clusters`-specific attributes if applicable).

---

#### `get_mean_n(self, vis=False)`
- **Purpose:** Calculates and optionally visualizes the mean number of events per frame using data from an 'offset_lm' file.
- **Parameters:**
    - `vis` (bool, optional): If `True`, displays a histogram of events per frame. Defaults to `False`.
- **Returns:**
    - `float` | `None`: The mean number of events per frame. Returns `None` and prints a message if offset data (`self.nevents_per_frame`) is not loaded.
- **Side Effects:** Sets `self.mean_n`.

---

#### `estimate_missed_timestamps(self)`
- **Purpose:** Estimates timestamps for missed events using data from an 'offset_lm' file.
- **Parameters:** None.
- **Returns:**
    - `np.ndarray` | `None`: An array of estimated timestamps for missed events. Returns `None` and prints a message if offset data (`self.miss`, `self.offset_frame_time`) is not loaded.

---

#### `filter_singles(self, fmax, vis=False)`
- **Purpose:** Filters the event data to keep only events from frames that contain a single event.
- **Parameters:**
    - `fmax` (int): The maximum frame number in the acquisition, used for calculating percentages.
    - `vis` (bool, optional): If `True`, displays a histogram of events per frame and statistics about single/empty frames. Defaults to `False`.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray]`: The filtered `self.xC`, `self.yC`, `self.f`.
- **Side Effects:** Modifies `self.xC`, `self.yC`, `self.f` in place to only include single events.

---

#### `set_coin_params(self, fps, t0_dt, TS_ROI, binfac=1, verbose=True)`
- **Purpose:** Sets parameters for coincidence analysis between iQID events and external (e.g., IDM) timestamps.
- **Parameters:**
    - `fps` (float): Frames per second of the iQID camera.
    - `t0_dt` (float): Time offset (in seconds) to be added to `TS_ROI` (e.g., if IDM started after iQID, this is positive).
    - `TS_ROI` (np.ndarray): Timestamps from the external device (e.g., IDM) in seconds.
    - `binfac` (int, optional): Binning factor for time bins. Defaults to `1`.
    - `verbose` (bool, optional): If `True`, prints parameter info. Defaults to `True`.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray]`: `(a, b, s_bins)` where `a` are iQID event times, `b` are adjusted external event times, and `s_bins` are the time bins for histogramming.
- **Side Effects:** Sets instance attributes `self.fps`, `self.exp` (exposure time per frame), `self.a`, `self.b`, `self.s_bins`.
- **Raises:**
    - `AssertionError`: If iQID times (`a`) or IDM times (`b`) are not monotonically increasing.

---

#### `find_coin(self, singles=False, return_hist=False, verbose=True)`
- **Purpose:** Finds coincident events by histogramming iQID and external (IDM) event times and identifying shared time bins.
- **Parameters:**
    - `singles` (str | bool, optional): If `'iqid'`, `'idm'`, or `'both'`, filters for bins with only one event from the respective source(s). Defaults to `False` (no single-event filtering).
    - `return_hist` (bool, optional): If `True`, also returns the iQID and IDM histograms. Defaults to `False`.
    - `verbose` (bool, optional): If `True`, prints progress and results. Defaults to `True`.
- **Returns:**
    - `np.ndarray[bool]` or `tuple[np.ndarray[bool], np.ndarray, np.ndarray]`:
        - `coin`: A boolean array indicating coincident time bins.
        - If `return_hist` is `True`, also returns `iq_n` (iQID histogram) and `idm_n` (IDM histogram).
- **Side Effects:** Uses `self.a`, `self.b`, `self.s_bins` set by `set_coin_params`.

---

#### `image_from_coin(self, coin=None, verbose=True, binfac=1, **kwargs)`
- **Purpose:** Reconstructs an image using only the iQID events that are coincident with external events.
- **Parameters:**
    - `coin` (np.ndarray[bool], optional): Boolean array of coincident time bins from `find_coin`. If `None`, `find_coin` is called internally.
    - `verbose` (bool, optional): If `True`, prints progress. Defaults to `True`.
    - `binfac` (int, optional): Binning factor used for frame numbers, should match `set_coin_params`. Defaults to `1`.
    - `**kwargs`: Additional keyword arguments passed to `find_coin` if `coin` is `None`.
- **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray]`: `(x_good, y_good, nim)` where `x_good`, `y_good` are coordinates of coincident events, and `nim` is the reconstructed image of coincident events.
- **Side Effects:** Uses `self.f`, `self.xC`, `self.yC`, `self.XDIM`, `self.YDIM`.

---

#### `check_elements(self, a, idx, x)`
- **Purpose:** Recursive helper function to find the index of element `x` in array `a`, starting search from `idx`.
- **Parameters:**
    - `a` (np.ndarray): Array to search.
    - `idx` (int): Starting index for search.
    - `x`: Value to find.
- **Returns:**
    - `int`: Index of `x` in `a` at or after `idx`.
- **Raises:**
    - `IndexError` (implicitly via recursion limit if `x` not found and `idx` goes out of bounds).

---

#### `correct_frames(self, a1, a2, m1)`
- **Purpose:** Assigns missed frame counts (`m1` from offset file frames `a1`) to the corresponding frames in the full listmode data (`a2`).
- **Parameters:**
    - `a1` (np.ndarray): Frame numbers from the offset file.
    - `a2` (np.ndarray): Frame numbers from the listmode file.
    - `m1` (np.ndarray): Missed frame counts corresponding to `a1`.
- **Returns:**
    - `np.ndarray`: `m2`, an array of missed frame counts corresponding to `a2`.
- **Side Effects:** Prints a message and fills remaining `m2` values if listmode frames exceed offset frames.

---

#### `correct_listmode(self, offset_frames, missed_frames, vis=True)`
- **Purpose:** Corrects the frame numbers in `self.f` (listmode frames) by adding the number of missed frames, derived from an offset file.
- **Parameters:**
    - `offset_frames` (np.ndarray): Frame numbers from the offset file.
    - `missed_frames` (np.ndarray): Missed frame counts from the offset file.
    - `vis` (bool, optional): If `True`, plots a comparison of missed frames before and after alignment. Defaults to `True`.
- **Returns:**
    - `np.ndarray`: The corrected listmode frame numbers.
- **Side Effects:** Modifies `self.f` in place with the corrected frame numbers.

---

#### `set_contour_params(self, gauss=15, thresh=0)`
- **Purpose:** Sets parameters for contour detection.
- **Parameters:**
    - `gauss` (int, optional): Gaussian blur kernel size. Defaults to `15`.
    - `thresh` (float | int, optional): Threshold value for binarizing the image before contour detection. Defaults to `0`.
- **Returns:**
    - `tuple`: An empty tuple.
- **Side Effects:** Sets instance attributes `self.gauss` and `self.thresh`.

---

#### `prep_contour(self, im, gauss=15, thresh=0)`
- **Purpose:** Prepares an image for contour detection by binning, thresholding, and Gaussian blurring.
- **Parameters:**
    - `im` (np.ndarray): The input image.
    - `gauss` (int, optional): Gaussian blur kernel size. Uses `self.gauss` if not provided, but currently shadows it. Defaults to `15`.
    - `thresh` (float | int, optional): Threshold value. Uses `self.thresh` if not provided, but currently shadows it. Defaults to `0`.
- **Returns:**
    - `np.ndarray`: The prepared image (blurred binary mask).
- **Side Effects:** Uses `self.binfac`.

---

#### `get_contours(self, im)`
- **Purpose:** Finds contours in an image after preparing it.
- **Parameters:**
    - `im` (np.ndarray): The input image.
- **Returns:**
    - `list[np.ndarray]`: A list of "good" contours, where each contour is scaled by `self.binfac` and filtered by `self.ROI_area_thresh`.
- **Side Effects:** Uses `self.gauss`, `self.thresh`, `self.binfac`, `self.ROI_area_thresh`.

---

#### `get_contours_from_dir(self, mask_dir, fformat='png')`
- **Purpose:** Loads contours from pre-existing manual mask image files in a directory.
- **Parameters:**
    - `mask_dir` (str): Directory containing mask image files.
    - `fformat` (str, optional): File format of the mask images. Defaults to `'png'`.
- **Returns:**
    - `list[np.ndarray]`: A list of contours, scaled by `self.binfac` and filtered by `self.ROI_area_thresh`.
- **Side Effects:** Uses `self.binfac`, `self.ROI_area_thresh`.

---

#### `get_maskstack(self, im)`
- **Purpose:** Creates a stack of binary masks from the instance's `self.contours`.
- **Parameters:**
    - `im` (np.ndarray): An image whose dimensions are used for the mask stack.
- **Returns:**
    - `np.ndarray`: A 3D array where each slice `[i, :, :]` is a binary mask for `self.contours[i]`.
- **Side Effects:** Requires `self.contours` to be set.

---

#### `events_in_ROI(self, maskstack)`
- **Purpose:** Determines which events fall within each ROI defined by a mask stack.
- **Parameters:**
    - `maskstack` (np.ndarray): A 3D array of ROI masks.
- **Returns:**
    - `np.ndarray[bool]`: A 2D boolean array (`num_masks`, `num_events`) indicating if event `j` is in mask `i`.
- **Side Effects:** Uses `self.xC`, `self.yC`, `self.t_s`.

---

#### `get_ROIs(self, pad=10)`
- **Purpose:** Calculates bounding boxes for each contour in `self.contours`, with optional padding.
- **Parameters:**
    - `pad` (int, optional): Padding to add around the bounding box (half on each side). Defaults to `10`.
- **Returns:**
    - `np.ndarray`: A 2D array (`num_contours`, 4) where each row is `[x, y, w, h]` for a bounding box.
- **Side Effects:** Requires `self.contours` to be set.

---

#### `setup_ROIs(self, im, mode='auto', **kwargs)`
- **Purpose:** A convenience function to set up ROIs either automatically from an image or manually from a directory of masks.
- **Parameters:**
    - `im` (np.ndarray): The image to use for automatic ROI detection (if `mode='auto'`).
    - `mode` (str, optional): `'auto'` or `'manual'`. Defaults to `'auto'`.
    - `**kwargs`: Keyword arguments passed to `get_contours_from_dir` if `mode='manual'`.
- **Returns:** None.
- **Side Effects:** Populates `self.contours`, `self.maskstack`, `self.ROIbool` (boolean array of events in ROIs), and `self.ROIlists` (bounding boxes).
- **Raises:**
    - `TypeError`: If `mode` is not 'auto' or 'manual'.

---

#### `fitHist(self, t, n, func=exponential, p0=[1, 9.92*24*3600], tol=0.05)`
- **Purpose:** Fits a function (default: exponential) to histogram data (time `t`, counts `n`).
- **Parameters:**
    - `t` (np.ndarray): Time points (bin centers).
    - `n` (np.ndarray): Counts in bins.
    - `func` (callable, optional): Function to fit. Defaults to `exponential`.
    - `p0` (list, optional): Initial guess for function parameters. Defaults to `[1, 9.92*24*3600]` (approx 9.92 days half-life).
    - `tol` (float, optional): Tolerance for parameter bounds (relative to `p0[1]`). Defaults to `0.05`.
- **Returns:**
    - `tuple`: `(popt, pcov, param_std, res, chisq, chisqn)`
        - `popt`: Optimal parameters.
        - `pcov`: Estimated covariance of `popt`.
        - `param_std`: Standard deviation of parameters.
        - `res`: Residuals.
        - `chisq`: Chi-squared value.
        - `chisqn`: Reduced chi-squared value.

---

#### `fitROI(self, temporal_array, func=exponential, p0=None, binsize=1000, tol=0.05)`
- **Purpose:** Fits a decay curve to the temporal data of events within an ROI.
- **Parameters:**
    - `temporal_array` (np.ndarray): Array of timestamps for events in an ROI.
    - `func` (callable, optional): Function to fit. Defaults to `exponential`.
    - `p0` (list, optional): Initial guess for parameters. If `None`, uses `[1, self.t_half]`.
    - `binsize` (int | float, optional): Time bin size for histogramming `temporal_array`. Defaults to `1000` (seconds).
    - `tol` (float, optional): Tolerance for fitting bounds. Defaults to `0.05`.
- **Returns:**
    - `tuple`: `(count, timepoints, popt, pcov, param_std, res, chisq, chisqn)` similar to `fitHist`.
- **Side Effects:** Uses `self.t_half`.

---

#### `get_imslice(self, im, idx)`
- **Purpose:** Extracts a masked image slice corresponding to a specific ROI's bounding box.
- **Parameters:**
    - `im` (np.ndarray): The source image.
    - `idx` (int): Index of the ROI in `self.ROIlists` and `self.maskstack`.
- **Returns:**
    - `np.ndarray`: The image slice, masked by the ROI.
- **Side Effects:** Requires `self.ROIlists` and `self.maskstack` to be set.

---

#### `get_manual_maskstack(self)`
- **Purpose:** Loads manual masks from TIFF files in a 'manual_masks' subdirectory of `self.savedir`, processes them into contours, and updates ROI-related instance attributes.
- **Parameters:** None.
- **Returns:**
    - `np.ndarray`: The stack of loaded and processed manual masks.
- **Side Effects:** Populates/updates `self.ROIlists`, `self.ROIbool`, `self.contours`, `self.maskstack`.

---

#### `save_manual_mask(self, mask)`
- **Purpose:** Saves a manually created mask as both a PNG preview and a TIFF file.
- **Parameters:**
    - `mask` (np.ndarray): The mask image to save.
- **Returns:** None.
- **Side Effects:** Creates `self.savedir` if it doesn't exist. Saves "manual_mask_preview.png" and "manual_mask.tif" in `self.savedir`.

---

#### `fitROIs(self, im, vis=True, corr=0, tol=0.05, tstart=0, tcond=None, idxs='all', save=False, savemasks=False, save_ts=False)`
- **Purpose:** Performs activity fitting for multiple ROIs, calculates activity values (A0, Ai), and optionally visualizes and saves results.
- **Parameters:**
    - `im` (np.ndarray): The base image from which ROI slices are taken.
    - `vis` (bool, optional): If `True`, generates plots. Defaults to `True`.
    - `corr` (float, optional): Time correction (in seconds) to adjust activity to a reference time (e.g., time of sacrifice). Defaults to `0`.
    - `tol` (float, optional): Tolerance for fitting. Defaults to `0.05`.
    - `tstart` (float, optional): Time (in seconds) after which to start fitting data. Defaults to `0`.
    - `tcond` (tuple | None, optional): Time condition(s) for fitting (e.g., `(start1, stop1)` or `(start1, stop1, start2, stop2)`). Defaults to `None`.
    - `idxs` (str | list[int], optional): Indices of ROIs to process. If `'all'`, processes all ROIs. Defaults to `'all'`.
    - `save` (bool | str, optional): If `True` or `'slicer3d'`, saves resulting mBq images. Defaults to `False`.
    - `savemasks` (bool, optional): If `True`, saves ROI masks. Defaults to `False`.
    - `save_ts` (bool, optional): If `True`, saves timestamps for each ROI. Defaults to `False`.
- **Returns:**
    - `tuple`: `(all_A0, all_dA0)` if `vis` is `False`. `(all_A0, all_dA0, f, ax)` if `vis` is `True`.
        - `all_A0` (np.ndarray): Array of initial activities (A0 in Ci) for each ROI.
        - `all_dA0` (np.ndarray): Array of uncertainties for A0.
        - `f` (matplotlib.figure.Figure, optional): Figure object if `vis=True`.
        - `ax` (np.ndarray[matplotlib.axes.Axes], optional): Array of axes objects if `vis=True`.
- **Side Effects:**
    - Uses many instance attributes (`self.ROIlists`, `self.t_s`, `self.ROIbool`, `self.t_binsize`, `self.t_half`, `self.savedir`, `self.maskstack`).
    - Creates directories and saves files if `save`, `savemasks`, or `save_ts` are enabled.

---

#### `plot_vis_masks(self, im)`
- **Purpose:** Plots each mask from `self.maskstack` overlaid on the provided image `im`. (Appears to have an issue with `helper_im.tobytes()` for plotting with matplotlib `imshow`).
- **Parameters:**
    - `im` (np.ndarray): The background image.
- **Returns:** None. Displays plots.

---

#### `widget_labelling(self, im, vmax=1, deg=0, IMG_WIDTH=200, IMG_HEIGHT=200, COLS=4)`
- **Purpose:** Creates an IPython widget interface for interactively labeling and selecting/discarding ROIs based on their masks.
- **Parameters:**
    - `im` (np.ndarray): The image to display with masks.
    - `vmax` (float, optional): Multiplier for image maximum value for display scaling. Defaults to `1`.
    - `deg` (int, optional): Rotation degree for displaying images. Defaults to `0`.
    - `IMG_WIDTH` (int, optional): Width of each image in the widget. Defaults to `200`.
    - `IMG_HEIGHT` (int, optional): Height of each image in the widget. Defaults to `200`.
    - `COLS` (int, optional): Number of columns in the widget grid. Defaults to `4`.
- **Returns:** None. Displays an IPython widget.
- **Side Effects:** Requires `self.maskstack`. Interacts with IPython environment.

---

## Class: `Subset`

- **Overall Purpose:** Represents a subset of data derived from a `ClusterData` object. It inherits all methods from `ClusterData` and is typically created by `ClusterData.get_subset()`.
- **Inherits from:** `ClusterData`

### `__init__(self, file_name, c_area_thresh, makedir, ftype)`
- **Purpose:** Initializes a `Subset` object.
- **Parameters:**
    - `file_name` (str): Path to the original iQID data file.
    - `c_area_thresh` (int): Cluster area threshold.
    - `makedir` (bool): If `True`, creates an analysis subdirectory (typically `False` for subsets).
    - `ftype` (str): File type of the input data.
- **Side Effects:** Calls `super().__init__(...)` with the provided arguments. Instance attributes are typically populated by the `ClusterData.get_subset()` method after initialization.

```
