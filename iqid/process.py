import os
import numpy as np
import cv2
import re
from scipy.optimize import curve_fit

'''Functions for processing iQID (compressed, processed) listmode data into activity images.'''

def list_studies(rootdir):
    """Gets list of directory paths for all folders in the current dir."""
    study_list = [f.path for f in os.scandir(rootdir) if f.is_dir()]
    return(study_list)


def list_substudies(rootdir):
    """Gets list of directory paths for all subfolders one level down."""
    study_list = list_studies(rootdir)
    substudy_list = []
    for study in study_list:
        substudies = list_studies(study)
        substudy_list = substudy_list + substudies
    return(substudy_list)


def atoi(text):
    """Helper function for natural sort order."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Helper function for natural sort order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def load_cluster_data(file_name):
    """Reads an iQID listmode .dat binary file into numpy array.
    Data file must be Compressed Processed Listmode type (14-element data).

    Number of registered clusters N can be inferred from the file size.
    A cluster is equivalent to a detected alpha particle event.

    Parameters
    ----------
    file_name : str
        The name of the .dat listmode file

    Returns
    -------
    cluster_data : 2D array-like, 14 x N
        Cluster data matrix with 14 elements and N clusters.
        See elements in function delineate_cluster_filters.
    """

    file_size_bytes = os.path.getsize(file_name)
    HEADER_SIZE, *_ = load_header(file_name)
    NUM_DATA_ELEMENTS = 14
    NUM_CLUSTERS = np.floor(
        (file_size_bytes - 4*HEADER_SIZE) / (8*NUM_DATA_ELEMENTS))

    NAN_DATA = 50
    unshaped_data = np.fromfile(
        file_name, dtype='float', count=NAN_DATA + int(NUM_CLUSTERS*NUM_DATA_ELEMENTS))
    cluster_data = unshaped_data[NAN_DATA:].reshape(
        int(NUM_DATA_ELEMENTS), int(NUM_CLUSTERS), order='F')
    return(cluster_data)


def load_legacy_data(file_name):
    """Reads an iQID listmode .dat binary file into numpy array.
    Data file can be legacy type (uncompressed listmode).

    Parameters
    ----------
    file_name : str
        The name of the .dat listmode file

    Returns
    -------
    cluster_data : 2D array-like, 14 x N
        Cluster data matrix with 14 elements and N clusters.
        See elements in function delineate_cluster_filters.
    """

    file_size_bytes = os.path.getsize(file_name)
    HEADER_SIZE, *_ = load_header(file_name)
    NUM_DATA_ELEMENTS = 5
    NUM_FRAMES_PROCESSED = np.floor((file_size_bytes - 4*HEADER_SIZE)/ (NUM_DATA_ELEMENTS*4))  

    cluster_data = np.fromfile(
        file_name, dtype='int', count=int(NUM_FRAMES_PROCESSED*NUM_DATA_ELEMENTS))
    cluster_data = cluster_data.reshape(int(NUM_DATA_ELEMENTS), int(NUM_FRAMES_PROCESSED))
    return(cluster_data)


def load_header(file_name):
    """Reads only the header of an iQID listmode .dat binary file.

    Currently only reads certain elements. User may modify to add
    additional information: see iQID Acquisition Info for header elements.

    Parameters
    ----------
    file_name : str
        The name of the .dat listmode file

    Returns
    -------
    tuple : int
        Tuple of desired header elements, each type int.
    """
    HEADER = np.fromfile(file_name, dtype='int', count=100)
    HEADER_SIZE = HEADER[0]
    XDIM = HEADER[1]
    YDIM = HEADER[2]

    # e.g., additional options:
    # CCD_FRAMES = HEADER[5]
    # FRAMERATE = HEADER[16]
    # CCL_THRESHOLD = HEADER[21]
    return(HEADER_SIZE, XDIM, YDIM)  # , CCD_FRAMES, CCL_THRESHOLD, FRAMERATE)


def delineate_cluster_filters(cluster_data):
    """Splits a cluster data matrix into constituent elements.

    Parameters
    ----------
    cluster_data : 2D array-like, from load_cluster_data
        Cluster data matrix with 14 elements and N clusters.

    Returns
    -------
    tuple : arrays
        Tuple of arrays of each data type.
    """
    frame_num = cluster_data[0, :]
    time_ms = cluster_data[1, :]
    sum_cluster_signal = cluster_data[2, :]
    cluster_area = cluster_data[3, :]
    yC_global = cluster_data[4, :]
    xC_global = cluster_data[5, :]
    var_y = cluster_data[6, :]
    var_x = cluster_data[7, :]
    covar_xy = cluster_data[8, :]
    eccentricity = cluster_data[9, :]
    skew_y = cluster_data[10, :]
    skew_x = cluster_data[11, :]
    kurt_y = cluster_data[12, :]
    kurt_x = cluster_data[13, :]
    return(frame_num, time_ms, sum_cluster_signal, cluster_area,
           yC_global, xC_global, var_x, var_y, covar_xy,
           eccentricity, skew_y, skew_x, kurt_y, kurt_x)


def delineate_legacy_data(cluster_data):
    """Splits a legacy (uncompressed) data matrix into constituent elements.

    Parameters
    ----------
    cluster_data : 2D array-like, from load_legacy_data
        Cluster data matrix with 5 elements.

    Returns
    -------
    tuple : arrays
        Tuple of arrays of each data type.
    """
    frame_num = cluster_data[0,:]
    time_ms = cluster_data[1,:]
    missed_im = cluster_data[2,:]
    num_events_in_frame = cluster_data[3,:]
    total_px_above_threshold = cluster_data[4,:]
    return(frame_num, time_ms, missed_im, 
        num_events_in_frame, total_px_above_threshold)


def pixelize_image(cluster_data, XDIM, YDIM, cluster_area=15, subpx=1):
    """Gets x-, y-, and t-coordinates cluster data matrix to return spatial image
    and associated x-, y-, and t- arrays.

    Only selects events with minimum scintillation light area.
    Events may be interpolated to increase resoltution by a factor,
    at the expense of sensitivity.

    Parameters
    ----------
    cluster_data : 2D array-like, from load_cluster_data
        Cluster data matrix with 14 elements and N clusters.

    XDIM : int, from load_header
    YDIM : int, from load_header
        The dimensions of the image acquisition.

    cluster_area : int, default 15
        Minimum area threshold (px) for event clusters.

    subpx : int, default 1 (no interpolation)
        Factor by which to subpixelize the event clusters.

    Returns
    -------
    cluster_image : 2D array-like (2D image)
        Quantitative 2D image of cluster events. Each pixel contains
        the number of alpha particle events detected in that pixel.

    xC_good : 1D array-like
        Array of x-coordinates of events in cluster_image.
    yC_good : 1D array-like
        Array of y-coordinates of events in cluster_image.
    time_s : 1D array-like
        Array of time of detection (s) of events in cluster_image.

    All three 1D arrays are indexed the same and ordered somewhat in time.
    e.g., xC_good[0] corresponds to yC[0] and time_s[0], all of "event 0",
    but "event 0" could be in a different tissue than "event 1".
    """

    _, time_ms, _, cluster_area, yC, xC, * \
        _ = delineate_cluster_filters(cluster_data)
    xC_filtered = xC[np.logical_and(xC > 0, cluster_area > 15)]
    yC_filtered = yC[np.logical_and(yC > 0, cluster_area > 15)]

    # apply subpixelization if desired
    if subpx == 1:
        xC_px_rounded = np.round(xC_filtered, 0)
        yC_px_rounded = np.round(yC_filtered, 0)
    else:
        xC_px_rounded = np.floor(xC_filtered * subpx)
        yC_px_rounded = np.floor(yC_filtered * subpx)

    # logical "and" with four statements to get positive coordinates only
    cluster_bool = ((xC_px_rounded > 0) * (yC_px_rounded > 0)
                    * np.isfinite(xC_px_rounded) * np.isfinite(yC_px_rounded))
    xC_good = xC_px_rounded[cluster_bool].astype(int)
    yC_good = yC_px_rounded[cluster_bool].astype(int)

    # build spatial image (no temporal information)
    cluster_image = np.zeros((int(subpx*YDIM), int(subpx*XDIM)))
    for i in range(len(yC_good)):
        cluster_image[yC_good[i], xC_good[i]] += 1

    # sort into arrays and get associated temporal information
    time_s = time_ms[cluster_area > 15]/1e3
    time_s = time_s[cluster_bool]

    return(cluster_image, xC_good, yC_good, time_s)


def prep_contour(source_image, binfac=4, gauss=15, thresh=0):
    '''Wrapper function to preprocess an image for contour-finding:
        -bin_ndarray by an integer factor (default 4)
        -binary thresholding (values > 0)
        -gaussian blur with a square kernel (default (15,15))

    This will NOT change the image in-place. The original image can
    still be quantitatively analyzed using the contours found if 
    they are properly transformed back.

    Parameters
    ----------
    source_image : 2D array-like
        Cluster data matrix with 14 elements and N clusters.

    binfac : int, default 4
        Factor by which to bin pixels.
        Dimensions must be evenly divisible by this number.

    gauss : ODD int, default 15
        Dimension of the blurring kernel is (gauss, gauss).
        cv2.GaussianBlur requires an odd-integer kernel (see docs).

    Returns
    -------
    im : 2D array-like (2D image)
        Binned and smoothed image of reduced dimensions for contouring.
    '''
    binned_image = bin_ndarray(source_image, (np.array(
        np.shape(source_image))/binfac).astype(int), operation='sum')
    mask = binned_image > thresh
    bin_im = mask.astype('uint8')
    im = cv2.GaussianBlur(bin_im, (gauss, gauss), 0)
    return(im)


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    From J.F. Sebastian, https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
        [102 110 118 126 134]
        [182 190 198 206 214]
        [262 270 278 286 294]
        [342 350 358 366 374]]

    """
    operation = operation.lower()
    if operation not in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            "Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [x for p in compression_pairs for x in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def get_maskstack(source_image, contours, binfac=4, minA=500):
    """Fills in contours to generate masks for discrete tissue sections.

    Masks are made over the original (non-binned, non-smoothed) image.
    Only selects contours of area greater than a certain threshold.

    Method is automatic but not guaranteed to be good.........

    Parameters
    ----------
    source_image : 2D array-like
        Pixelized event image from pixelize_image.

    contours : weird data structure from cv2.findContours, i.e.
        contours, hierarchy = cv2.findContours(...)

    binfac : int, default 4
        Factor by which THE IMAGE WAS BINNED BEFORE CONTOURING.
        i.e., the bin factor that was used in prep_contour.

    minA : int, default 500
        Minimum area threshold (pixels) to build a ROI from contour.

    Returns
    -------
    maskstack : 3D array-like (N x YDIM X XDIM)
        Stack of irregular masks, each of the original image size.
        Each shows one of N detected tissue slices.
    """
    good_contours = [binfac*c for c in contours if cv2.contourArea(c) > minA]
    xdim, ydim = np.shape(source_image)

    maskstack = np.zeros((len(good_contours), xdim, ydim))
    for i in range(len(good_contours)):
        mask = np.zeros_like(source_image)
        cv2.drawContours(mask, good_contours, i, (255, 255, 255), -1)
        mask_norm = (mask/255).astype(np.uint8)
        maskstack[i, :, :] = mask_norm

    return(maskstack)


def events_in_ROI(x, y, time_s, maskstack):
    """Creates a boolean mask to associate temporal events with tissue ROIs.

    Converts each 2D spatial mask from get_maskstack into a boolean mask that 
    corresponds to the 1D arrays from pixelize_image.

    Parameters
    ----------
    x, y, time_s: 1D array-likes
        Arrays for the x, y, and time coordinates from pixelize_image.

    maskstack: 3D array-like
        Stack of masks over spatial image for each tissue ROI, of len N.

    Returns
    -------
    ROI_array_bool : 2D array-like (N x len(x))
        Array of N 1D arrays that indicate whether each event is in the ROI.
    """
    ROI_array_bool = np.zeros((len(maskstack), len(x)))
    for i in range(len(maskstack)):
        mask = maskstack[i, :, :]
        inROI_bool = mask[y.astype(int), x.astype(int)] * \
            (time_s > 0) * np.isfinite(x) * np.isfinite(y)
        ROI_array_bool[i, :] = inROI_bool

    return(ROI_array_bool.astype(bool))


def get_ROIs(contours, pad=10, minA=500, binfac=4):
    '''Gets the coordinates in image space to "cut out" the ROIs with some pad.
    Wrapper function that:
        -finds bounding rectangles for cv2 contours
        -adds ROI border padding around bounding rectangles (10px)
        -rescales ROIs by amount that image was binned by to generate contours

    BINFAC MUST BE THE SAME AS IN prep_contour.

    Minimum area and border padding may be adjusted.
    Returns (n,4) 2D array with n the number of ROIs and columns (x,y,w,h).

    TODO: Contour finding is currently redundant with get_maskstack,
    possibility of mismatch if minA changes between functions.

    '''
    good_contours = [cont for cont in contours if cv2.contourArea(cont) > minA]
    ROI_array = np.zeros((len(good_contours), 4))

    for i in range(len(good_contours)):
        cnt = good_contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        ROI_array[i, :] = x-pad//2, y-pad//2, w+pad, h+pad
    ROI_array = ROI_array * binfac

    return(ROI_array.astype(int))


def constrain_exponential(x, a, thalf):
    return a*np.exp(-np.log(2)/thalf*x)


def fit_routine(temporal_array, func=constrain_exponential, p0=[0, 9.92*24*3600], binsize=1000):
    """Bins events into time histogram and performs curve fit.

    Uses Scipy curve_fit (least-square optimization) and returns fit
    parameters and goodness-of-fit for activity calculation.

    Parameters
    ----------
    temporal array : 1D array-like
        Data containing the time of events to be fit.

    func : function, default exponential A*exp(-ln(2)*x / t_half)
        Function to which to fit the histogram.
        No linear constant is returned by default.

    p0 : 1D array-like (1 x 2), default [0, 9.92*24*3600]
        Initial parameters for Ai and half-life to be fed into scipy.curve_fit.

    binsize : int, default 1000
        Time (s) of bin sizes. The activity of a given point in time
        will be estimated as N/binsize events per second = Bq.
        1000 is convenient because it yields N mBq per bin.

        Note that if you use binsize =/= you will need to correct
        the resulting fit parameter A by that factor.

    Returns
    -------
    count, bins : output from np.histogram

    popt, pcov : output from scipy.curve_fit

    param_std : uncertainty measure for parameters (popt)
        Equivalent to sqrt(diag(pcov))

    res : residuals of the data

    chisq, chisqn : chi-squared and reduced chi-squared values for the fit.
    """
    nbins = np.round(temporal_array[-1]/binsize)
    count, bins = np.histogram(temporal_array, np.arange(0, nbins)*binsize)

    timepoints = bins[:-1] + binsize/2
    popt, pcov = curve_fit(f=func, xdata=timepoints, ydata=count,
                           p0=p0, sigma=np.maximum(np.ones_like(count), np.sqrt(count)), bounds=(0, np.inf))
    param_std = np.sqrt(np.diag(pcov))
    res = count - func(timepoints, *popt)

    chisq = np.sum(res**2/func(timepoints, *popt))
    chisqn = chisq/len(timepoints)

    return(count, bins, popt, pcov, param_std, res, chisq, chisqn)