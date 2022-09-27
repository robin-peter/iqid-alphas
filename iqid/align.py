import os
import glob
import numpy as np
from skimage import transform, io, exposure, filters, draw, util
import re
'''Functions for alignment and registration of an iQID activity image stack.'''


# two helper functions for natural sorting of file names
# from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    """Helper function for natural sort order."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Helper function for natural sort order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def assemble_stack(imdir=None, fformat='tif'):
    """Alphabetizes (natural sort) a set of similarly named images into a np stack.

    Future work can look for similarly named images and separate into types.
    For proper alphabetization, files must have the same prefixes, e.g.:

    H700_LN_2hrPI_iQID_01_Activity_Estimate_mBq_2x.tif
    H700_LN_2hrPI_iQID_02_Activity_Estimate_mBq_2x.tif
    H700_LN_2hrPI_iQID_03_Activity_Estimate_mBq_2x.tif
    ...
    H700_LN_2hrPI_iQID_14_Activity_Estimate_mBq_2x.tif

    Parameters
    ----------
    imdir : str
        The name of the directory containing the images

    fformat : str
        File extension of the images, default 'tif'.

    Returns
    -------
    unreg : array-like
        The concatenated but unregistered stack of images.
    """

    data_path = os.path.join('.', imdir)
    fileList = glob.glob(data_path + '\*.' + fformat)
    fileList.sort(key=natural_keys)
    unreg = io.ImageCollection(fileList)
    unreg = np.array(unreg)
    return(unreg)


def ignore_images(data, exclude_list, pad=True):
    """Generates a new image stack excluding images defined in idx_list.

    Parameters
    ----------
    data : 2D array-like
        The image stack being modified

    idx_list: 1D array-like
        The indices of images to be excluded

    pad: bool
        A flag used to replace excluded images with a copy (default True)
        if True, replaces with a copy of the previous image.
        if False, eliminate bad images and shift stack accordingly.

    Returns
    -------
    unreg : array-like
        The concatenated but unregistered stack of images.

    NOTE: This function can only handle 2 "bad" images in a row.
    User should use discretion and adjust their data set accordingly
    (e.g., manually replace a torn image with a copy).
    """

    idx_list = np.array(exclude_list)

    if len(idx_list) == 0:
        return(data)

    if pad:
        data_clean = np.copy(data)
        data_clean[idx_list] = data_clean[np.abs(idx_list-1)]
        # abs(): if the bad image is the 0th, replace with the 1st instead
    else:
        pseudo_mask = np.ones(len(data), dtype=bool)
        pseudo_mask[idx_list] = 0
        data_clean = data[pseudo_mask]
    return(data_clean)


def get_SSD(im1, im2):
    """Simple helper to compute SSD (MSE) between two same-shape images."""
    N = np.size(im1)
    SSD = np.sum((im1-im2)**2)/N
    return(SSD)


def coarse_rotation(mov, ref, deg=2):
    """Coarse rotation of an image to minimize SSD (MSE) compared to reference.

    Brute-force method that checks the SSD between the reference image (ref)
    and a rotated version of the input image (mov). The comparison is made at
    each position within one full rotation in increments of the specified
    degree angle.

    Parameters
    ----------
    mov : 2D array-like
        The "moving" image to be rotated

    ref: 2D array-like
        The reference image

    deg: float
        The angular increment in degrees (default 2)

    Returns
    -------
    rot : 2D array-like
        The rotated version of input image mov
    """

    num_measurements = int(np.floor(360/deg))
    SSD = np.zeros(num_measurements)
    for i in range(num_measurements):
        rim = transform.rotate(mov, deg*i)
        SSD[i] = get_SSD(rim, ref)
    rot = transform.rotate(mov, deg*np.argmin(SSD))
    outdeg = deg*np.argmin(SSD)
    return(rot, outdeg)


def coarse_stack(unreg, deg=2, avg_over=1):
    """Align a stack of images using SSD-minimizing coarse rotation.

    Each image n is coarse aligned (see coarse_rotation) to the previous
    image (avg_over=1) or the mean of previously aligned images (avg_over > 1).
    Use averaging to make it more robust to rotationally similar samples.
    The 0th image is not changed.

    If avg_over = m, then the first m images are aligned to the average of
    as many aligned images as possible.

    Parameters
    ----------
    unreg : 3D array-like
        A stack of unregistered images of same shape, e.g. from either
        assemble_stack() or ignore_images()

    deg: float, optional
        The angular increment in degrees (default 2)

    avg_over: int, optional
        The number of prior aligned images to average to use as the
        next reference image

    Returns
    -------
    reg : 3D array-like
        The (coarsely) registered stack of images
    """

    reg = np.zeros_like(unreg)
    reg[0:avg_over,:,:] = unreg[0:avg_over,:,:]

    for i in np.arange(1, avg_over):
        ref = np.mean(reg[:i,:,:], axis=0)
        mov = unreg[i,:,:]
        reg[i,:,:],_ = coarse_rotation(mov, ref, deg)

    for i in np.arange(avg_over, len(unreg)):
        ref = np.mean(reg[(i-avg_over):i,:,:], axis=0)
        mov = unreg[i,:,:]
        reg[i,:,:],_ = coarse_rotation(mov, ref, deg)
    return(reg)


def format_circle(h, k, r, numpoints=400):
    '''Initialize a circle enclosing the ROI to start the contour.

    Circle has center (h,k) and radius r.

    Uses row-column instead of x-y format due to the convention
    of the skimage active_contour function.

    Future work could incorporate oblate-ness.
    '''
    s = np.linspace(0, 2*np.pi, numpoints)
    row = h + r*np.sin(s)
    col = k + r*np.cos(s)
    init_circle = np.array([row, col]).T
    return(init_circle)


def binary_mask(img, finagle=1):
    '''Simple Otsu threshold method to generate a mask.

    Utilizes a non-optimized "finagle-factor" to adjust the threshold,
    left up to user discretion.
    '''
    # simple Otsu method to generate a mask
    thresh = filters.threshold_otsu(img)
    mask = img > finagle*thresh
    return(mask)


def mask_from_contour(img, snake):
    '''Create a mask of values from those enclosed by the contour snake.'''
    rr, cc = draw.polygon(snake[:,0], snake[:,1], img.shape)
    newmask = np.zeros_like(img)
    newmask[rr, cc] = 1
    return(newmask)

def to_shape(mov, x_, y_):
    # from https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape
    """Zero-pads an image to a certain shape (x_, y_).

    Parameters
    ----------
    mov : 2D array-like
        Image to be padded.

    x_, y_ : 1D array-like
        The x and y dimensions to which to pad.

    Returns
    -------
    2D padded image of dimension (x_, y_)
    """

    y, x = np.shape(mov)

    y_pad = y_-y
    x_pad = x_-x
    return np.pad(mov,((y_pad//2, y_pad//2 + y_pad%2), 
                 (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant', constant_values=(1,1))


def pad_2d_masks(movmask, refmask):
    """Detects the larger dimensions between two images and pads them both to them.

    Parameters
    ----------
    movmask : 2D array-like
        Binary mask of the "moving" image (iQID dose-rate).

    refmask : 2D array-like
        Binary mask of the "reference" image (H&E stain).

    Returns
    -------
    movpad : 2D array-like
        Padded binary mask of the "moving" image (iQID dose-rate).

    refpad : 2D array-like
        Padded binary mask of the "reference" image (H&E stain).
    """
    y, x = np.shape(movmask)
    y_, x_ = np.shape(refmask)

    newy = max(y, y_)
    newx = max(x, x_)

    movpad = to_shape(movmask, newx, newy)
    refpad = to_shape(refmask, newx, newy)
    
    return(movpad, refpad)


def to_shape_rgb(mov, x_, y_):
    """Same function as to_shape but for RGB images (instead of binary).
    Useful when trying to overlay contours or pad the H&E stain."""
    y, x, _ = np.shape(mov)

    y_pad = y_-y
    x_pad = x_-x
    return np.pad(mov,
                  ((y_pad//2, y_pad//2 + y_pad%2),
                   (x_pad//2, x_pad//2 + x_pad%2),
                   (0,0)),
                  mode = 'constant', constant_values=(255, 255))


def pad_rgb_im(im_2d, im_rgb):
    """Same function as pad_2d_masks but where one image is RGB 
    and the other is grayscale/binary.
    Useful when trying to overlay contours or pad the H&E stain."""
    y, x = np.shape(im_2d)
    y_, x_, _ = np.shape(im_rgb)

    newy = max(y, y_)
    newx = max(x, x_)

    movpad = to_shape(im_2d, newx, newy)
    refpad = to_shape_rgb(im_rgb, newx, newy)

    return(movpad, refpad)


def crop_down(rgb_overlay, rgb_ref):
    """Crops an overlay image (e.g., some colored contours) to the correct size 
    for co-registration with a reference image rgb_ref."""
    dwidth, dheight = np.array(np.shape(rgb_overlay))[:2]-np.array(np.shape(rgb_ref))[:2]
    cropped_ol = util.crop(rgb_overlay, 
                        ((int(dwidth/2)+dwidth%2, int(dwidth/2)),
                         (int(dheight/2)+dheight%2, int(dheight/2)),
                         (0,0)))
    return(cropped_ol)

    
# Several helpful visualization functions from PyStackReg documentation
# https://pystackreg.readthedocs.io/en/latest/

def overlay_images(imgs, equalize=False, aggregator=np.mean):
    '''Overlay two images to display alignment.'''
    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)
    return aggregator(imgs, axis=0)


def composite_images(imgs, equalize=False, aggregator=np.mean):
    '''Overlay two images, using colors to show alignment and misalignment.'''
    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]
    imgs = [img / img.max() for img in imgs]
    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))
    imgs = np.dstack(imgs)
    return imgs