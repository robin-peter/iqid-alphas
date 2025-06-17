import os
import glob
import numpy as np
from skimage import transform, io, exposure, filters, draw, util
import cv2
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import warnings
import imagesize


from iqid import helper


'''Functions for alignment and registration of an iQID activity image stack.'''


def get_maxdim(fileList):
    '''Gets the maximum image dimensions from a stack of images without reading in each image.
    Uses imagesize package. (https://pypi.org/project/imagesize/)
    TODO: incorporate this into the assemble_stack functions.

    Parameters
    ----------
    fileList : list
        List of file names (of images), i.e. from glob.glob().

    Returns
    -------
    (maxh, maxw): tuple of ints
        Maximum dimensions of images in the list.

    '''
    temp = np.zeros((len(fileList), 2))
    for i in range(len(fileList)):
        w, h = imagesize.get(fileList[i])
        temp[i, :] = h, w

    maxh = int(np.max(temp[:, 0]))
    maxw = int(np.max(temp[:, 1]))
    return (maxh, maxw)


def assemble_stack(imdir=None, fformat='tif', pad=False):
    """Alphabetizes (natural sort) a set of similarly named images into a np stack.
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

    pad : bool
        True if image files are not the same size and should be padded with zeros to match.

    Returns
    -------
    imcollection : array-like
        The concatenated but unregistered stack of images.
    """

    data_path = os.path.join('.', imdir)
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if pad:
        temp = np.zeros((len(fileList), 2))
        for i in range(len(fileList)):
            im = io.imread(fileList[i])
            h, w = im.shape
            temp[i, :] = h, w

        maxh = int(np.max(temp[:, 0]))
        maxw = int(np.max(temp[:, 1]))

        imcollection = np.zeros((len(fileList), maxh, maxw))
        for i in range(len(fileList)):
            im = io.imread(fileList[i])
            h, w = im.shape
            # use shape and not w/h in case image was on edge and not full value of rectangle
            padx = maxw - w
            pady = maxh - h
            im = cv2.copyMakeBorder(im, int(np.ceil(pady/2)), int(np.floor(pady/2)),
                                    int(np.ceil(padx/2)), int(np.floor(padx/2)), cv2.BORDER_CONSTANT, value=(0, 0, 0))
            imcollection[i, :, :] = im
    else:
        imcollection = io.ImageCollection(fileList)
        # , dtype=object) # this toggle sometimes fixes/breaks things
        imcollection = np.array(imcollection)

    return (imcollection)


def assemble_stack_hne(imdir=None, fformat='tif', color=(0, 0, 0), pad=True):
    data_path = os.path.join('.', imdir)
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if pad:
        temp = np.zeros((len(fileList), 2))
        for i in range(len(fileList)):
            im = io.imread(fileList[i])
            h, w, CHA = im.shape
            temp[i, :] = h, w

        maxh = int(np.max(temp[:, 0]))
        maxw = int(np.max(temp[:, 1]))

        imcollection = np.zeros((len(fileList), maxh, maxw, CHA))
        for i in range(len(fileList)):
            im = io.imread(fileList[i])
            h, w, _ = im.shape
            result = np.full((maxh, maxw, CHA), color, dtype=np.uint8)
            x_center = (maxw - w) // 2
            y_center = (maxh - h) // 2
            result[y_center:y_center+h, x_center:x_center+w, :] = im
            imcollection[i, :, :] = result
    else:
        imcollection = io.ImageCollection(fileList)
        imcollection = np.array(imcollection)

    return (imcollection.astype(int))


def pad_stack_he(data_path, fformat='png', color=(0, 0, 0), savedir=None, verbose=False):
    fileList = glob.glob(os.path.join(data_path, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    if verbose:
        print(*fileList, sep='\n')

    newdir = os.path.join(savedir, 'padded')
    Path(newdir).mkdir(parents=True, exist_ok=True)

    print('Detecting dimensions...')
    maxh, maxw = get_maxdim(fileList)

    for i in trange(len(fileList), desc='Padding images'):
        im = io.imread(fileList[i])
        imname = os.path.basename(os.path.splitext(fileList[i])[0])

        s = im.shape
        if len(s) == 3:
            h, w, CHA = im.shape
            result = np.full((maxh, maxw, CHA), color, dtype=np.uint8)
        elif len(s) == 2:
            h, w = im.shape
            result = np.full((maxh, maxw), color[0], dtype=np.float32)
        else:
            raise
        x_center = (maxw - w) // 2
        y_center = (maxh - h) // 2

        if len(s) == 3:
            result[y_center:y_center+h, x_center:x_center+w, :] = im
        elif len(s) == 2:
            result[y_center:y_center+h, x_center:x_center+w] = im
        else:
            raise

        if fformat == 'tif':  # i.e. for Slicer 3d
            io.imsave(os.path.join(newdir, '{}_pad.'.format(imname) +
                      fformat), result.astype(np.float32), plugin='tifffile')
        else:
            io.imsave(os.path.join(
                newdir, '{}_pad.'.format(imname) + fformat), result)


def organize_onedir(imdir=None, include_idx=[], order_idx=[], fformat='png'):
    nameDict = {'full_masks': 'mask',
                'mBq_images': 'mBq',
                'ROI_masks': 'mask'}

    fileList = glob.glob(os.path.join(imdir, '*.' + fformat))
    fileList.sort(key=helper.natural_keys)

    # error checking for input indices
    if len(include_idx) == 0:
        include_idx = np.ones(len(fileList))
    if len(order_idx) == 0:
        order_idx = np.arange(len(fileList))

    err_1 = "Inclusion index length doesn't match saved images."
    err_2 = "Order index length doesn't match inclusion indices."
    assert len(include_idx) == len(fileList), err_1
    assert len(order_idx) == sum(include_idx), err_2

    usr_yn = helper.get_yn(input('Delete images in-place?'))
    if usr_yn:
        del_list = np.array(fileList)[np.logical_not(include_idx).astype(bool)]
        for f in del_list:
            os.remove(f)

    usr_yn = helper.get_yn(input('Rename images in-place?'))
    if usr_yn:
        fileList = glob.glob(onepath + '\*.' + fformat)
        prefix = os.path.dirname(fileList[0])
        for i in range(len(fileList)):
            newname = os.path.join(prefix, nameDict[os.path.basename(
                prefix)] + '_{}.{}'.format(order_idx[i], fformat))
            os.rename(fileList[i], newname)


def preprocess_topdir(topdir, include_idx=[], order_idx=[]):
    # rewrite function to just do one prompt for all three folderrs
    # then prompt to move to subdirectory under Mouse 1 kidney 1st
    nameDict = {'full_masks': 'mask',
                'mBq_images': 'mBq',
                'mBq_image_previews': 'mBq_preview',
                'ROI_masks': 'mask'}

    fformatDict = {'full_masks': 'png',
                   'mBq_images': 'tif',
                   'mBq_image_previews': 'png',
                   'ROI_masks': 'png'}

    subdirs = helper.list_studies(topdir)

    usr_yn = helper.get_yn(input('Delete images in-place?'))
    if usr_yn:
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            try:
                fformat = fformatDict[os.path.basename(subdir)]
            except KeyError:
                continue
            fileList = glob.glob(os.path.join(subdir, '*.' + fformat))
            fileList.sort(key=helper.natural_keys)
            assert len(include_idx) == len(
                fileList), "Inclusion index length ({}) doesn't match saved images ({}).".format(
                    len(include_idx), len(fileList)
            )
            del_list = np.array(fileList)[
                np.logical_not(include_idx).astype(bool)]
            for f in del_list:
                os.remove(f)

    usr_yn = helper.get_yn(input('Rename images in-place?'))
    if usr_yn:
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            try:
                fformat = fformatDict[os.path.basename(subdir)]
            except KeyError:
                continue
            fileList = glob.glob(os.path.join(subdir, '*.' + fformat))
            fileList.sort(key=helper.natural_keys)

            assert len(order_idx) == sum(
                include_idx), "Order index length doesn't match inclusion indices."

            # rename all arbitrarily to circumvent permissions and duplicates problem
            for j in range(len(fileList)):
                tempname = os.path.join(
                    subdir, 'temp_{}.{}'.format(j, fformat))
                os.rename(fileList[j], tempname)

            # reimport and sort appropriately
            fileList = glob.glob(os.path.join(subdir, '*.' + fformat))
            fileList.sort(key=helper.natural_keys)
            for j in range(len(fileList)):
                newname = os.path.join(subdir, nameDict[os.path.basename(
                    subdir)] + '_{}.{}'.format(order_idx[j], fformat))
                os.rename(fileList[j], newname)

    usr_yn = helper.get_yn(input('Rename as new subdirectory?'))
    if usr_yn:
        name = input('New folder name:')
        newname = os.path.join(os.path.dirname(topdir), name)
        os.rename(topdir, newname)


def ignore_images(data, exclude_list, pad='backwards'):
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
        return (data)

    if pad:
        data_clean = np.copy(data)
        if pad == 'backwards':
            data_clean[idx_list] = data_clean[np.abs(idx_list-1)]
        elif pad == 'forwards':
            data_clean[idx_list] = data_clean[np.abs(idx_list+1)]
        else:
            print('unaccepted pad type')

        # abs(): if the bad image is the 0th, replace with the 1st instead
    else:
        pseudo_mask = np.ones(len(data), dtype=bool)
        pseudo_mask[idx_list] = 0
        data_clean = data[pseudo_mask]
    return (data_clean)


def get_SSD(im1, im2):
    """Simple helper to compute SSD (MSE) between two same-shape images."""
    N = np.size(im1)
    SSD = np.sum((im1-im2)**2)/N
    return (SSD)


def coarse_rotation(mov, ref, deg=2, interpolation=0, gauss=5, preserve_range=True, recenter=False, convert_to_grayscale_for_ssd=False):
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

    interpolation: int from 0 to 5
        Passed to skimage.transform.rotate to describe the order of interpolation.
        0: nearest-neighbor
        See docs for other options.

    gauss: int
        Amount to smooth images for SSD comparison.

    Returns
    -------
    rot : 2D array-like
        The rotated version of input image mov

    outdeg : float
        The degree rotation to be applied to the moving image
    """

    if recenter:
        mov = recenter_im(mov)
        ref = recenter_im(ref)

    # Convert to float32 for GaussianBlur compatibility
    mov_for_blur = mov.astype(np.float32)
    ref_for_blur = ref.astype(np.float32)

    # Prepare images for SSD calculation (optional grayscale conversion)
    mov_for_ssd = mov_for_blur
    ref_for_ssd = ref_for_blur

    if convert_to_grayscale_for_ssd and mov_for_blur.ndim == 3 and mov_for_blur.shape[-1] in [3, 4]:
        # Ensure uint8 for cvtColor if it's not already scaled 0-1 float
        # Assuming mov_for_blur might be 0-255 float32 from .astype(np.float32) on uint8 images
        if np.max(mov_for_blur) > 1.5: # Heuristic for 0-255 range
            mov_for_ssd = cv2.cvtColor(mov_for_blur.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else: # Assuming 0-1 range float
            mov_for_ssd = cv2.cvtColor((mov_for_blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    if convert_to_grayscale_for_ssd and ref_for_blur.ndim == 3 and ref_for_blur.shape[-1] in [3, 4]:
        if np.max(ref_for_blur) > 1.5:
            ref_for_ssd = cv2.cvtColor(ref_for_blur.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            ref_for_ssd = cv2.cvtColor((ref_for_blur * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if gauss:
        gmov = cv2.GaussianBlur(mov_for_ssd, (gauss, gauss), 0)
        gref = cv2.GaussianBlur(ref_for_ssd, (gauss, gauss), 0)
    else:
        gmov = mov_for_ssd
        gref = ref_for_ssd

    num_measurements = int(np.floor(360/deg))
    SSD = np.zeros(num_measurements)
    for i in range(num_measurements):
        rim = transform.rotate(gmov, deg*i, order=interpolation)
        SSD[i] = get_SSD(rim, gref)

    rot = transform.rotate(mov, deg*np.argmin(SSD),
                           preserve_range=preserve_range, order=interpolation)
    outdeg = deg*np.argmin(SSD)
    return (rot, outdeg)


def coarse_stack(unreg, deg=2, avg_over=1, preserve_range=True, return_deg=False, convert_to_grayscale_for_ssd=False):
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
    reg[0:avg_over, :, :] = unreg[0:avg_over, :, :]

    degs = np.zeros(len(unreg))

    for i in np.arange(1, avg_over):
        ref = np.mean(reg[:i, :, :], axis=0)
        mov = unreg[i, :, :]
        reg[i, :, :], degs[i] = coarse_rotation(
            mov, ref, deg, preserve_range=preserve_range, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)

    for i in np.arange(avg_over, len(unreg)):
        ref = np.mean(reg[(i-avg_over):i, :, :], axis=0)
        mov = unreg[i, :, :]
        reg[i, :, :], degs[i] = coarse_rotation(
            mov, ref, deg, preserve_range=preserve_range, convert_to_grayscale_for_ssd=convert_to_grayscale_for_ssd)

    if return_deg:
        return (reg, degs)
    else:
        return (reg)


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
    return (init_circle)


def binary_mask(img, finagle=1):
    '''Simple Otsu threshold method to generate a mask.

    Utilizes a non-optimized "finagle-factor" to adjust the threshold,
    left up to user discretion.
    '''
    # simple Otsu method to generate a mask
    thresh = filters.threshold_otsu(img)
    mask = img > finagle*thresh
    return (mask)


def mask_from_contour(img, snake):
    '''Create a mask of values from those enclosed by the contour snake.'''
    rr, cc = draw.polygon(snake[:, 0], snake[:, 1], img.shape)
    newmask = np.zeros_like(img)
    newmask[rr, cc] = 1
    return (newmask)


def to_shape(mov, x_, y_, vals=(0, 0)):
    # from https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape
    """One-pads an image to a certain shape (x_, y_).

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
    return np.pad(mov, ((y_pad//2, y_pad//2 + y_pad % 2),
                        (x_pad//2, x_pad//2 + x_pad % 2)),
                  mode='constant', constant_values=vals)


def pad_2d_masks(movmask, refmask, func=to_shape):
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

    movpad = func(movmask, newx, newy)
    refpad = func(refmask, newx, newy)

    return (movpad, refpad)


def to_shape_rgb(mov, x_, y_):
    """Same function as to_shape but for RGB images (instead of binary).
    Useful when trying to overlay contours or pad the H&E stain."""
    y, x, _ = np.shape(mov)

    y_pad = y_-y
    x_pad = x_-x
    return np.pad(mov,
                  ((y_pad//2, y_pad//2 + y_pad % 2),
                   (x_pad//2, x_pad//2 + x_pad % 2),
                   (0, 0)),
                  mode='constant', constant_values=(255, 255))


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

    return (movpad, refpad)


def to_shape_center(im, x_, y_):
    h, w = im.shape
    result = np.full((y_, x_), 0, dtype=type(im[0, 0]))
    x_center = (x_ - w) // 2
    y_center = (y_ - h) // 2
    result[y_center:y_center+h, x_center:x_center+w] = im
    return result


def crop_down(rgb_overlay, rgb_ref, axis='both'):
    """Crops an overlay image (e.g., some colored contours) to the correct size 
    for co-registration with a reference image rgb_ref."""
    dwidth, dheight = np.array(np.shape(rgb_overlay))[
        :2]-np.array(np.shape(rgb_ref))[:2]

    if axis == 'both':
        xbool = 1
        ybool = 1
    elif axis == 'x':
        xbool = 1
        ybool = 0
    elif axis == 'y':
        xbool = 0
        ybool = 1
    else:
        print('invalid axis choice')

    cropped_ol = util.crop(rgb_overlay,
                           (tuple(xbool * np.array([int(dwidth/2)+dwidth % 2, int(dwidth/2)])),
                            tuple(
                                ybool * np.array([int(dheight/2)+dheight % 2, int(dheight/2)])),
                            (0, 0)))
    return (cropped_ol)


def crop_to(im, x_, y_):
    # if you're seeing a nonsensical line image
    # it's because y < y_ or x < x_

    if im.shape[1] < x_ or im.shape[0] < y_:
        warnings.warn(
            "Dimensions of image must be larger than new shape. Returning unmodified image.")

    elif len(im.shape) > 2:  # rgb image
        dwidth, dheight = np.array(np.shape(im))[:2]-np.array([y_, x_])
        im = util.crop(im,
                       (np.array([int(dwidth/2)+dwidth % 2, int(dwidth/2)]),
                        np.array([int(dheight/2)+dheight % 2, int(dheight/2)]),
                        (0, 0)))
    elif len(im.shape) == 2:  # gs image
        dwidth, dheight = np.array(np.shape(im)) - np.array([y_, x_])
        im = util.crop(im,
                       (np.array([int(dwidth/2)+dwidth % 2, int(dwidth/2)]),
                        np.array([int(dheight/2)+dheight % 2, int(dheight/2)])))
    return (im)


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


def save_imbatch(imstack, newdir, prefix, fformat='tif'):
    Path(newdir).mkdir(parents=True, exist_ok=True)
    for i in trange(len(imstack)):
        io.imsave(os.path.join(newdir, prefix+'_{}'.format(i)+'.' + fformat),
                  imstack[i], plugin='tifffile', photometric='minisblack', check_contrast=False)


def concatenate_dsets(astack_1, astack_2):
    # pick larger one to pad up to
    xx = max(astack_1.shape[1], astack_2.shape[1])
    yy = max(astack_1.shape[2], astack_2.shape[2])

    new_a1 = np.zeros((len(astack_1), xx, yy))
    new_a2 = np.zeros((len(astack_2), xx, yy))

    for i in range(len(astack_1)):
        new_a1[i], _ = pad_2d_masks(
            astack_1[i], astack_2[0], func=to_shape)

    for i in range(len(astack_2)):
        new_a2[i], _ = pad_2d_masks(
            astack_2[i], astack_1[0], func=to_shape)
    astack_mBq = np.concatenate((new_a1, new_a2))
    return (astack_mBq)


def quantify_err(imstack, reg, tmat, vis=True):
    Ss = np.zeros(len(tmat))
    Zs = np.zeros((len(tmat), 2))
    for i in range(len(tmat)):
        T, R, Z, S = helper.decompose_affine(tmat[i])
        Ss[i] = S
        Zs[i] = Z

    print(
        "The largest amount of shear in this stack is {:.2f}.".format(np.max(Ss)))
    print("The largest amount of zoom in this stack is {:.1f}%.".format(
        (np.max(Zs)-1)*100))

    # Quantification of small-value errors introduced by rotations.
    pct_diff = (np.sum(reg)-np.sum(imstack))/np.sum(imstack)
    print(
        'Small-value errors result in summed activity difference of {:.2f}%.'.format(pct_diff*100))

    agg_unreg = overlay_images(imstack, aggregator=np.mean)
    agg_aff = overlay_images(reg, aggregator=np.mean)

    if vis:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(agg_unreg, cmap='inferno')
        ax[0].set_title('Unregistered')
        ax[0].axis('off')

        ax[1].imshow(agg_aff, cmap='inferno')
        ax[1].set_title('Registered')
        ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return pct_diff


def simple_slice(arr, inds, axis):
    """
    from Pystackreg
    https://github.com/glichtner/pystackreg/blob/b5d9c032f7d0ba48d8472f8c8e6b4589a52bcdab/pystackreg/util/__init__.py#L57
    """
    sl = [slice(None)] * arr.ndim
    sl[axis] = inds
    return arr[tuple(sl)]


def transform_stack(img, tmat, axis=0, order=0):
    '''Copy of pystackreg function transform_stack, but modified to allow
    for selection of specific interpolation order.

    Defaults to 0 (nearest-neighbor) to preserve quantitative accuracy
    over spatial precision for this application.'''

    out = img.copy().astype(float)

    for i in range(img.shape[axis]):
        slc = [slice(None)] * len(out.shape)
        slc[axis] = i

        # replace pystackreg transform with skimage transform to allow order=0
        out[tuple(slc)] = transform.warp(simple_slice(img, i, axis),
                                         tmat[i, :, :],
                                         order=order,
                                         mode='constant',
                                         cval=0.)

    return out


def recenter_im(im):
    # based on https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    non_empty_columns = np.where(im.max(axis=0) > 0)[0]
    non_empty_rows = np.where(im.max(axis=1) > 0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows),
               min(non_empty_columns), max(non_empty_columns))
    image_data_new = im[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    x, y = np.shape(im)
    im_padded = to_shape(image_data_new, y, x)
    return (im_padded)


def downsamp(ref, factor):
    ds_array = ref/255
    r = block_reduce(ds_array[:, :, 0], (factor, factor), np.mean)
    g = block_reduce(ds_array[:, :, 1], (factor, factor), np.mean)
    b = block_reduce(ds_array[:, :, 2], (factor, factor), np.mean)
    ds_array = np.stack((r, g, b), axis=-1)
    return ds_array


def shape_colorise(dr, ref, cmap=plt.cm.inferno):
    norm = plt.Normalize(vmin=0, vmax=np.max(dr))
    dr = cmap(norm(dr))[:, :, :3]
    x_, y_, _ = np.shape(ref)
    dr = crop_down(dr, ref, axis='both')
    dr = to_shape_rgb(dr, y_, x_)
    dr = (dr * 255).astype(np.uint8)
    return dr


def do_transform(mov, fac, deg, tf):
    out = transform.rescale(mov, fac)
    out = transform.rotate(out, deg)
    out = transform.warp(out, tf)
    out = np.round(out.clip(min=0))
    return out


def rescale_tmat(tmat, s):
    # scaling the translation part of the matrix to appropriate pixel dims
    # in the workflow, s = 1/fac, fac = iq_um_px / downsample /he_um_px

    tmat[0, 2] = tmat[0, 2] * s
    tmat[1, 2] = tmat[1, 2] * s
    return tmat


def tmat_3to2(tmat):
    # change 3d affine transform into 2d when importing from BigWarp
    tmat_2d = np.zeros((3, 3))
    tmat_2d[:2, :2] = tmat[:2, :2]
    tmat_2d[:2, 2] = tmat[:2, -1]
    tmat_2d[2, :] = tmat[2, :3]
    return tmat_2d


def do_transform_noscale(mov, ref, deg, tf):
    out = transform.rotate(mov, deg, order=0, preserve_range=True)
    out = transform.warp(out, tf, order=0, preserve_range=True)
    out = out.clip(min=0)
    out = recenter_im(out)
    out = crop_to(out, ref.shape[1], ref.shape[0])
    return out


def do_transform_noPSR(mov, ref, deg):
    # just coarse rotation and housekeeping
    out = transform.rotate(mov, deg, order=0, preserve_range=True)
    out = out.clip(min=0)
    out = recenter_im(out)
    out = crop_to(out, ref.shape[1], ref.shape[0])
    return out


def plot_compare(mov, ref, lab1=None, lab2=None, cmap1='inferno', cmap2=False, axis='off'):
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mov, cmap=cmap1)
    if cmap2:
        ax[1].imshow(ref, cmap2)
    else:
        ax[1].imshow(ref)
    ax[0].axis(axis)
    ax[1].axis(axis)
    ax[0].set_title(lab1)
    ax[1].set_title(lab2)
    plt.tight_layout()
    plt.show()
    plt.close()


def norm_im(im):
    return (255*im/np.max(im)).astype(np.uint8)


def colorise_out(im, cmap='inferno'):
    cm = plt.get_cmap(cmap)
    return norm_im(cm(norm_im(im))[:, :, :-1])


def myround(x, base=5):
    # return base * round(x/base)
    # updated to floor for the image processing use case
    # based on https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    # -------------------- ex. -------------------------
    # test_cases = np.array([10, 12, 13, 14, 16, 18, 20])
    # myround(test_cases) = array([10., 10., 10., 10., 15., 15., 20.])
    return base * np.floor(x/base)


def bin_bin(binary, he_um_px, iq_um_px, method='ndarray', op='sum'):
    # operation: sum or mean
    # how many bins are in the whole image?
    y, x = binary.shape  # HE pixels
    x_um = he_um_px * x  # um = um/px_HE * px_HE
    y_um = he_um_px * y

    xdim = int(x_um // iq_um_px)  # iQ px = um / (um/px_iq)
    # maximum number of HE pixels that will be used
    xtemp = int(np.floor(xdim * iq_um_px / he_um_px))
    xtt = myround(xtemp, xdim)

    ydim = int(y_um // iq_um_px)
    ytemp = int(np.floor(ydim * iq_um_px / he_um_px))
    ytt = myround(ytemp, ydim)

    cropped = crop_to(binary, int(xtt), int(ytt))
    if method == 'ndarray':
        h = helper.bin_ndarray(cropped/255, (ydim, xdim),
                               operation=op)  # older method
    elif method == 'cv2_nn':
        h = cv2.resize(cropped, (xdim, ydim), interpolation=cv2.INTER_NEAREST)
        # newer method - does not result in noise on the side of the microtumors due to interpolation
    return h


def bin_centroids(fileName, imsize, he_um_px, iq_um_px, minA=1):
    a, X, Y = np.genfromtxt(fileName, delimiter=',',
                            skip_header=1, usecols=(1, 2, 3), unpack=True)
    X = X[a > minA]
    Y = Y[a > minA]

    fac = iq_um_px / he_um_px
    y, x = imsize
    x_um = he_um_px * x  # um = um/px_HE * px_HE
    y_um = he_um_px * y

    xdim = int(x_um // iq_um_px)  # iQ px = um / (um/px_iq)
    ydim = int(y_um // iq_um_px)

    xc = X / fac
    yc = Y / fac

    xedges = np.linspace(0, xdim, num=xdim+1)
    yedges = np.linspace(0, ydim, num=ydim+1)

    h, _, _, _ = plt.hist2d(xc, yc, [xedges, yedges])
    return h
