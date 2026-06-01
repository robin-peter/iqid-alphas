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
    for i in trange(len(fileList), desc='Evaluating image sizes...'):
        w, h = imagesize.get(fileList[i])
        temp[i, :] = h, w

    maxh = int(np.max(temp[:, 0]))
    maxw = int(np.max(temp[:, 1]))
    return (maxh, maxw)


def assemble_stack(imdir=None, fformat='tif', pad=False, pad_color=(0, 0, 0)):
    """Natural-sort a directory of images into a numpy stack.

    Handles both grayscale (2D) and colour (3D) images automatically.
    When pad=True, images are centre-padded to the maximum dimensions in the set.

    Parameters
    ----------
    imdir : str
        Directory containing the images.
    fformat : str
        File extension, default 'tif'.
    pad : bool
        Pad images to a common size when they differ in shape.
    pad_color : tuple of int
        RGB fill value used when padding colour images (default black).

    Returns
    -------
    imcollection : ndarray, shape (N, H, W) or (N, H, W, C)
    """
    import imageio as iio

    fileList = helper.natural_sort(glob.glob(os.path.join(imdir, '*.' + fformat)))

    if not pad:
        images = [np.array(iio.imread(f)) for f in fileList]
        shapes = [im.shape for im in images]
        if len(set(shapes)) == 1:
            return np.array(images)
        # Inhomogeneous shapes — fall through to pad path
        pad = True

    first = np.array(iio.imread(fileList[0]))
    is_color = first.ndim == 3
    dims = np.array([np.array(iio.imread(f)).shape[:2] for f in fileList])
    maxh, maxw = int(dims[:, 0].max()), int(dims[:, 1].max())

    if is_color:
        CHA = first.shape[2]
        out = np.zeros((len(fileList), maxh, maxw, CHA), dtype=np.uint8)
        for i, f in enumerate(fileList):
            im = np.array(iio.imread(f))
            h, w = im.shape[:2]
            result = np.full((maxh, maxw, CHA), pad_color, dtype=np.uint8)
            yc, xc = (maxh - h) // 2, (maxw - w) // 2
            result[yc:yc+h, xc:xc+w] = im
            out[i] = result
        return out.astype(int)
    else:
        out = np.zeros((len(fileList), maxh, maxw), dtype=first.dtype)
        for i, f in enumerate(fileList):
            im = np.array(iio.imread(f))
            h, w = im.shape
            padx, pady = maxw - w, maxh - h
            out[i] = cv2.copyMakeBorder(
                im,
                int(np.ceil(pady/2)), int(np.floor(pady/2)),
                int(np.ceil(padx/2)), int(np.floor(padx/2)),
                cv2.BORDER_CONSTANT, value=0)
        return out


def assemble_stack_hne(imdir=None, fformat='tif', color=(0, 0, 0), pad=True):
    """Backward-compatible wrapper — use assemble_stack instead."""
    return assemble_stack(imdir, fformat=fformat, pad=pad, pad_color=color)


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


def preprocess_topdir(topdir, include_idx=None, order_idx=None,
                      delete=True, rename=True, new_name=None):
    """Delete excluded images and rename kept images across all export subdirectories.

    Parameters
    ----------
    topdir : str
        Top-level export directory produced by iQData.fitROIs (contains
        subdirs like mBq_images/, full_masks/, etc.).
    include_idx : array of 0/1, optional
        One entry per exported image; 0 = exclude. Defaults to keeping all.
    order_idx : array of int, optional
        New sequential indices for the kept images. Defaults to existing order.
    delete : bool
        If True, remove files where include_idx == 0.
    rename : bool
        If True, rename kept files to canonical <prefix>_<n>.<ext> names.
    new_name : str, optional
        If given, rename topdir to this name inside its parent directory.
    """
    nameDict = {'full_masks': 'mask',
                'mBq_images': 'mBq',
                'mBq_image_previews': 'mBq_preview',
                'ROI_masks': 'mask'}
    fformatDict = {'full_masks': 'png',
                   'mBq_images': 'tif',
                   'mBq_image_previews': 'png',
                   'ROI_masks': 'png'}

    subdirs = helper.list_studies(topdir)

    # resolve defaults against the first recognised subdir
    if include_idx is None or order_idx is None:
        for subdir in subdirs:
            fformat = fformatDict.get(os.path.basename(subdir))
            if fformat is None:
                continue
            fileList = helper.natural_sort(glob.glob(os.path.join(subdir, '*.' + fformat)))
            n = len(fileList)
            if include_idx is None:
                include_idx = np.ones(n, dtype=int)
            if order_idx is None:
                order_idx = np.arange(int(np.sum(include_idx)))
            break

    include_idx = np.asarray(include_idx)
    order_idx = np.asarray(order_idx)

    for subdir in subdirs:
        fformat = fformatDict.get(os.path.basename(subdir))
        if fformat is None:
            continue
        fileList = helper.natural_sort(glob.glob(os.path.join(subdir, '*.' + fformat)))

        assert len(include_idx) == len(fileList), (
            "Inclusion index length ({}) doesn't match saved images ({}).".format(
                len(include_idx), len(fileList)))
        assert len(order_idx) == int(np.sum(include_idx)), (
            "Order index length doesn't match inclusion indices.")

        if delete:
            for f in np.array(fileList)[include_idx == 0]:
                os.remove(f)

        if rename:
            kept = helper.natural_sort(glob.glob(os.path.join(subdir, '*.' + fformat)))
            prefix = nameDict[os.path.basename(subdir)]
            # first pass: rename to temp names to avoid collisions
            for j, f in enumerate(kept):
                os.rename(f, os.path.join(subdir, 'temp_{}.{}'.format(j, fformat)))
            # second pass: apply canonical names in order
            temps = helper.natural_sort(glob.glob(os.path.join(subdir, 'temp_*.' + fformat)))
            for j, f in enumerate(temps):
                os.rename(f, os.path.join(subdir, '{}_{}.{}'.format(prefix, order_idx[j], fformat)))

    if new_name is not None:
        os.rename(topdir, os.path.join(os.path.dirname(topdir), new_name))


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


def coarse_rotation(mov, ref, deg=2, interpolation=0, gauss=5, preserve_range=True, recenter=False):
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

    if gauss:
        gmov = cv2.GaussianBlur(mov, (gauss, gauss), 0)
        gref = cv2.GaussianBlur(ref, (gauss, gauss), 0)
    else:
        gmov = mov
        gref = ref

    num_measurements = int(np.floor(360/deg))
    SSD = np.zeros(num_measurements)
    for i in range(num_measurements):
        rim = transform.rotate(gmov, deg*i, order=interpolation)
        SSD[i] = get_SSD(rim, gref)

    rot = transform.rotate(mov, deg*np.argmin(SSD),
                           preserve_range=preserve_range, order=interpolation)
    outdeg = deg*np.argmin(SSD)
    return (rot, outdeg)


def coarse_stack(unreg, deg=2, avg_over=1, preserve_range=True, return_deg=False):
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
            mov, ref, deg, preserve_range=preserve_range)

    for i in np.arange(avg_over, len(unreg)):
        ref = np.mean(reg[(i-avg_over):i, :, :], axis=0)
        mov = unreg[i, :, :]
        reg[i, :, :], degs[i] = coarse_rotation(
            mov, ref, deg, preserve_range=preserve_range)

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


def pad_to(im, x_, y_, fill=0):
    """Centre-pad im to shape (y_, x_), handling 2D and 3D (colour) arrays.

    Parameters
    ----------
    im : ndarray
        Image to pad. May be 2D (H, W) or 3D (H, W, C).
    x_, y_ : int
        Target width and height.
    fill : scalar
        Constant fill value (default 0). Use 255 for white H&E backgrounds.

    Returns
    -------
    ndarray of shape (y_, x_) or (y_, x_, C)
    """
    y, x = im.shape[:2]
    y_pad, x_pad = y_ - y, x_ - x
    pad_width = [(y_pad//2, y_pad//2 + y_pad % 2),
                 (x_pad//2, x_pad//2 + x_pad % 2)]
    if im.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(im, pad_width, mode='constant', constant_values=fill)


def pad_to_common(im1, im2, fill=0):
    """Pad two images to the maximum dimensions of each other.

    Handles any combination of 2D and 3D arrays.

    Returns
    -------
    im1_padded, im2_padded : ndarrays at common (max_h, max_w[, C]) shape
    """
    newy = max(im1.shape[0], im2.shape[0])
    newx = max(im1.shape[1], im2.shape[1])
    return pad_to(im1, newx, newy, fill), pad_to(im2, newx, newy, fill)


# ── Backward-compatible aliases ────────────────────────────────────────────────

def to_shape(mov, x_, y_, vals=(0, 0)):
    return pad_to(mov, x_, y_, fill=vals[0] if hasattr(vals, '__len__') else vals)


def to_shape_rgb(mov, x_, y_):
    return pad_to(mov, x_, y_, fill=255)


def to_shape_center(im, x_, y_):
    return pad_to(im, x_, y_, fill=0)


def pad_2d_masks(movmask, refmask, func=None):
    return pad_to_common(movmask, refmask)


def pad_rgb_im(im_2d, im_rgb):
    return pad_to_common(im_2d, im_rgb)


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
    maxh = max(astack_1.shape[1], astack_2.shape[1])
    maxw = max(astack_1.shape[2], astack_2.shape[2])
    new_a1 = np.stack([pad_to(astack_1[i], maxw, maxh) for i in range(len(astack_1))])
    new_a2 = np.stack([pad_to(astack_2[i], maxw, maxh) for i in range(len(astack_2))])
    return np.concatenate((new_a1, new_a2))


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
    im_padded = pad_to(image_data_new, y, x)
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
    dr = pad_to(dr, y_, x_, fill=255)
    dr = (dr * 255).astype(np.uint8)
    return dr


def do_transform(mov, fac, deg, tf=None):
    """Rescale, rotate, and optionally warp a moving image.

    Parameters
    ----------
    mov : 2D array
    fac : float — rescale factor
    deg : float — rotation angle in degrees
    tf : ndarray, optional warp matrix; skipped when None
    """
    out = transform.rescale(mov, fac)
    out = transform.rotate(out, deg)
    if tf is not None:
        out = transform.warp(out, tf)
    return np.round(out.clip(min=0))


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


def do_transform_noscale(mov, ref, deg, tf=None):
    """Rotate and optionally warp a moving image, then recenter and crop to ref.

    Parameters
    ----------
    mov : 2D array — moving image
    ref : 2D array — reference image (used only for output shape)
    deg : float — rotation angle in degrees
    tf : ndarray, optional — PSR warp matrix; skipped when None (coarse rotation only)
    """
    out = transform.rotate(mov, deg, order=0, preserve_range=True)
    if tf is not None:
        out = transform.warp(out, tf, order=0, preserve_range=True)
    out = out.clip(min=0)
    out = recenter_im(out)
    out = crop_to(out, ref.shape[1], ref.shape[0])
    return out


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
