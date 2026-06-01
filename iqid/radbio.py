"""Radiobiology functions: TCP (Tumor Control Probability) calculation."""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from iqid import align


def fill_erode(c, kernel, n=1):
    """Erode a cell mask while preserving holes (non-cell spaces)."""
    cmask = (c > 0).astype(np.uint8)

    contour, _ = cv2.findContours(cmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(cmask, [cnt], 0, 255, -1)
    cmask_erode = cv2.erode(cmask, kernel) * (c > 0)

    if n > 1:
        cmask_erode = fill_erode(cmask_erode, kernel, n - 1)

    return cmask_erode


def get_tcp(dr, c, dfac, alpha, kerode, nerode=1, vis=False, hist_thresh=True, contour_thresh=False):
    """Compute Tumor Control Probability from a dose-rate image.

    Parameters
    ----------
    dr : 2D array
        Dose-rate image (mGy/h per pixel).
    c : 2D array
        Cell density map (cells per pixel).
    dfac : float
        Dose-rate to dose conversion factor; dose (Gy) = dr * dfac * 1e-3.
    alpha : float
        Linear-quadratic alpha parameter (Gy^-1).
    kerode : ndarray
        Erosion kernel (cv2 structuring element).
    nerode : int
        Number of erosion iterations.
    vis : bool
        If True, return underdosed mask and VCP map in addition to TCP.
    hist_thresh : bool
        Mask background via histogram threshold (default True).
    contour_thresh : bool
        Mask background via contour binary mask (alternative to hist_thresh).

    Returns
    -------
    TCP : float
        Scalar tumor control probability.
    underdosed, VCPi_inMask : arrays (only when vis=True)
    """
    iq_dose = dr * dfac * 1e-3  # Gy

    cmask = fill_erode(c, kerode, n=nerode)

    if hist_thresh:
        try:
            n, bins = np.histogram(iq_dose.ravel(), bins='auto')
        except IndexError:
            print('Numpy floating point error during binning. Defaulting to 100 bins.\n'
                  'See https://github.com/numpy/numpy/issues/12435')
            n, bins = np.histogram(iq_dose.ravel(), bins=100)
        except MemoryError:
            print('Memory error (auto-histogram too large). Defaulting to 100 bins.')
            n, bins = np.histogram(iq_dose.ravel(), bins=100)
        iq_mask = (iq_dose > bins[1])
    elif contour_thresh:
        iq_mask = align.binary_mask(dr, finagle=0)
        iq_mask = cv2.GaussianBlur(iq_mask.astype(np.uint8), (5, 5), 0).astype(bool)
    else:
        iq_mask = np.ones_like(cmask).astype(bool)

    idxs = cmask.astype(bool) * iq_mask

    Dv = iq_dose[idxs]
    Nc = c[idxs]
    SFi = np.exp(-alpha * Dv)
    Ns = SFi * Nc
    VCPi = np.exp(-Ns)
    TCP = np.product(VCPi)

    if vis:
        Dv = iq_dose * idxs
        SFi = np.exp(-alpha * Dv)
        Ns = SFi * c
        VCPi = np.exp(-Ns)
        underdosed = ((VCPi * idxs) > 0) * ((VCPi * idxs) < 0.95)
        VCPi_inMask = VCPi * (idxs > 0)
        plt.imshow(underdosed)
        plt.show()
        return TCP, underdosed, VCPi_inMask

    return TCP


def get_tcp_fromdose(d, c, alpha, kerode, nerode=1, vis=False):
    """Compute TCP directly from a dose image (skipping dose-rate conversion).

    Parameters
    ----------
    d : 2D array
        Dose image (Gy per pixel).
    c : 2D array
        Cell density map (cells per pixel).
    alpha : float
        Linear-quadratic alpha parameter (Gy^-1).
    kerode : ndarray
        Erosion kernel.
    nerode : int
        Number of erosion iterations.
    vis : bool
        If True, display underdosed mask.
    """
    iq_dose = d

    cmask = fill_erode(c, kerode, n=nerode)

    n, bins = np.histogram(iq_dose.ravel(), bins='auto')
    idxs = cmask.astype(bool) * (iq_dose > bins[1])

    Dv = iq_dose[idxs]
    Nc = c[idxs]
    SFi = np.exp(-alpha * Dv)
    Ns = SFi * Nc
    VCPi = np.exp(-Ns)
    TCP = np.product(VCPi)

    if vis:
        Dv = iq_dose * idxs
        SFi = np.exp(-alpha * Dv)
        Ns = SFi * c
        VCPi = np.exp(-Ns)
        underdosed = ((VCPi * idxs) > 0) * ((VCPi * idxs) < 0.95)
        plt.imshow(underdosed)
        plt.show()

    return TCP
