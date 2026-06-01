import numpy as np
import matplotlib.pyplot as plt
import cv2
from iqid import align
from iqid import helper as iq
from skimage.measure import block_reduce


def bin_he(he_im, he_um_px, iq_um_px, thresh=100, mode='binary'):

    if mode == 'binary':
        cseg = CellSeg('', invert=True)  # variable does nothing... should fix
        gray = cv2.cvtColor(he_im, cv2.COLOR_RGB2GRAY)
        _, binary = cseg.binarize(gray, thresh, vis=False)

        return align.bin_bin(binary, he_um_px, iq_um_px)

    elif mode == 'color':
        y, x, _ = he_im.shape
        x_um = he_um_px * x
        y_um = he_um_px * y

        xdim = int(x_um // iq_um_px)  # iQ px = um / (um/px_iq)
        # maximum number of HE pixels that will be used
        xtemp = int(np.floor(xdim * iq_um_px / he_um_px))
        xtt = align.myround(xtemp, xdim)

        ydim = int(y_um // iq_um_px)
        ytemp = int(np.floor(ydim * iq_um_px / he_um_px))
        ytt = align.myround(ytemp, ydim)

        cropped = align.crop_to(he_im, int(xtt), int(ytt))

        yc, xc, _ = np.shape(cropped)
        yfac, xfac = int(yc/ydim), int(xc/xdim)

        h = block_reduce(cropped, block_size=(yfac, xfac, 1), func=np.mean)
        h = np.round(h).astype(int)
        return h

    else:
        print('Err: "mode" must be "binary" or "color". (Input:{})'.format(mode))


class CellSeg:

    def __init__(self, file_name, invert=False):
        self.file_name = file_name
        self.invert = invert

    def visu_patch(self, im, x, y, w, cmap='gray'):
        '''
        x, y: starting location of patch
        w: patch width (square)'''

        patch = self.get_patch(im, x, y, w)
        f, ax = plt.subplots(1, 2, gridspec_kw={
                             'width_ratios': [3, 1]}, figsize=(15, 5))
        ax[0].imshow(im, cmap=cmap)
        ax[1].imshow(patch, cmap=cmap)
        plt.tight_layout()
        plt.show()
        plt.close()
        return (patch)

    def get_patch(self, im, x, y, w):
        self.patch = im[y:y+w, x:x+w]
        self.xmin = x
        self.ymin = y
        self.w = w
        return (self.patch)

    def binarize(self, im, th='otsu', vis=False):
        if self.invert:
            algo = cv2.THRESH_BINARY_INV
        else:
            algo = cv2.THRESH_BINARY

        if th == 'otsu':
            th, binary = cv2.threshold(im, 0, 255, algo + cv2.THRESH_OTSU)
        elif th > 0:
            th, binary = cv2.threshold(im, th, 255, algo)
        else:
            raise TypeError("Invalid threshold picked")

        if vis:
            plt.hist(im.ravel(), bins=100)
            plt.axvline(th, color='k', ls=':', label='threshold')
            plt.xlabel('gray value')
            plt.ylabel('count')
            plt.legend()
            plt.show()
            align.plot_compare(im, binary, 'Image', 'Binarized Image',
                               cmap1='gray', cmap2='gray')

        self.binary = binary
        return (th, binary)

    def watershed(self, im, param=0.35, ksize=3, iter=3):
        kernel = np.ones((ksize, ksize), np.uint8)
        sure_bg = cv2.dilate(self.binary, kernel, iterations=iter)
        dist_transform = cv2.distanceTransform(self.binary, cv2.DIST_L2, ksize)
        _, sure_fg = cv2.threshold(
            dist_transform, param*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Add one to all labels so that sure background is not 0, but 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(im, markers)
        self.markers = markers
        return (markers)

    def plot_wshed(self, im, markers, x=0, y=0, w=200, color=[255, 0, 0]):
        seg_img = np.copy(im)
        seg_img[markers == -1] = color

        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(seg_img)
        ax[1].imshow(seg_img[y:y+w, x:x+w])
        plt.show()
        plt.close()
        return (seg_img)

    def get_stats(self, seg_img, color=(255, 0, 0)):
        th = cv2.inRange(seg_img, color, color).astype(np.uint8)
        th = cv2.bitwise_not(th)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th, connectivity=4)
        num_labels -= 1
        sizes = stats[1:, -1]
        centroids = centroids[1:, :]
        return (num_labels, labels, stats, sizes, centroids)

    def vis_centroids(self, clean_img, seg_img, color=(255, 0, 0), s=1):
        plt.figure(figsize=(5, 5))
        _, _, _, _, centroids = self.get_stats(seg_img, color)
        plt.imshow(clean_img)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    color='red', alpha=1, s=s)
        plt.show()
        plt.close()
