# Code for importing and processing the various forms of listmode data from the iQID camera
# It's not the cleanest code, but please contact Robin Peter if you need any assistance or find errors

import os
import numpy as np
import cv2
import glob

from scipy.optimize import curve_fit
from skimage import io, transform, filters
import matplotlib.pyplot as plt
import matplotlib.cm

from datetime import datetime
from tqdm import tqdm, trange

import ipywidgets as widgets
import functools

from iqid import helper


def exponential(x, a, thalf):
    return a*np.exp(-np.log(2)/thalf*x)


class ClusterData:

    def __init__(self, file_name, c_area_thresh=15, makedir=False, ftype='processed_lm'):
        self.file_name = file_name
        self.ftype = ftype
        base = os.path.basename(os.path.normpath(self.file_name))
        self.savedir = os.path.join(
            os.path.dirname(self.file_name), base[:-4] + '_Analysis')
        if ftype not in ["processed_lm", "offset_lm", "clusters"]:
            raise (
                TypeError, "File format not specified: processed_lm, offset_lm, clusters")
        # self.NUM_NAN_DATA = 50
        self.c_area_thresh = c_area_thresh

        if makedir:
            os.makedirs(self.savedir, exist_ok=True)

    def init_header(self):
        HEADER = np.fromfile(self.file_name, dtype=np.int32, count=100)
        self.HEADER_SIZE = HEADER[0]
        self.XDIM = HEADER[1]
        self.YDIM = HEADER[2]

        if self.ftype == 'processed_lm':
            self.NUM_DATA_ELEMENTS = 14
        elif self.ftype == 'offset_lm':
            self.NUM_DATA_ELEMENTS = 6
        elif self.ftype == 'clusters':
            # "Cropped Listmode File" is the other name for cluster image file type.
            self.cluster_radius = HEADER[20]
            self.cluster_imsize = 2*self.cluster_radius + \
                1  # e.g. 10-px radius = 21x21 clusters
            self.NUM_DATA_ELEMENTS = 8 + 2 * (self.cluster_imsize**2)
        else:
            raise (
                TypeError, "File format not specified: processed_lm, offset_lm, clusters")

        return (self.HEADER_SIZE, self.XDIM, self.YDIM)

    def set_process_params(self, binfac, ROI_area_thresh, t_binsize, t_half):
        self.binfac = binfac
        self.ROI_area_thresh = ROI_area_thresh
        self.t_binsize = t_binsize
        self.t_half = t_half
        return ()

    def parse_acqt(self):
        self.info_file = helper.natural_sort(
            glob.glob(
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(self.file_name)),
                    "*Acquisition_Info*.txt")))[-2]

        with open(self.info_file, "r") as file:
            lines = file.readlines()[:2]
            # Format: "Date: Weekday, Month XX, 20XX"
            date_string = lines[0].rstrip('\n')
            time_string = lines[1].rstrip('\n')  # Format: "Time: XX:YY:ZZ AM"

        date_string = date_string[6:]
        time_string = time_string[6:]  # XX:YY:ZZ AM
        combined_string = date_string + ',' + time_string
        time_format = "%A, %B %d, %Y,%I:%M:%S %p"

        datetime_result = datetime.strptime(combined_string, time_format)
        self.acq_time = datetime_result

        return datetime_result

    def load_cluster_data(self, event_fx=1, dtype=np.float64):
        self.init_header()
        file_size_bytes = os.path.getsize(self.file_name)

        if dtype == np.float32 or dtype == np.int32:
            byteSize = 4
            byteFac = 1
        else:
            byteSize = 8
            # if loading whole thing as f64 (8), header will only take up 50=100//2 values instead of 100
            byteFac = 2
        NUM_CLUSTERS = np.floor(
            (file_size_bytes - 4*self.HEADER_SIZE) / (byteSize*self.NUM_DATA_ELEMENTS))

        # for very large data, you may only want to load the first 10% , e.g.
        NUM_LOAD = int(event_fx * NUM_CLUSTERS)

        unshaped_data = np.fromfile(
            self.file_name, dtype=dtype, count=self.HEADER_SIZE // byteFac + int(NUM_LOAD*self.NUM_DATA_ELEMENTS))
        data = unshaped_data[self.HEADER_SIZE // byteFac:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_LOAD), order='F')
        return (data)

    def load_raws(self, cluster_size=10):

        # find associated raw listmode file in folder if there is one
        # ensure that there is only one in the directory, or it could grab the wrong one
        rlistmode = glob.glob(os.path.join(
            os.path.dirname(self.file_name), '*Cropped_Raw_Listmode.dat'))

        if not rlistmode:
            raise Exception(
                "No cropped raw listmode file found in directory. " +
                "Check that it is located in the same location: \n{}".format(self.file_name))
        else:
            rlistmode = rlistmode[0]

        byteSize = 4  # INTs rather than DOUBLEs
        file_size_bytes = os.path.getsize(rlistmode)

        # in all cases, HEADER SIZE *should* be the same between cprlm and crlm
        NUM_CLUSTERS = np.floor(
            (file_size_bytes - 4*self.HEADER_SIZE) / (byteSize*self.NUM_DATA_ELEMENTS))

        unshaped_data = np.fromfile(
            rlistmode, dtype=np.int32, count=self.HEADER_SIZE + int(NUM_CLUSTERS*self.NUM_DATA_ELEMENTS))

        data = unshaped_data[self.HEADER_SIZE:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_CLUSTERS), order='F')

        return data

    def init_metadata(self, data):

        # parses the loaded data into relevant arrays
        # data formats for each file type are provided in the iQID header info.
        # offset_lm: previously "listmode_frames"

        if self.ftype == 'processed_lm':
            frame_num = data[0, :]
            time_ms = data[1, :]
            sum_cluster_signal = data[2, :]
            cluster_area = data[3, :]
            yC_global = data[4, :]
            xC_global = data[5, :]
            var_y = data[6, :]
            var_x = data[7, :]
            covar_xy = data[8, :]
            eccentricity = data[9, :]
            skew_y = data[10, :]
            skew_x = data[11, :]
            kurt_y = data[12, :]
            kurt_x = data[13, :]

            self.cluster_area = cluster_area
            self.cluster_sum = sum_cluster_signal

            self.xC = xC_global
            self.yC = yC_global
            self.f = frame_num
            self.time_ms = time_ms

            # initialize the offset variables with None
            self.miss = None
            self.nevents_per_frame = None
            self.offset_frame_time = None

            return time_ms, cluster_area, xC_global, yC_global, frame_num
            # in the future, update this to be the same order as offset lm

        elif self.ftype == 'offset_lm':
            frame_num = data[0, :]
            time_ms = data[1, :]
            n_miss = data[2, :]
            n_cluster = data[3, :]
            pix = data[4, :]
            cam_temp_10K = data[5, :]

            self.f = frame_num
            self.time_ms = time_ms

            self.miss = n_miss
            self.nevents_per_frame = n_cluster
            self.offset_frame_time = time_ms
            return frame_num, time_ms, n_miss, n_cluster, pix, cam_temp_10K

        elif self.ftype == 'clusters':
            a = self.cluster_imsize

            raw_imgs = data[:a**2, :].T
            # fil_imgs = data[a**2: 2*a**2, :] # filtered/binarized version of image

            frame_num = data[2*a**2, :]  # Frame number (size INT)
            # yC (row) centroid coordinate (size INT) of the cropped cluster
            yC = data[2*a**2 + 1, :]
            # xC (column) centroid coordinate (size INT) of the cropped cluster
            xC = data[2*a**2 + 2, :]
            # Sum of the filtered cluster signal within the cropped sub-image (size INT)
            cim_sum = data[2*a**2 + 5, :]
            # Time elapsed since start of acquisition (size Unsigned INT)
            time_ms = data[2*a**2 + 6, :]
            # Number of pixels (area) in the cluster (size INT)
            cluster_area = data[2*a**2 + 7, :]

            self.xC = xC
            self.yC = yC
            self.f = frame_num
            self.cim_sum = cim_sum
            self.cluster_area = cluster_area
            self.cluster_sum = sum_cluster_signal
            self.raws = raw_imgs
            self.time_ms = time_ms

            return frame_num, time_ms, xC, yC, raw_imgs, cim_sum, cluster_area

        else:
            print('Accepted types: process_lm, offset_lm, clusters')

    def image(self):
        cim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(self.xC), desc='Building image...'):
            cim[self.yC[i].astype(int), self.xC[i].astype(int)] += 1
        return cim

    def image_from_xy(self, x, y):
        x = np.round(x)
        y = np.round(y)

        cim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(x), desc='Building image...'):
            cim[y[i].astype(int), x[i].astype(int)] += 1
        return cim

    def image_from_listmode(self, subpx=1):
        self.init_header()
        data = self.load_cluster_data()
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        xC_filtered = xC[np.logical_and(
            xC > 0, self.cluster_area > self.c_area_thresh)]
        yC_filtered = yC[np.logical_and(
            yC > 0, self.cluster_area > self.c_area_thresh)]

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
        cluster_image = np.zeros((int(subpx*self.YDIM), int(subpx*self.XDIM)))
        for i in range(len(yC_good)):
            cluster_image[yC_good[i], xC_good[i]] += 1

        # sort into arrays and get associated temporal information
        time_s = time_ms[self.cluster_area > self.c_area_thresh]/1e3
        time_s = time_s[cluster_bool]

        # the listmode coordinates that are actually used in the cluster image
        self.xC = xC_good
        self.yC = yC_good
        self.t_s = time_s

        return (cluster_image)

    def image_from_big_listmode(self, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
        # only subpx=1
        self.init_header()

        xlim = np.array(xlim)
        ylim = np.array(ylim)

        if xlim[1] is None:
            xlim[1] = self.XDIM
        if ylim[1] is None:
            ylim[1] = self.YDIM

        if xlim[1] > self.XDIM:
            raise ValueError(
                'X-limit ({}, {}) exceeds dimensions of acquisition ({})'.format(xlim[0], xlim[1], self.XDIM))
        if ylim[1] > self.YDIM:
            raise ValueError(
                'Y-limit ({}, {}) exceeds dimensions of acquisition ({})'.format(ylim[0], ylim[1], self.YDIM))

        data = self.load_cluster_data(event_fx=event_fx)
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        # more efficient version of image_from_listmode for only subpx=1
        cluster_bool = ((xC > xlim[0]) * (yC > ylim[0])
                        * (xC < xlim[1]) * (yC < ylim[1])
                        * np.isfinite(xC) * np.isfinite(yC)
                        * (self.cluster_area > self.c_area_thresh))
        xC = np.round(xC[cluster_bool]).astype(int)
        yC = np.round(yC[cluster_bool]).astype(int)
        t_s = time_ms[cluster_bool] / 1e3

        # build spatial image (no temporal information)
        cluster_image = np.zeros((self.YDIM, self.XDIM))
        for i in range(len(yC)):
            cluster_image[yC[i], xC[i]] += 1

        self.xC = xC
        self.yC = yC
        self.t_s = t_s

        return (cluster_image)

    def apply_selection(self, selection_bool):
        try:
            self.xC = self.xC[selection_bool]
            self.yC = self.yC[selection_bool]
            self.f = self.f[selection_bool]
        except Exception as e:
            print(e)  # do this catch so that the in-place doesn't happen if any fail
        return (self.xC, self.yC, self.f)

    def get_subset(self, selection_bool):
        # basically, not doing it in place.
        # makes a subset class with each partial dataset.
        # might need caution for big data...
        subset_data = Subset(self.file_name,
                             c_area_thresh=self.c_area_thresh,
                             makedir=False,
                             ftype=self.ftype)
        subset_data.HEADER_SIZE = self.HEADER_SIZE
        subset_data.XDIM = self.XDIM
        subset_data.YDIM = self.YDIM

        subset_data.xC = self.xC[selection_bool]
        subset_data.yC = self.yC[selection_bool]
        subset_data.f = self.f[selection_bool]
        subset_data.time_ms = self.time_ms[selection_bool]
        subset_data.cluster_area = self.cluster_area[selection_bool]
        subset_data.cluster_sum = self.cluster_sum[selection_bool]

        if self.ftype == 'clusters':
            subset_data.raws = self.raws[selection_bool]
            subset_data.cim_sum = self.cim_sum[selection_bool]
            subset_data.cluster_area = self.cluster_area[selection_bool]
            subset_data.cluster_imsize = self.cluster_imsize
            subset_data.cluster_sum = self.cluster_sum[selection_bool]

        return subset_data

    def get_mean_n(self, vis=False):
        # finds and sets mean number of events per frame
        # requires load of the offset file
        if self.nevents_per_frame is not None:
            n = self.nevents_per_frame
            self.mean_n = np.mean(n)
            if vis:
                hist_N, _, _ = plt.hist(
                    n, bins=np.arange(12), edgecolor='white')
                plt.xlabel('number of events per frame')
                plt.ylabel('number of frames')
                plt.axvline(np.mean(n), color='gray', ls='--')
                plt.title('mean events/frame = {:.2f}'.format(np.mean(n)))
                plt.show()

            return self.mean_n
        else:
            print('Offset file not loaded. TODO implement fallback using LM file.')

    def estimate_missed_timestamps(self, verbose=True):
        # estimate the approximate timestamp of missed frames using offset file
        # this allows for correction in the time histogram for activity quantification
        # do not use this for rigorous timing coincidence
        # Just does 1 event per frame - should check the n-value to see if accurate

        if verbose:
            def iter_fn(integer):
                return trange(integer, desc="Estimating missed timestamps")
        else:
            iter_fn = range

        if self.miss is not None:
            m = self.miss
            t = self.offset_frame_time

            num_new_missed = np.diff(m)
            missed_events_time = np.array([])
            for i in iter_fn(len(num_new_missed)):
                if num_new_missed[i] > 0:
                    missed_events_time = np.append(missed_events_time,
                                                   np.repeat(t[i], num_new_missed[i]))
            return missed_events_time
        else:
            print('Offset file not loaded. TODO implement fallback using LM file.')

    def filter_singles(self, fmax, vis=False, save=False, filter_time=False):
        vals, N = np.unique(self.f, return_counts=True)
        single_fnums = vals[N == 1]

        focc = len(vals)  # occupied frames
        fempty = fmax - focc

        if vis:
            pct_singles = len(single_fnums) / fmax * 100
            hist_N, _, _ = plt.hist(N, bins=np.arange(12), edgecolor='white')
            plt.xlabel('number of events per frame')
            plt.ylabel('number of frames')
            plt.title('{:.1f} % of frames are single-event,\n{} ({:.1f}%) are empty'.format(pct_singles,
                                                                                            fempty, fempty/fmax * 100))  # much improved singles rate
            plt.show()

        # NOTE:: DOES NOT INCLUDE CLUSTER AREA THRESHOLD FILTRATION
        cluster_bool = ((self.xC > 0) * (self.yC > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)  # finite value check
                        * np.isin(self.f, single_fnums))   # singles only

        self.xC = self.xC[cluster_bool].astype(int)
        self.yC = self.yC[cluster_bool].astype(int)
        self.f = self.f[cluster_bool]

        if filter_time:
            self.time_ms = self.time_ms[cluster_bool]
            self.cluster_area = self.cluster_area[cluster_bool]

        return (self.xC, self.yC, self.f, self.time_ms)

    def set_coin_params(self, fps, t0_dt, TS_ROI, binfac=1, verbose=True):
        # TS_ROI: external from IDM data, code this in later
        # t0_dt: amount of time to delay IDM timestamp, i.e. IDM STARTED AFTER IQID
        # must adjust to be NEGATIVE if IDM STARTED BEFORE IQID
        self.fps = fps
        self.exp = 1 / fps

        a = self.f * self.exp
        b = TS_ROI + t0_dt
        s_bins = np.arange(np.max(self.f) + 2, step=binfac) * self.exp
        # reason for +2: 1 so that it goes up to max f, +1 so that it includes the time "during" that frame

        # check that we aren't accidentally stacking two acquisitions
        # not sure how to replicate this , but it happens possibly when reset is not properly done between IDM LM acquisitions
        assert np.all(
            a[:-1] <= a[1:]), "iQID array not monotonically increasing"
        assert np.all(
            b[:-1] <= b[1:]), "IDM array not monotonically increasing"

        if verbose:
            print('{:.2f} ms per frame'.format(self.exp * 1e3))
            print('iQID first 10:', a[:10])
            print('IDM first 10:', b[:10])
            print('Bins', s_bins[:10])

        self.a = a
        self.b = b
        self.s_bins = s_bins

        return a, b, s_bins

    def find_coin(self, singles=False, return_hist=False, verbose=True):
        if verbose:
            print('Generating iQID hist...')
        iq_n, _ = np.histogram(self.a, bins=self.s_bins)
        if verbose:
            print('Generating IDM hist...')
        idm_n, _ = np.histogram(self.b, bins=self.s_bins)

        coin = np.logical_and(iq_n, idm_n)  # if both are positive (non-zero)
        multi = np.sum(coin)
        if verbose:
            print('{} multi events found'.format(multi))

        if singles == 'iqid':
            # discard bins for which multiple iqid events id'd
            iq_n = (iq_n == 1)
        elif singles == 'idm':
            idm_n = (idm_n == 1)  # discard bins for which multiple gammas id'd
        elif singles == 'both':
            iq_n = (iq_n == 1)
            idm_n = (idm_n == 1)
        else:
            pass

        coin = np.logical_and(iq_n, idm_n)  # if both are positive (non-zero)
        if verbose:
            print('Selected {} coincident bins'.format(np.sum(coin)))

        if return_hist:
            return coin, iq_n, idm_n
        else:
            return coin

    def image_from_coin(self, coin=None, verbose=True, binfac=1, **kwargs):
        if coin is None:
            coin = self.find_coin(**kwargs)

        # recover the frame number from the histogram
        f_bins = np.arange(np.max(self.f) + 2, step=binfac)
        good_f = f_bins[:-1][coin]
        good_events = np.isin(self.f, good_f)

        if verbose:
            print('Found {:.2e} "good" events.'.format(np.sum(good_events)))

        x_good = self.xC[good_events]
        y_good = self.yC[good_events]

        # manually rebuild spatial image
        nim = np.zeros((int(self.YDIM), int(self.XDIM)))

        if verbose:
            for i in trange(len(y_good), desc='Building selected image...'):
                nim[y_good[i], x_good[i]] += 1
        else:
            for i in range(len(y_good)):
                nim[y_good[i], x_good[i]] += 1

        return x_good, y_good, nim

    def check_elements(self, a, idx, x):
        if a[idx] == x:
            return idx
        else:
            idx += 1
            return self.check_elements(a, idx, x)

    def correct_frames(self, a1, a2, m1):
        '''
        a1: offset frames (N)
        a2: listmode frames (M, M>N)
        m1: missed frames from offset (N)

        return: 
        m2: missed frames in listmode (M)
        '''
        running_idx = 0
        m2 = np.zeros_like(a2)

        for i in trange(len(a2), desc='Assigning missed frames'):
            # check that current offset element is equal to current full element
            # if not, increment offset element
            try:
                running_idx = self.check_elements(a1, running_idx, a2[i])
                m2[i] = m1[running_idx]
            except IndexError:  # sometimes listmode data has values that offset doesn't have
                # I am not certain why this is the case, maybe old-version iqid bug
                print('aborting : list-mode data contains higher frames than offset file. Remainder will be filled with previous m-values')
                # fill remaining values with same number of missed as previous
                m2[i:] = m2[i-1]
                break

        return m2

    def correct_listmode(self, offset_frames, missed_frames, vis=True):
        # use offset file to correct missed frames
        a1 = offset_frames
        a2 = self.f
        m1 = missed_frames
        m2 = self.correct_frames(a1, a2, m1)

        u2, idx2 = np.unique(a2, return_index=True)
        u1, idx1, _ = np.intersect1d(a1, u2, return_indices=True)

        if vis:
            plt.plot(m1[idx1])
            plt.plot(m2[idx2])
            plt.xlabel('Index')
            plt.ylabel('Number of missed frames')
            plt.show()
            plt.close()

        corr_lm = m2 + self.f

        self.f = corr_lm  # overwrites listmode frames in place, be careful to save separate variable if needed for some reason
        return corr_lm

    def get_coms(self):
        '''
        Get center of mass (CoM) of a cluster image.
        Uses Otsu threshold to binarize the pixel image before evaluating CoM.
        '''
        if self.ftype != 'clusters':
            raise TypeError(
                "Data set is not of type 'cluster': {}".format(self.ftype))

        coms = np.zeros((len(self.raws), 2))
        for i in trange(len(self.raws), desc='Finding CoMs...'):
            im = self.raws[i].reshape(self.cluster_imsize, self.cluster_imsize)
            thresh = filters.threshold_otsu(im)
            bin_im = (im > thresh).astype(int)
            cx, cy = helper.com(bin_im)
            coms[i, :] = cx, cy

        self.coms = coms
        return coms

    def filter_coms(self, size=7):
        '''
        Discard events for which the center of mass is outside of a certain centered region.
        Only applicable to cluster data set.
        Size parameter determines size of the internal square boundary for CoM discrimination.
        '''

        if self.ftype != 'clusters':
            raise TypeError(
                "Data set is not of type 'cluster': {}".format(self.ftype))

        coms = self.get_coms()

        x0 = self.cluster_imsize//2 - size//2
        good_coms = (coms[:, 0] > x0) * (coms[:, 0] < x0 + size) * \
            (coms[:, 1] > x0) * (coms[:, 1] < x0 + size)

        self.com_bool = good_coms  # can be used for debugging
        return self.get_subset(good_coms)

    def set_contour_params(self, gauss=15, thresh=0):
        self.gauss = gauss
        self.thresh = thresh
        return ()

    def prep_contour(self, im, gauss=15, thresh=0):
        binned_image = helper.bin_ndarray(
            im, (np.array(np.shape(im))/self.binfac).astype(int), operation='sum')
        mask = binned_image > thresh
        bin_im = mask.astype('uint8')
        prep_im = cv2.GaussianBlur(bin_im, (gauss, gauss), 0)
        return (prep_im)

    def get_contours(self, im):
        prep_im = self.prep_contour(
            im=im, gauss=self.gauss, thresh=self.thresh)
        contours, _ = cv2.findContours(
            prep_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        good_contours = [
            self.binfac*c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
        return (good_contours)

    def get_contours_from_dir(self, mask_dir, fformat='png'):
        # generate contours from manual masks
        c = []
        file_names = glob.glob(os.path.join(mask_dir, "*."+fformat))
        helper.natural_sort(file_names)
        # print('Loading {} masks...'.format(len(file_names)))
        for i in range(len(file_names)):
            im = io.imread(file_names[i])
            contours, _ = cv2.findContours(
                im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            good_contours = [
                self.binfac*c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
            c.append(good_contours[0])
        # print('Returning {} contours'.format(len(c)))
        return c

    def get_maskstack(self, im):
        xdim, ydim = np.shape(im)
        maskstack = np.zeros((len(self.contours), xdim, ydim))
        for i in range(len(self.contours)):
            mask = np.zeros_like(im)
            cv2.drawContours(mask, self.contours, i, (255, 255, 255), -1)
            mask_norm = (mask/255).astype(np.uint8)
            maskstack[i, :, :] = mask_norm
        return (maskstack)

    def events_in_ROI(self, maskstack):
        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = mask[self.yC.astype(int), self.xC.astype(int)] * \
                (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)
            ROI_array_bool[i, :] = inROI_bool
        return (ROI_array_bool.astype(bool))

    def get_ROIs(self, pad=10):
        ROI_array = np.zeros((len(self.contours), 4))
        for i in range(len(self.contours)):
            cnt = self.contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            ROI_array[i, :] = x-pad//2, y-pad//2, w+pad, h+pad
        return (ROI_array.astype(int))

    def setup_ROIs(self, im, mode='auto', **kwargs):
        if mode == 'auto':
            self.contours = self.get_contours(im)
        elif mode == 'manual':
            self.contours = self.get_contours_from_dir(**kwargs)
        elif mode == 'single':
            pass  # make this provide one single ROI that is just the whole image
        else:
            raise TypeError('Mode must be "auto", "manual", or "single".')
        self.maskstack = self.get_maskstack(im)
        self.ROIbool = self.events_in_ROI(self.maskstack)
        self.ROIlists = self.get_ROIs()

    def fitHist(self, t, n, func=exponential, p0=None, tol=0.05):
        if p0 is None:
            p0 = [1, self.t_half]
            # changed from default p0= [1, 9.92*24*3600] for Ac-225
        thalf = p0[1]
        popt, pcov = curve_fit(f=func, xdata=t, ydata=n,
                               p0=p0, sigma=np.maximum(
                                   np.ones_like(n), np.sqrt(n)),
                               bounds=([0, thalf * (1-tol)], [np.inf, thalf * (1 + tol)]))
        param_std = np.sqrt(np.diag(pcov))
        res = n - func(t, *popt)
        chisq = np.sum(res**2/func(t, *popt))
        chisqn = chisq/len(n)
        return (popt, pcov, param_std, res, chisq, chisqn)

    def fitROI(self, temporal_array, func=exponential, p0=None, binsize=1000, tol=0.05):

        if p0 is None:
            p0 = [1, self.t_half]

        nbins = np.round(temporal_array[-1]/binsize)
        count, bins = np.histogram(temporal_array, np.arange(0, nbins)*binsize)
        timepoints = bins[:-1] + binsize/2
        popt, pcov, param_std, res, chisq, chisqn = self.fitHist(
            timepoints, count, func=func, p0=p0, tol=tol)

        return (count, timepoints, popt, pcov, param_std, res, chisq, chisqn)

    def get_imslice(self, im, idx):
        x, y, w, h = self.ROIlists[idx, :]
        imslice = im[y:y+h, x:x+w] * self.maskstack[idx, y:y+h, x:x+w]
        return imslice

    # def test_countours(self):
    #     good_contours = []
    #     for i in range(len(man_maskstack)):
    #         contours, _ = cv2.findContours(man_maskstack[i, :, :],
    #                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         gcontours = [c for c in contours if cv2.contourArea(
    #             c) > self.ROI_area_thresh]
    #         good_contours.append(gcontours)

    def get_manual_maskstack(self):
        base = os.path.basename(os.path.normpath(self.file_name))
        maskdir = os.path.join(self.savedir, 'manual_masks')
        fileList = glob.glob(os.path.join(maskdir, '*.tif'))
        man_maskstack = np.array(io.ImageCollection(fileList))

        good_contours = []

        for i in range(len(man_maskstack)):
            contours, _ = cv2.findContours(man_maskstack[i, :, :],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            gcontours = [c for c in contours if cv2.contourArea(
                c) > self.ROI_area_thresh]
            good_contours.append(gcontours)

        maskstack = np.zeros((len(good_contours), self.YDIM, self.XDIM))
        for i in range(len(good_contours)):
            mask = np.zeros((self.YDIM, self.XDIM))
            cv2.drawContours(mask, good_contours[i], -1, (255, 255, 255), -1)
            mask_norm = (mask/255).astype(np.uint8)
            maskstack[i, :, :] = mask_norm

        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = mask[self.yC.astype(int), self.xC.astype(int)] * \
                (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC)
            ROI_array_bool[i, :] = inROI_bool

        pad = 10
        ROI_array = np.zeros((len(good_contours), 4))
        for i in range(len(good_contours)):
            x, y, w, h = cv2.boundingRect(good_contours[i][0])
            ROI_array[i, :] = x-pad//2, y-pad//2, w+pad, h+pad

        self.ROIlists = ROI_array.astype(int)
        self.ROIbool = ROI_array_bool.astype(bool)
        self.contours = good_contours
        self.maskstack = maskstack
        return (maskstack)

    def save_manual_mask(self, mask):
        # base = os.path.basename(os.path.normpath(self.file_name))
        # newdir = os.path.join(self.file_name, '..', base[:-4] + '_Analysis')
        os.makedirs(self.savedir, exist_ok=True)
        plt.imsave(os.path.join(self.savedir, 'manual_mask_preview.png'), mask)
        io.imsave(os.path.join(self.savedir, 'manual_mask.tif'),
                  mask, plugin='tifffile')
        print('Manual mask saved to:')
        print(self.savedir)

    def fitROIs(self, im, vis=True, corr=0,
                tol=0.05, tstart=0, tcond=None,
                idxs='all',
                save=False, savemasks=False, save_ts=False):

        if idxs == 'all':
            iterlist = range(len(self.ROIlists))
            # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison if idxs == 'all':
        else:
            iterlist = idxs

        base = os.path.basename(os.path.normpath(self.file_name))

        if vis:
            f, ax = plt.subplots(nrows=len(iterlist), ncols=2,
                                 figsize=(8, 4*len(iterlist)))
            if save:
                os.makedirs(self.savedir, exist_ok=True)
                pvdir = os.path.join(self.savedir, 'mBq_image_previews')
                os.makedirs(pvdir, exist_ok=True)

        if save:
            os.makedirs(self.savedir, exist_ok=True)
            imdir = os.path.join(self.savedir, 'mBq_images')
            os.makedirs(imdir, exist_ok=True)

        all_A0 = np.zeros(len(iterlist))
        all_dA0 = np.zeros(len(iterlist))
        for i in tqdm(range(len(iterlist)), desc='Getting ROIs'):
            # cut out the ROI from the full image
            imslice = self.get_imslice(im, iterlist[i])  # cts

            t_data = self.t_s[self.ROIbool[iterlist[i]]]

            # use the events (with temporal data) inside the ROI to fit whole slice activity
            count, ts, popt, _, pstd, _, _, chisqn = self.fitROI(t_data,
                                                                 binsize=self.t_binsize,
                                                                 tol=tol)

            if save_ts:
                os.makedirs(self.savedir, exist_ok=True)
                newdir = os.path.join(self.savedir, 'ROI_tstamps')
                os.makedirs(newdir, exist_ok=True)
                np.savetxt(os.path.join(
                    newdir, 'tstamps_{}.txt'.format(i)), t_data)

            if tstart > 0:
                if vis:
                    ax[i, 1].axvspan(tstart, ts[-1], color='gray', alpha=0.3)
                gidx = ts > tstart
                popt, _, pstd, _, _, chisqn = self.fitHist(
                    ts[gidx], count[gidx])
            elif tcond is not None:
                # this is extremely hacky. beware
                # input should be (start1, stop1, start2, stop2) in s
                if len(tcond) != 4 and len(tcond) != 2:
                    raise ValueError(
                        'tcond input should be (start1, stop1) OR (start1, stop1, start2, stop2)')

                if len(tcond) == 2:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1],
                                         color='gray', alpha=0.3)

                    gidx = (ts > tcond[0])*(ts < tcond[1])
                    popt, _, pstd, _, _, chisqn = self.fitHist(
                        ts[gidx], count[gidx])

                elif len(tcond) == 4:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1],
                                         color='gray', alpha=0.3)
                        ax[i, 1].axvspan(tcond[2], tcond[3],
                                         color='gray', alpha=0.3)

                    gidx = (ts > tcond[0])*(ts < tcond[1]) + \
                        (ts > tcond[2])*(ts < tcond[3])
                    popt, _, pstd, _, _, chisqn = self.fitHist(
                        ts[gidx], count[gidx])

            # activity (mBq) at the start of imaging
            Ai = popt[0] * 1000/self.t_binsize
            dAi = pstd[0] * 1000/self.t_binsize

            # activity correction to the time of sacrifice
            A0 = Ai * np.exp(np.log(2) * corr / self.t_half)  # mBq
            A0_Bq = A0 / 1e3
            A0_Ci = A0_Bq / 3.7e10
            all_dA0[i] = dAi * \
                np.exp(np.log(2) * corr / self.t_half) / 1e3 / 3.7e10
            all_A0[i] = A0_Ci

            # total A over whole image * pxcts / totalcts = A in each px
            aslice = A0 * imslice / np.sum(imslice)

            if vis:
                ax[i, 0].imshow(aslice, cmap='gray',
                                vmax=0.15 * np.max(aslice))
                ax[i, 0].axis('off')
                xdummy = np.linspace(0, max(ts))
                ax[i, 1].scatter(ts, count)
                ax[i, 1].errorbar(ts, count, np.sqrt(
                    count), ls='none', capsize=3)
                ax[i, 1].plot(
                    xdummy, exponential(xdummy, *popt))
                ax[i, 1].set_xlabel('time (s)')
                ax[i, 1].set_ylabel(
                    'Counts in {} s bins'.format(self.t_binsize))
                ax[i, 1].set_title('Ai={:.0f} pm {:.0f} mBq\n'.format(Ai, dAi) +
                                   '$\\chi^2/N$: {:.3f}\nA0: {:.0f} mBq ({:.3f} nCi)'.format(
                                       chisqn, A0, A0_Ci*1e9),
                                   y=-0.5, fontsize=12)

            if save:
                if save == 'slicer3d':
                    aslice = aslice.astype(np.float32)
                if vis:
                    plt.imsave(os.path.join(pvdir, 'mBq_preview_{}.png'.format(i)),
                               aslice, cmap='gray', vmax=0.15 * np.max(aslice))
                io.imsave(os.path.join(imdir, 'mBq_{}.tif'.format(i)),
                          aslice, plugin='tifffile', photometric='minisblack', check_contrast=False)

        if savemasks:
            base = os.path.basename(os.path.normpath(self.file_name))
            maskdir = os.path.join(self.savedir, 'full_masks')
            os.makedirs(maskdir, exist_ok=True)
            for i in range(len(iterlist)):
                io.imsave(os.path.join(maskdir, 'mask_{}.png'.format(i)),
                          (255*self.maskstack[iterlist[i], :, :]).astype(np.uint8), check_contrast=False)

            maskdir = os.path.join(self.savedir, 'ROI_masks')
            os.makedirs(maskdir, exist_ok=True)
            for i in range(len(iterlist)):
                x, y, w, h = self.ROIlists[iterlist[i], :]
                mask = self.maskstack[iterlist[i], y:y+h, x:x+w]
                io.imsave(os.path.join(maskdir, 'mask_{}.png'.format(i)),
                          (255*mask).astype(np.uint8))
            print('masks saved to:', maskdir)

        if vis:
            return (all_A0, all_dA0, f, ax)
        else:
            return (all_A0, all_dA0)

    def plot_vis_masks(self, im):
        for i in range(len(self.maskstack)):
            plt.figure(figsize=(8, 6))
            # plt.imshow((1-weight) * im + weight*self.maskstack[i,:,:], cmap='gray')
            # masked_array = np.ma.array(im, mask=self.maskstack[i,:,:])

            helper_im = np.ma.array(im, mask=self.maskstack[i, :, :])
            _, helper_im = cv2.imencode('.png', helper_im.astype(np.uint8))
            helper_im = helper_im.tobytes()
            cmap = matplotlib.cm.inferno
            plt.imshow(helper_im, interpolation='nearest', cmap=cmap)
            plt.axis('off')
            plt.title(i)
            plt.show()
            plt.close()

    def widget_labelling(self, im, vmax=1, deg=0, IMG_WIDTH=200, IMG_HEIGHT=200, COLS=4):
        ROWS = int(np.ceil(len(self.maskstack)/COLS))

        rows = []
        for row in tqdm(range(ROWS), desc="Building widget"):
            cols = []
            for col in range(COLS):
                try:
                    idx = row * COLS + col

                    helper_im = np.copy(im)
                    midx = (self.maskstack[idx, :, :] == 1)
                    helper_im[midx] = np.max(helper_im)
                    helper_im = transform.rotate(helper_im, deg)
                    helper_im *= 255.0/(vmax * helper_im.max())
                    helper_im = cv2.imencode('.png', helper_im)[1].tobytes()

                    image = widgets.Image(
                        value=helper_im, width=IMG_WIDTH, height=IMG_HEIGHT
                    )

                    labelbox = widgets.BoundedIntText(
                        value=idx+1,
                        min=-1,
                        max=len(self.maskstack)+1,
                        step=1,
                        disabled=False,
                        layout=widgets.Layout(flex='1 1 0%', width='auto')
                    )

                    nobutton = widgets.ToggleButton(
                        value=False,
                        description='discard',
                        button_style='',
                        layout=widgets.Layout(flex='1 1 0%', width='auto'))

                    # Create a vertical layout box, image above the button
                    buttonbox = widgets.HBox([labelbox, nobutton],
                                             layout=widgets.Layout(display='flex',
                                                                   align_items='stretch',
                                                                   width='100%'))
                    box = widgets.VBox([image, buttonbox])
                    cols.append(box)
                except IndexError:
                    break  # for when # of images is not divisible by 4

            # Create a horizontal layout box, grouping all the columns together
            rows.append(widgets.HBox(cols))

        # Create a vertical layout box, grouping all the rows together
        result = widgets.VBox(rows)
        submit_button = widgets.Button(
            description="Generate Indices Array",
            layout=widgets.Layout(width='100%'))

        output = widgets.Output()

        def button_click(rs_, b):
            generate_idxs(rs_)

        def text_yn(arr_, darr_, wdgt):
            yn = wdgt.value
            with output:
                yn = helper.get_yn(yn)
                if yn:
                    arr_ = arr_[np.logical_not(darr_)]
                    new_arr = np.arange(1, len(arr_) + 1)[np.argsort(arr_)]
                    print('Inclusion:', repr(np.logical_not(darr_).astype(int)))
                    print('Order:', repr(new_arr.astype(int)))
                else:
                    print(repr(arr_.astype(int)))

        def generate_idxs(row_widgt):
            # obtain value of all of the manual buttons
            manual_idxs = np.zeros(len(self.maskstack))
            manual_discard = np.zeros(len(self.maskstack))
            counter = 0  # could be more elegant, possibly fix in future with grid and flatten
            for i in range(ROWS):
                single_row = row_widgt[i]
                for j in range(COLS):
                    try:
                        single_box = single_row.children[j]
                        manual_idxs[counter] = single_box.children[1].children[0].value
                        manual_discard[counter] = single_box.children[1].children[1].value
                        counter += 1
                    except IndexError:
                        break

            idx_arr = manual_idxs.flatten()
            dis_arr = manual_discard.flatten()
            out_arr = idx_arr[np.logical_not(dis_arr)]

            with output:
                err_mesg = "Duplicate indices detected. Please ensure each positive index appears exactly once."
                assert len(out_arr) == len(np.unique(out_arr)), err_mesg

                err_mesg = "Skipped indices detected:\n{}".format(
                    np.sort(out_arr))
                if np.array_equal(np.sort(out_arr), np.arange(1, len(out_arr) + 1)):
                    print('Inclusion:', repr(np.logical_not(dis_arr).astype(int)))
                    print('Order:', repr(out_arr.astype(int) - 1))
                else:
                    print(err_mesg)
                    prompt = widgets.Text(
                        value='yes',
                        placeholder='yes/no',
                        description="Fix?",
                        style={'description_width': 'initial'}
                    )
                    with output:
                        display(prompt)
                    prompt.on_submit(functools.partial(
                        text_yn, idx_arr, dis_arr))

        submit_button.on_click(functools.partial(button_click, rows))

        display(result)
        display(submit_button, output)


class Subset(ClusterData):
    def __init__(self, file_name, c_area_thresh, makedir, ftype):
        super().__init__(file_name, c_area_thresh, makedir, ftype)
