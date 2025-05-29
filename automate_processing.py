import os
import numpy as np
from iqid.process_object import ClusterData

def load_and_process_listmode_data(file_name, c_area_thresh=15, makedir=False, ftype='processed_lm'):
    cluster_data = ClusterData(file_name, c_area_thresh, makedir, ftype)
    cluster_data.init_header()
    data = cluster_data.load_cluster_data()
    time_ms, cluster_area, xC_global, yC_global, frame_num = cluster_data.init_metadata(data)
    return cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num

def filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half):
    cluster_data.set_process_params(binfac, ROI_area_thresh, t_binsize, t_half)
    cluster_data.get_mean_n()
    cluster_data.estimate_missed_timestamps()
    return cluster_data

def generate_spatial_images(cluster_data, subpx=1):
    cluster_image = cluster_data.image_from_listmode(subpx)
    return cluster_image

def generate_temporal_information(cluster_data, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
    cluster_image = cluster_data.image_from_big_listmode(event_fx, xlim, ylim)
    return cluster_image

def detect_contours_extract_ROIs(cluster_data, im, gauss=15, thresh=0):
    cluster_data.set_contour_params(gauss, thresh)
    contours = cluster_data.get_contours(im)
    ROIs = cluster_data.get_ROIs()
    return contours, ROIs

def save_processed_data(cluster_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'xC.npy'), cluster_data.xC)
    np.save(os.path.join(output_dir, 'yC.npy'), cluster_data.yC)
    np.save(os.path.join(output_dir, 'f.npy'), cluster_data.f)
    np.save(os.path.join(output_dir, 'time_ms.npy'), cluster_data.time_ms)
    if cluster_data.ftype == 'clusters':
        np.save(os.path.join(output_dir, 'raws.npy'), cluster_data.raws)
        np.save(os.path.join(output_dir, 'cim_sum.npy'), cluster_data.cim_sum)
        np.save(os.path.join(output_dir, 'cim_px.npy'), cluster_data.cim_px)

def main(file_name, output_dir, c_area_thresh=15, makedir=False, ftype='processed_lm', binfac=1, ROI_area_thresh=100, t_binsize=1000, t_half=3600, subpx=1, event_fx=0.1, xlim=(0, None), ylim=(0, None), gauss=15, thresh=0):
    cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num = load_and_process_listmode_data(file_name, c_area_thresh, makedir, ftype)
    cluster_data = filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half)
    cluster_image = generate_spatial_images(cluster_data, subpx)
    temporal_image = generate_temporal_information(cluster_data, event_fx, xlim, ylim)
    contours, ROIs = detect_contours_extract_ROIs(cluster_data, cluster_image, gauss, thresh)
    save_processed_data(cluster_data, output_dir)

if __name__ == "__main__":
    file_name = "path/to/your/listmode_data.dat"
    output_dir = "path/to/save/processed_data"
    main(file_name, output_dir)
