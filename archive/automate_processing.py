import os
import json
import logging
import numpy as np
from iqid.process_object import ClusterData

# Configure logging
logging.basicConfig(filename='automate_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_and_process_listmode_data(file_name, c_area_thresh=15, makedir=False, ftype='processed_lm'):
    try:
        cluster_data = ClusterData(file_name, c_area_thresh, makedir, ftype)
        cluster_data.init_header()
        data = cluster_data.load_cluster_data()
        time_ms, cluster_area, xC_global, yC_global, frame_num = cluster_data.init_metadata(data)
        return cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num
    except Exception as e:
        logging.error("Failed to load and process listmode data: %s", str(e))
        raise

def filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half):
    try:
        cluster_data.set_process_params(binfac, ROI_area_thresh, t_binsize, t_half)
        cluster_data.get_mean_n()
        cluster_data.estimate_missed_timestamps()
        return cluster_data
    except Exception as e:
        logging.error("Failed to filter, correct, and analyze data: %s", str(e))
        raise

def generate_spatial_images(cluster_data, subpx=1):
    try:
        cluster_image = cluster_data.image_from_listmode(subpx)
        return cluster_image
    except Exception as e:
        logging.error("Failed to generate spatial images: %s", str(e))
        raise

def generate_temporal_information(cluster_data, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
    try:
        cluster_image = cluster_data.image_from_big_listmode(event_fx, xlim, ylim)
        return cluster_image
    except Exception as e:
        logging.error("Failed to generate temporal information: %s", str(e))
        raise

def detect_contours_extract_ROIs(cluster_data, im, gauss=15, thresh=0):
    try:
        cluster_data.set_contour_params(gauss, thresh)
        contours = cluster_data.get_contours(im)
        ROIs = cluster_data.get_ROIs()
        return contours, ROIs
    except Exception as e:
        logging.error("Failed to detect contours and extract ROIs: %s", str(e))
        raise

def save_processed_data(cluster_data, output_dir):
    try:
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
    except Exception as e:
        logging.error("Failed to save processed data: %s", str(e))
        raise

def main():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)

        file_name = config['automate_processing']['file_name']
        output_dir = config['automate_processing']['output_dir']
        c_area_thresh = config['automate_processing']['c_area_thresh']
        makedir = config['automate_processing']['makedir']
        ftype = config['automate_processing']['ftype']
        binfac = config['automate_processing']['binfac']
        ROI_area_thresh = config['automate_processing']['ROI_area_thresh']
        t_binsize = config['automate_processing']['t_binsize']
        t_half = config['automate_processing']['t_half']
        subpx = config['automate_processing']['subpx']
        event_fx = config['automate_processing']['event_fx']
        xlim = config['automate_processing']['xlim']
        ylim = config['automate_processing']['ylim']
        gauss = config['automate_processing']['gauss']
        thresh = config['automate_processing']['thresh']

        cluster_data, time_ms, cluster_area, xC_global, yC_global, frame_num = load_and_process_listmode_data(file_name, c_area_thresh, makedir, ftype)
        cluster_data = filter_correct_analyze_data(cluster_data, binfac, ROI_area_thresh, t_binsize, t_half)
        cluster_image = generate_spatial_images(cluster_data, subpx)
        temporal_image = generate_temporal_information(cluster_data, event_fx, xlim, ylim)
        contours, ROIs = detect_contours_extract_ROIs(cluster_data, cluster_image, gauss, thresh)
        save_processed_data(cluster_data, output_dir)
    except Exception as e:
        logging.error("Failed to complete main processing: %s", str(e))
        raise

if __name__ == "__main__":
    main()
