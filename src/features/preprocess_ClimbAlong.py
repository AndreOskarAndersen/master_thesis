import os
import zipfile
import numpy as np
import torch
from utils import make_dir, remove_file
from global_variables import *

def _remove_zip_files():
    """
    Function for deleting files not to be used.
    """
    
    print("Removing ClimbAlong input-zip-file...")
    remove_file(CA_INPUT_ZIP_PATH)
    print("ClimbAlong input-zip-file has been removed")
    
    print("Removing ClimbAlong groundtruth-zip-file...")
    remove_file(CA_GT_ZIP_PATH)
    print("ClimbAlong groundtruth-zip-file has been removed")

def _extract_dataset():
    """
    Function for extracting the ClimbAlong-dataset.
    """
    
    try:
        print("Extracting Climbalong input-dataset...")
        with zipfile.ZipFile(CA_INPUT_ZIP_PATH, 'r') as f:
            f.extractall(CA_INPUT_DIR_PATH)
        print("Extraction of ClimbALong input-dataset done")   
    except:
        print(f"{CA_INPUT_DIR_PATH} does not exist")
        
    try:
        print("Extracting Climbalong groundtruth-dataset...")
        with zipfile.ZipFile(CA_GT_ZIP_PATH, 'r') as f:
            f.extractall(CA_GT_DIR_PATH)
        print("Extraction of ClimbALong groundtruth-dataset done")
    except:
        print(f"{CA_GT_ZIP_PATH} does not exist")
    
def _clean_groundtruth_data():
    
    # Extensions to use
    extensions = {"video": ".mp4", "keypoint": ".npz"}
    
    # List of all of the samples
    sample_names = list(map(lambda x: x.split(".")[0], os.listdir(CA_INPUT_DIR_PATH)))
    
    
def _preprocess_input_heatmaps(heatmaps):
    """
    Processes a single input-frame of heatmaps
    """
    
    # All elements below 0 are set to 0
    heatmaps[heatmaps < 0] = 0
    
    # We make sure that each heatmap sums up to 1
    heatmaps /= np.sum(heatmaps, axis=(1, 2))[:, np.newaxis, np.newaxis]
    
    return heatmaps

def _preprocess_input_bboxes(bbox):
    """
    Processes the bbox of a single input-frame
    """
    x_1 = bbox[0]
    x_2 = bbox[2]
    
    y_1 = bbox[1]
    y_2 = bbox[2]
    
    bbox = np.array([[y_1, x_1], [y_1, x_2], [y_2, x_1], [y_2, x_2]]).reshape((4, 2))
    bbox = np.round(bbox).astype(int)
    
    return bbox
    
def _preprocess_data():
    """
    Function for processing the ClimbAlong dataset
    """
    
    # List of all of the videos
    video_names = os.listdir(CA_INPUT_DIR_PATH)
    
    for video_name in video_names:
        input_video = np.load(CA_INPUT_DIR_PATH + video_name)
        input_video_heatmaps = input_video["heatmaps"]
        input_video_bboxes = input_video["bboxes"]
        num_frames = len(input_video_heatmaps)
        
        for frame in range(num_frames):
            input_frame_heatmaps = _preprocess_input_heatmaps(input_video_heatmaps[frame])
            input_frame_bboxes = _preprocess_input_bboxes(input_video_bboxes[frame])
            break
        break             

def preprocess():
    # Making dir of where to store the extracted data.
    make_dir(CA_DATA_PATH)
    
    # Extracting the data
    _extract_dataset()
    
    # Removing unnecessary zip-files
    #_remove_zip_files()
    
    # Cleaning up groundtruth dataset
    #_clean_groundtruth_data()
    
    # Preprocessing the keypoints
    #_preprocess_data()
    
    pass


if __name__ == "__main__":
    preprocess()
    