import os
import numpy as np
import torch
from tqdm.auto import tqdm
from skimage.filters import gaussian
from utils import make_dir
from global_variables import CA_PROCESSED_PATH, CA_PROCESSED_SUBDIRS, CA_RAW_PATH, CA_RAW_SUBDIRS, TARGET_HEIGHT, TARGET_WIDTH, NUM_KEYPOINTS

def _make_dirs():
    """
    Function for making directories to be used.
    """
    
    for subdir in CA_PROCESSED_SUBDIRS.values():
        make_dir(CA_PROCESSED_PATH + subdir)
        
def _make_target_heatmaps(xs: np.ndarray, ys: np.ndarray):
    """
    Function for making the target heatmaps given the x- and y-coordinates of a sample.
    
    Parameters
    ----------
    xs : np.ndarray
        x-coordinates of the sample
        
    ys : np.ndarray
        y-coordinates of the sample
        
    Returns
    -------
    The heatmaps
    """
    
    # Preallocating the memory of the heatmaps
    heatmaps = torch.zeros((NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH))
    
    # Looping through each coordinate
    for i, (x, y) in enumerate(zip(xs, ys)):
        
        # Clipping the coordinate to the bounding-box
        # in case they are not already in it
        x = np.clip(x, 0, TARGET_WIDTH - 1)
        y = np.clip(y, 0, TARGET_HEIGHT - 1)
        
        # Making the heatmap
        heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH))
        heatmap[y, x] = 1
        heatmaps[i] = torch.from_numpy(gaussian(heatmap, sigma=1))
        
    return heatmaps

def _make_input_heatmaps(frame_input_heatmaps: np.ndarray):
    """
    Function for preprocessing the input heatmaps of a single frame.
    
    Parameters
    ----------
    frame_input_heatmaps : np.ndarray
        Heatmaps of a single frame
        
    Returns
    -------
    frame_input_heatmaps : toch.Tensor
        Preprocessed heatmaps of a single frame
    """
    
    frame_input_heatmaps[frame_input_heatmaps < 0] = 0
    s = np.sum(frame_input_heatmaps, axis=(1, 2))
    s[s == 0] = 1
    s = s[:, np.newaxis, np.newaxis]
    frame_input_heatmaps = frame_input_heatmaps/s
    frame_input_heatmaps = torch.from_numpy(frame_input_heatmaps)
    
    return frame_input_heatmaps
        
def _preprocess_sample(sample_name: str, input_storing_path: str, target_storing_path: str):
    """
    Function for preprocessing a single sample
    
    Parameters
    ----------
    sample_name : str
        Name of the sample to preprocess
        
    input_storing_path : str
        Path of where to store the processed input data
        
    target_storing_path : str
        Path of where to store the processed target path
    """
    
    # Path of the raw input and target data
    input_raw_path = CA_RAW_PATH + CA_RAW_SUBDIRS["x"] + sample_name + ".npz"
    target_raw_path = CA_RAW_PATH + CA_RAW_SUBDIRS["y"] + sample_name + ".npz"
    
    # Loading the raw input and target data
    input_data = np.load(input_raw_path)
    target_data = np.load(target_raw_path)
    
    # Loading the input bounding-boxes, heatmaps and keypoints
    input_bboxes = input_data["bboxes"]
    input_heatmaps = input_data["heatmaps"]
    target_keypoints = target_data["keypoints"]
    num_annoated_keypoints = np.any(target_keypoints[0].squeeze() > [0, 0, 0], axis=1).sum()
    if num_annoated_keypoints < 23:
        return
    
    # Making folders
    make_dir(input_storing_path)
    make_dir(target_storing_path)
    
    # Looping through each frame
    for i, (frame_input_bbox, frame_input_heatmaps, frame_target_keypoints) in enumerate(zip(input_bboxes, input_heatmaps, target_keypoints)):
        frame_target_keypoints = frame_target_keypoints.squeeze()
        
        assert frame_input_bbox.shape == (4,)
        assert frame_input_heatmaps.shape == (25, 56, 56)
        assert frame_target_keypoints.shape == (25, 3)
        
        # Paths of where to store the processed input and target data of this frame
        input_frame_storing_path = input_storing_path + str(i) + ".pt"
        target_frame_storing_path = target_storing_path + str(i) + ".pt"
        
        # If the input does not contain a bounding-box, we skip it
        if frame_input_bbox[2] - frame_input_bbox[0] == 0 or frame_input_bbox[3] - frame_input_bbox[1] == 0:
            break
        
        # Shitfing the target keypoints to the correct range, such that they fit inside the bounding-box
        xs = np.round((frame_target_keypoints[:, 0] - frame_input_bbox[0]) * TARGET_WIDTH/(frame_input_bbox[2] - frame_input_bbox[0])).astype(int)
        ys = np.round((frame_target_keypoints[:, 1] - frame_input_bbox[1]) * TARGET_HEIGHT/(frame_input_bbox[3] - frame_input_bbox[1])).astype(int)
        
        # Making the target heatmaps
        frame_target_heatmaps = _make_target_heatmaps(xs, ys)
        
        # Making the input heatmaps
        frame_input_heatmaps = _make_input_heatmaps(frame_input_heatmaps)
        
        # Saving the preprocessed data
        torch.save(frame_input_heatmaps, input_frame_storing_path)
        torch.save(frame_target_heatmaps, target_frame_storing_path)
        
def _preprocess_samples():
    """
    Function for preprocessing the ClimbAlong-sampels
    """
    
    # Names of the samples
    sample_names = list(map(lambda sample_name: sample_name.split(".")[0], os.listdir(CA_RAW_PATH + CA_RAW_SUBDIRS["x"])))
    
    # Iterating through each sample
    for sample_name in tqdm(sample_names, disable=False):
        
        # Folder of where to store the processed input
        input_storing_path = CA_PROCESSED_PATH + CA_PROCESSED_SUBDIRS["x"] + sample_name + "/"
        
        # Folder of where to store the processed output
        target_storing_path = CA_PROCESSED_PATH + CA_PROCESSED_SUBDIRS["y"] + sample_name + "/"
        
        # Processing the current sample
        _preprocess_sample(sample_name, input_storing_path, target_storing_path)

def preprocess():
    """
    Main entrypoint for preprocessing the ClimbAlong dataset.
    """
    
    # Making directories to be used
    _make_dirs()
    
    # Preprocesses the samples
    _preprocess_samples()

if __name__ == "__main__":
    preprocess()