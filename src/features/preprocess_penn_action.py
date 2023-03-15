import numpy as np
import os
import scipy.io
import torch
from typing import Dict
from tqdm.auto import tqdm
from utils import make_dir, turn_keypoint_to_featuremap
from global_variables import *

def _load_bboxes(keypoints: np.array):
    """
    Extracts the tightest possible bounding-box from a set of keypoints.
    
    Parameters
    ----------
    keypoints : np.array
        3D array of the frames and keypoints.
        
    Returns
        x_mins : np.array
            Minimum x-coordinates of the keypoints of each frame
            
        y_mins : np.array
            Minimum y-coordinates of the keypoints of each frame
            
        x_maxs : np.array
            Maximum x-coordinates of the keypoints of each frame
            
        y_maxs : np.array
            Maximum y-coordinates of the keypoints of each frame
    """
    
    x_mins = np.min(keypoints[:, :, 0], axis=1).astype(float)
    y_mins = np.min(keypoints[:, :, 1], axis=1).astype(float)
    x_maxs = np.max(keypoints[:, :, 0], axis=1).astype(float)
    y_maxs = np.max(keypoints[:, :, 1], axis=1).astype(float)
    
    return x_mins, y_mins, x_maxs, y_maxs
    
def _preprocess_keypoints(label: Dict):
    """
    Preprocesses keypoints of a single video.
    
    Parameters
    ----------
    label : Dict
        Meta-information about the current video
        
    Returns
    -------
    processed_heatmaps : List[torch.Tensor]
        List where the i'th entry contains the heatmaps of the
        i'th frame. 
    """
    
    keypoints = np.dstack((label["x"], label["y"])).astype(float)
    
    # Extracting cornors of bboxes
    x_mins, y_mins, x_maxs, y_maxs = _load_bboxes(keypoints)
    
    # Making bboxes a square, by expanding the shortest side
    widths = x_maxs - x_mins
    heights = y_maxs - y_mins
    diffs = np.abs(heights - widths)
    expand_factors = diffs/2
    
    x_mins[widths < heights] -= expand_factors[widths < heights]
    x_maxs[widths < heights] += expand_factors[widths < heights]
    
    y_mins[widths > heights] -= expand_factors[widths > heights]
    y_maxs[widths > heights] += expand_factors[widths > heights]
    
    widths = x_maxs - x_mins
    heights = y_maxs - y_mins
    
    # Expanding sides by 10%
    expand_factors = 0.1 * widths * 0.5
    x_mins -= expand_factors 
    x_maxs += expand_factors 
    y_mins -= expand_factors 
    y_maxs += expand_factors 
    
    # Shifts keypoints, corresponding to such that the upper left koordinate
    # of the bbox has coordinates (0, 0)
    x_maxs -= x_mins
    y_maxs -= y_mins
        
    keypoints[:, :, 0] -= x_mins.reshape((-1, 1))
    keypoints[:, :, 1] -= y_mins.reshape((-1, 1))
    
    # Rescaling keypoints to the correct range    
    rescale_width = (TARGET_WIDTH - 1) / np.round(x_maxs)
    rescale_height = (TARGET_HEIGHT - 1) / np.round(y_maxs)

    keypoints[:, :, 0] *= rescale_width.reshape((-1, 1))
    keypoints[:, :, 1] *= rescale_height.reshape((-1, 1))
    
    # Rounding to nearest integer
    keypoints = np.round(keypoints).astype(int)
    
    # Flipping the keypoints horizontally
    keypoints[:, :, 1] = TARGET_HEIGHT - 1 - keypoints[:, :, 1]
    
    # Function for translating keypoints for translating
    # Penn action keypoint-index to ClimbAlong keypoint-index
    translate = lambda i: CLIMBALONG_KEYPOINTS[PENN_ACTION_KEYPOINTS[i]]
    
    # List for storing final heatmaps
    processed_heatmaps = [torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH).bool() for _ in range(keypoints.shape[0])]
    
    # Looping through each frame
    for i, frame_keypoints in enumerate(keypoints):
        
        # Looping through each keypoint of the frame
        for j, keypoint in enumerate(frame_keypoints):            
            if not (0 <= keypoint[0] < TARGET_WIDTH and 0 <= keypoint[1] < TARGET_HEIGHT):
                continue
            
            # Translating keypoint-index to correct index
            j = translate(j)
            
            # Making heatmap from keypoint
            heatmap = turn_keypoint_to_featuremap(keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
            
            # Inserting data
            processed_heatmaps[i][j] = heatmap
            
    return processed_heatmaps
            
def preprocess():
    """
    Function for preprocessing all the relevant Penn Action samples.
    """
    
    # List of sample-names
    samples = os.listdir(PENN_ACTION_RAW_KEYPOINTS_PATH)
    
    # Looping through each video
    for video_id in tqdm(samples, desc="Processing videos", disable=False):
        
        # Removing file type
        video_id = video_id[:-4]
        
        # Path for storing the keypoints of this video
        keypoints_path = OVERALL_DATA_FOLDER + video_id + "/"
        
        # Path for storing meta-info of this sample
        label_path = PENN_ACTION_RAW_KEYPOINTS_PATH + video_id + ".mat"
        
        # Loading the label and the action
        label = scipy.io.loadmat(label_path)
        action = label["action"][0]
        
        if action not in RELEVANT_ACTIONS:
            continue
        
        # Making folder for storing keypoints of current video
        make_dir(keypoints_path)
        
        # Process the keypoints
        preprocessed_keypoints = _preprocess_keypoints(label)
        
        # Storing keypoints as individual frames
        for frame_number, peprocessed_keypoint in enumerate(preprocessed_keypoints):
        
            # Path for storing stuff
            keypoints_storing_path = keypoints_path + str(frame_number) + ".pt"
            
            # Saving keypoints of frame as tensor
            torch.save(peprocessed_keypoint, keypoints_storing_path)
        
if __name__ == "__main__":
    preprocess()