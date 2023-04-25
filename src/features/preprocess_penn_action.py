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
    
def _preprocess_keypoints(label: Dict, blurr_sigma: float, noise_scalar: int):
    """
    Preprocesses keypoints of a single video.
    
    Parameters
    ----------
    label : Dict
        Meta-information about the current video
        
    Returns
    -------
    processed_input_heatmaps : List[torch.Tensor]
        List where the i'th entry contains the input 
        heatmaps of the i'th frame. 
        
    processed_output_heatmaps : List[torch.Tensor]
        List where the i'th entry contains the output 
        heatmaps of the i'th frame. 
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
    #keypoints[:, :, 1] = TARGET_HEIGHT - 1 - keypoints[:, :, 1]
    
    # Function for translating keypoints for translating
    # Penn action keypoint-index to ClimbAlong keypoint-index
    translate = lambda i: CLIMBALONG_KEYPOINTS[PENN_ACTION_KEYPOINTS[i]]
    
    # List for storing input heatmaps
    processed_input_heatmaps = [torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH) for _ in range(keypoints.shape[0])]
    
    # List for storing output heatmaps
    processed_output_heatmaps = [torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH) for _ in range(keypoints.shape[0])]
    
    # List for storing output heatmaps with varying stds for gaussian blur
    processed_input_heatmaps_mixed_std = [torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH) for _ in range(keypoints.shape[0])]
    
    # Values for randomly shifting keypoints
    shifts = np.round(np.random.normal(0, noise_scalar * NOISE_STD, size=(len(keypoints), 2))).astype(int)
    
    # Looping through each frame
    for i, frame_keypoints in enumerate(keypoints):
        
        # Looping through each keypoint of the frame
        for j, keypoint in enumerate(frame_keypoints):            
            if not (0 <= keypoint[0] < TARGET_WIDTH and 0 <= keypoint[1] < TARGET_HEIGHT):
                continue
            
            # Value for randomly shifting keypoints
            shift = shifts[j]
            
            # Translating keypoint-index to correct index
            j = translate(j)            
            
            # Randomly shifting input keypoints
            shifted_keypoint = np.clip(keypoint + shift, [0, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1])
            
            # Making heatmap from keypoint
            input_heatmap = turn_keypoint_to_featuremap(shifted_keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
            input_heatmap_mixed_std = turn_keypoint_to_featuremap(shifted_keypoint, (TARGET_HEIGHT, TARGET_WIDTH), blurr_sigma=blurr_sigma)
            output_heatmap = turn_keypoint_to_featuremap(keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
            
            # Inserting data
            processed_input_heatmaps[i][j] = input_heatmap
            processed_input_heatmaps_mixed_std[i][j] = input_heatmap_mixed_std
            processed_output_heatmaps[i][j] = output_heatmap
            
    return processed_input_heatmaps, processed_output_heatmaps, processed_input_heatmaps_mixed_std
            
def preprocess(noise_scalar: int):
    """
    Function for preprocessing all the relevant Penn Action samples.
    """
    
    # List of sample-names
    samples = os.listdir(PENN_ACTION_RAW_KEYPOINTS_PATH)
    
    sigmas = np.random.choice(GAUSSIAN_STDS, size=len(samples))
    
    # Looping through each video
    for i, video_id in tqdm(enumerate(samples), desc="Processing videos", disable=False, total=len(samples)):
        
        # Removing file type
        video_id = video_id[:-4]
        
        # Path for storing the keypoints of this video
        keypoints_input_path = OVERALL_DATA_FOLDER(noise_scalar) + SUBFOLDERS["x"] + video_id + "/"
        keypoints_input_mixed_std_path = OVERALL_DATA_FOLDER(noise_scalar) + SUBFOLDERS["x_std"] + video_id + "/"
        keypoints_output_path = OVERALL_DATA_FOLDER(noise_scalar) + SUBFOLDERS["y"] + video_id + "/"
        
        # Path for storing meta-info of this sample
        label_path = PENN_ACTION_RAW_KEYPOINTS_PATH + video_id + ".mat"
        
        # Loading the label and the action
        label = scipy.io.loadmat(label_path)
        action = label["action"][0]
        
        if action not in RELEVANT_ACTIONS:
            continue
        
        # Making folder for storing keypoints of current video
        make_dir(keypoints_input_path)
        make_dir(keypoints_input_mixed_std_path)
        make_dir(keypoints_output_path)
        
        # Process the keypoints
        processed_input_heatmaps, processed_output_heatmaps, processed_input_heatmaps_mixed_std = _preprocess_keypoints(label, blurr_sigma=sigmas[i], noise_scalar=noise_scalar)
        
        # Storing keypoints as individual frames
        for frame_number in range(len(processed_input_heatmaps)):
            
            # Extracting preprocessed heatmaps
            input_heatmaps = processed_input_heatmaps[frame_number]
            input_heatmaps_mixed_std = processed_input_heatmaps_mixed_std[frame_number]
            output_heatmaps = processed_output_heatmaps[frame_number]
            
            # Path for storing stuff
            input_heatmaps_storing_path = keypoints_input_path + str(frame_number) + ".pt"
            input_heatmaps_mixed_std_storing_path = keypoints_input_mixed_std_path + str(frame_number) + ".pt"
            output_heatmaps_storing_path = keypoints_output_path + str(frame_number) + ".pt"
            
            # Saving keypoints of frame as tensor
            torch.save(input_heatmaps, input_heatmaps_storing_path)
            torch.save(input_heatmaps_mixed_std, input_heatmaps_mixed_std_storing_path)
            torch.save(output_heatmaps, output_heatmaps_storing_path)
        
if __name__ == "__main__":
    preprocess()