import os
import numpy as np
import pandas as pd
import torch
import itertools
from tqdm import tqdm
from typing import Tuple, Dict, List
from utils import make_dir, turn_keypoint_to_featuremap
from global_variables import *

def groupc(arr: list):
    """
    Function for grouping sequential numbers into intervals.
    
    Parameters
    ----------
    arr : list
        list of numbers to group
        
    Note:
    -----
    Inspired by the following url
    https://www.geeksforgeeks.org/python-consecutive-elements-grouping-in-list/
            
    Returns
    -------
    elements : list
        list of string-intervals
        
    indicies : list
        list of tuples intervals-indicies
    """
    
    indicies = []
    elements = []
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) - 1 and arr[j + 1] == arr[j] + 1:
            j += 1
            
        indicies.append((i, j))
        elements.append(str(arr[i]) + "_" + str(arr[j]) + "/")
        
        i = j + 1
        
    return elements, indicies

def _load_keypoints(year: str, video_id: str):
    """
    Loads the automatically and manually annotated BRACE keypoints,
    as well as creates the bboxes.
    
    Parameters
    ----------
    year : str
        The year of the recording of the video.
        
    video_id : str
        The ID of the video.
        
    Returns
    -------
    video_annotations : Dict[str, Dict[str, Tuple[List[float]]]]
        Dict with length of the amount of clips in the video. The key 
        is the interval of frames of the clip, the corresponding value
        is another dict, where the key is the frame-number and the value
        is a tuple, where the first element is the bbox of the frame
        and the second element is the keypoints of the frame.
    """
    
    video_annotations = {}
    
    # loading automatic directories
    path = RAW_BRACE_PATH + RAW_KEYPOINT_FOLDERS["automatic"] + RAW_KEYPOINTS_SUBFOLDERS["automatic"] + year + "/" + video_id + "/"
    keypoints_listdir = os.listdir(path)
    
    # Looping through all of the automatic annotations
    for keypoint_file in keypoints_listdir:
        # Path to keypoints
        keypoint_path = path + keypoint_file
        
        # Reading annotations
        annotations = pd.read_json(keypoint_path)
        
        # Columns used for naming frames.
        keys = list(annotations.columns)
        keys = list(map(lambda x: int(x[-10:-4].lstrip("0")), keys))
        
        # Removing "score"-attribute
        clip_keypoints = annotations.loc["keypoints"]
        clip_keypoints = clip_keypoints.apply(lambda x: list(map(lambda y: [y[0], y[1]], x)))
        
        # Casting each row to torch.Tensor
        clip_keypoints = list(clip_keypoints.to_numpy())
        clip_keypoints = list(map(lambda x: torch.tensor(x), clip_keypoints))
        
        # Making bboxes
        clip_bboxes = [[clip_keypoint[:, 0].min().item(), 
                        clip_keypoint[:, 1].min().item(), 
                        clip_keypoint[:, 0].max().item(), 
                        clip_keypoint[:, 1].max().item()] 
                       for clip_keypoint in clip_keypoints]
        
        # Storing keypoints, bboxes and their corresponding frame-number
        interval_elements, interval_indices = groupc(keys)
        for interval_element, interval_index in zip(interval_elements, interval_indices):            
            interval_keys = list(map(str, keys[interval_index[0]:interval_index[1] + 1]))
            interval_bboxes = clip_bboxes[interval_index[0]:interval_index[1] + 1]
            interval_keypoints = clip_keypoints[interval_index[0]:interval_index[1] + 1]
            interval_annotations = dict(zip(interval_keys, zip(interval_bboxes, interval_keypoints)))
            video_annotations[interval_element] = interval_annotations
            
    # Loading manual annotations
    path = RAW_BRACE_PATH + RAW_KEYPOINT_FOLDERS["manual"] + RAW_KEYPOINTS_SUBFOLDERS["manual"] + year + "/" + video_id + "/"
    keypoints_listdir = os.listdir(path)
    
    # Looping through all of the manual annotations
    for keypoint_file in keypoints_listdir:
        keypoint_path = path + keypoint_file
        
        # Reading annotations
        keypoints = torch.from_numpy(np.load(keypoint_path)['coco_joints2d'][:, :2])
        
        # Making bbox that fits inside the frame.
        fitted_keypoints = keypoints[(0 <= keypoints[:, 0]) & (keypoints[:, 0] < BRACE_WIDTH) & (0 <= keypoints[:, 1]) & (keypoints[:, 1] < BRACE_HEIGHT)]
        clip_bboxes = [fitted_keypoints[:, 0].min().item(), fitted_keypoints[:, 1].min().item(), fitted_keypoints[:, 0].max().item(), fitted_keypoints[:, 1].max().item()]
        
        # Getting key-name
        key_name = keypoint_file[-10:-4].lstrip("0")
        
        # Inserting manual annotation in the correct dict of the list
        for v in video_annotations.values():
            if key_name in v:
                v[key_name] = (clip_bboxes, keypoints)

    return video_annotations

def _preprocess_keypoints(video_annotations : Dict[str, Tuple[List[float]]], blurr_sigma: float):
    """
    Preprocesses keypoints of a single frame.
    
    Parameters
    ----------
    video_annotations : Tuple[List[float]]
        The first element is the bbox of the frame
        and the second element is the keypoints of the frame.
        
    blurr_sigma: float
        Sigma to use during gaussian blurr
    
    Returns
    -------
    processed_output_heatmaps : torch.Tensor
        Preprocessed heatmaps of the keypoints.
        Used as the output to a model
    
    processed_input_heatmaps : torch.Tensor
        Preprocessed heatmaps of the keypoints.
        Used as the input to a model
    """
    
    # Reading annotations
    bbox, keypoints = video_annotations
    
    if keypoints.sum().item() == 0:
        # If the frame does not contain any keypoints
        return torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH), torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH), torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Making bbox a square, by expanding the shortest side
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    diff = np.abs(height - width)
    expand_factor = diff/2
    
    if width < height:
        x_min -= expand_factor
        x_max += expand_factor
    else:
        y_min -= expand_factor
        y_max += expand_factor
    
    width = x_max - x_min
    height = y_max - y_min
        
    # Expanding sides by 10%
    expand_factor = 0.1 * width * 0.5
    x_min -= expand_factor 
    x_max += expand_factor 
    y_min -= expand_factor 
    y_max += expand_factor 

    # Shifts keypoints, corresponding to such that the upper left koordinate
    # of the bbox has coordinates (0, 0)
    x_max -= x_min
    y_max -= y_min
        
    keypoints[:, 0] -= x_min
    keypoints[:, 1] -= y_min
    
    # Rescaling keypoints to the correct range
    rescale_width = (TARGET_WIDTH - 1) / round(x_max)
    rescale_height = (TARGET_HEIGHT - 1) / round(y_max)

    keypoints[:, 0] *= rescale_width
    keypoints[:, 1] *= rescale_height
    
    # Rounding to nearest integer
    keypoints = torch.round(keypoints).int()
    
    # Flipping the keypoints horizontally
    # keypoints[:, 1] = TARGET_HEIGHT - 1 - keypoints[:, 1]

    # Tensor for storing input heatmaps
    processed_input_heatmaps = torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Tensor for storing output heatmaps
    processed_output_heatmaps = torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Tensor for storing input heatmaps with varying stds for gaussian blur
    processed_input_heatmaps_mixed_std = torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH)
    
    # Function for translating keypoints for translating
    # BRACE keypoint-index to ClimbAlong keypoint-index
    translate = lambda i: CLIMBALONG_KEYPOINTS[BRACE_KEYPOINTS[i]]

    # Values for randomly shifting keypoints
    shifts = np.clip(np.round(np.random.normal(0, 2.5, size=(len(keypoints), 2))).astype(int), -10, 10)
    
    # Looping through each keypoint
    for i, keypoint in enumerate(keypoints):        
        if BRACE_KEYPOINTS[i] not in CLIMBALONG_KEYPOINTS:
            # Some keypoints from BRACE are not used in ClimbAlong-data
            continue
        
        if not (0 <= keypoint[0] < TARGET_WIDTH and 0 <= keypoint[1] < TARGET_HEIGHT):
            continue
        
        # Value for randomly shifting keypoints
        shift = shifts[i]
            
        # Translating keypoint-index to correct index
        i = translate(i)
        
        # Randomly shifting input keypoints
        shifted_keypoint = np.clip(keypoint + shift, [0, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1])
        
        # Making heatmap from keypoint
        input_heatmap = turn_keypoint_to_featuremap(shifted_keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
        input_heatmap_mixed_std = turn_keypoint_to_featuremap(shifted_keypoint, (TARGET_HEIGHT, TARGET_WIDTH), blurr_sigma)
        output_heatmap = turn_keypoint_to_featuremap(keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
        
        # Inserting data
        processed_input_heatmaps[i] = input_heatmap
        processed_input_heatmaps_mixed_std[i] = input_heatmap_mixed_std
        processed_output_heatmaps[i] = output_heatmap
            
    return processed_input_heatmaps, processed_output_heatmaps, processed_input_heatmaps_mixed_std
            
def preprocess():
    """
    Function for preprocessing the videos in the BRACE-dataset.
    """
    
    # Loading csv containing meta-information of videos
    meta_info = pd.read_csv(RAW_BRACE_PATH + METAINFO_NAME)
    
    # Iterating through each video and prepare it
    for _, row in tqdm(meta_info.iterrows(), desc="Preparing videos", leave=True, total=len(meta_info), disable=False):
        
        # Extracting meta info about video
        video_id = row["video_id"]
        year = str(row["year"])
        
        # Path for storing the keypoints of this video
        input_keypoints_path = OVERALL_DATA_FOLDER + SUBFOLDERS["x"] + video_id
        output_keypoints_path = OVERALL_DATA_FOLDER + SUBFOLDERS["y"] + video_id
        input_keypoints_mixed_std_path = OVERALL_DATA_FOLDER + SUBFOLDERS["x_std"] + video_id
        
        # Loading keypoints of video
        video_annotations = _load_keypoints(year, video_id)
        
        # Iterating through the keypoint-annotations of each clip of the current video
        for clip_interval, clip_annotations in tqdm(video_annotations.items(), desc="Storing clip-keypoints", leave=False, disable=False):
            input_clip_storing_path = input_keypoints_path + "_" + clip_interval
            output_clip_storing_path = output_keypoints_path + "_" + clip_interval
            input_clip_storing_mixed_std_path = input_keypoints_mixed_std_path + "_" + clip_interval
            
            make_dir(input_clip_storing_path)
            make_dir(output_clip_storing_path)
            make_dir(input_clip_storing_mixed_std_path)
            
            sigmas = np.random.choice(GAUSSIAN_STDS, size=len(clip_annotations))
            
            for i, (frame_number, frame_annotation) in tqdm(enumerate(clip_annotations.items()), desc="Storing frame-keypoints", leave=False, disable=False, total=len(clip_annotations)):
                
                # Paths for storing stuff
                input_frame_storing_path = input_clip_storing_path + frame_number + ".pt"
                output_frame_storing_path = output_clip_storing_path + frame_number + ".pt"
                input_frame_storing_mixed_std_path = input_clip_storing_mixed_std_path + frame_number + ".pt"
                
                # Processing keypoints of the loaded clip
                processed_input_heatmaps, processed_output_heatmaps, processed_input_heatmaps_mixed_std = _preprocess_keypoints(frame_annotation, blurr_sigma=sigmas[i])
                    
                # Saving keypoints of frame as tensor
                torch.save(processed_input_heatmaps, input_frame_storing_path)
                torch.save(processed_output_heatmaps, output_frame_storing_path)
                torch.save(processed_input_heatmaps_mixed_std, input_frame_storing_mixed_std_path)
            
if __name__ == "__main__":
    preprocess()