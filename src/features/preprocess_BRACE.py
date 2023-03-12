import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple, Dict, List
from utils import make_dir, turn_keypoint_to_featuremap
from global_variables import *

def load_keypoints(year: str, video_id: str):
    """
    Loads the automatically and manually annotated BRACE keypoints.
    
    Parameters
    ----------
    year : str
        The year of the recording of the video.
        
    video_id : str
        The ID of the video.
        
    Returns
    -------
    video_annotations : Dict[str, Tuple[List[float]]]
        dict with length of the amount of annotated frames of the video,
        where each key is the frame-number and the corresponding value
        is a tuple, where the first element is the bbox of the frame
        and the second element is the keypoints of the frame.
    """
    
    video_annotations = {}
    
    # loading automatic directories
    path = RAW_BRACE_PATH + RAW_KEYPOINT_FOLDERS["automatic"] + RAW_KEYPOINTS_SUBFOLDERS["automatic"] + year + "/" + video_id + "/"
    keypoints_listdir = os.listdir(path)
    
    # Looping through all of the automatic annotations
    for keypoint_file in keypoints_listdir:
        keypoint_path = path + keypoint_file
        
        # Reading annotations
        annotations = pd.read_json(keypoint_path)
        
        # Columns used for naming frames.
        keys = list(annotations.columns)
        keys = list(map(lambda x: x[-10:-4].lstrip("0"), keys))
        
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
        clip_annotations = dict(zip(keys, zip(clip_bboxes, clip_keypoints)))
        video_annotations.update(clip_annotations)
            
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
        video_annotations[key_name] = (clip_bboxes, keypoints)

    return video_annotations

def preprocess_keypoints(video_annotations : Dict[str, Tuple[List[float]]]):
    """
    Preprocesses keypoints.
    
    Parameters
    ----------
    video_annotations : Dict[str, Tuple[List[float]]]
        dict with length of the amount of annotated frames of the video,
        where each key is the frame-number and the corresponding value
        is a tuple, where the first element is the bbox of the frame
        and the second element is the keypoints of the frame.
    
    Returns
    -------
    preprocessed_keypoints_dict : torch.Tensor
        Preprocessed keypoints.
    
    """
    
    # Reading annotations
    bbox, keypoints = video_annotations
    
    if keypoints.sum().item() == 0:
        # If the frame does not contain any keypoints
        return torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH).bool()
    
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
    
    x_min = 0
    y_min = 0
    
    # Rescaling keypoints to the correct range
    rescale_width = TARGET_WIDTH / round(x_max - x_min)
    rescale_height = TARGET_HEIGHT / round(y_max - y_min)

    keypoints[:, 0] *= rescale_width
    keypoints[:, 1] *= rescale_height
    
    # Rounding to nearest integer
    keypoints = torch.round(keypoints).int()
    
    # Flipping the keypoints horizontally
    keypoints[:, 1] = TARGET_HEIGHT - 1 - keypoints[:, 1]
    
    # Tensor for storing heatmaps
    processed_heatmaps = torch.zeros(NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH).bool()
    
    # Function for translating keypoints for translating
    # BRACE keypoint-index to ClimbAlong keypoint-index
    translate = lambda i: CLIMBALONG_KEYPOINTS[BRACE_KEYPOINTS[i]]
    
    # Looping through each keypoint
    for i, keypoint in enumerate(keypoints):        
        if BRACE_KEYPOINTS[i] not in CLIMBALONG_KEYPOINTS:
            # Some keypoints from BRACE are not used in ClimbAlong-data
            continue
        
        if not (0 <= keypoint[0] < TARGET_WIDTH and 0 <= keypoint[1] < TARGET_HEIGHT):
            continue
            
        # Translating keypoint-index to correct index
        i = translate(i)
        
        # Making heatmap from keypoint
        heatmap = turn_keypoint_to_featuremap(keypoint, (TARGET_HEIGHT, TARGET_WIDTH))
        
        # Inserting data
        processed_heatmaps[i] = heatmap
            
    return processed_heatmaps
            
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
        
        # Path for the keypoints of this video
        keypoints_path = OVERALL_DATA_FOLDER + video_id + "/"
        
        # Loading keypoints of video
        video_annotations = load_keypoints(year, video_id)
        
        # Making folder for storing keypoints of current video
        make_dir(keypoints_path)
        
        # Iterating through the keypoint-annotations of each clip of the current video
        for frame_number, frame_annotation in tqdm(video_annotations.items(), desc="Storing clip-keypoints", leave=False, disable=False):
            
            # Path for storing stuff
            keypoints_storing_path = keypoints_path + frame_number + ".pt"
            
            # Processing keypoints of the loaded clip
            try:
                preprocessed_keypoints = preprocess_keypoints(frame_annotation)
            except Exception:
                import traceback
                print()
                print()
                print("video_id", video_id)
                print("year", year)
                print(frame_number)
                print(traceback.format_exc())
                exit(1)
                
            # Saving keypoints of frame as tensor
            torch.save(preprocessed_keypoints, keypoints_storing_path)
            
if __name__ == "__main__":
    preprocess()