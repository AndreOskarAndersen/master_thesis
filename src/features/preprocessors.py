import numpy as np
import pandas as pd
import pims
import torch
import cv2
import json
import torchvision
from tqdm import tqdm
from global_variables import *
from loading import *

# Rescaling factors
original_height = 1080
original_width = 1920
target_width = 800
rescale_factor = target_width/original_width

new_width = int(original_width * rescale_factor)
new_height = int(original_height * rescale_factor)

def _preprocess_clip(clip: pims.Video, keypoints_dict: dict):
    """
    Function for preprocessing a single video-clip from the BRACE-dataset.
    
    Parameters
    ----------
    clip : pims.Video
        Clip to preprocess
        
    keypoints_dict : dict
        Dict containing the keypoint annotations of the clip.
        Each key is a frame-number and the corresponding value
        is the keypoint annotations of that frame.
        
    Returns
    -------
    clip : torch.Tensor
        tensor of the processed video
        
    """
    
    # Casting to numpy
    clip = np.array(clip)
    
    # Resizing
    resized_clip = np.zeros((len(keypoints_dict.keys()), new_height, new_width, 3), dtype=np.uint8)
    for i in range(len(keypoints_dict.keys())):
        frame = clip[i]
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_clip[i] = resized_frame
    clip = resized_clip
    
    # Casting to torch tensor
    clip = torch.from_numpy(clip)
    
    return clip

def preprocess_keypoints(keypoints_dict: dict):
    """
    Preprocesses keypoints.
    
    Parameters
    ----------
    keypoints_dict : dict
        Dict containing the keypoint annotations of the clip.
        Each key is a frame-number and the corresponding value
        is the keypoint annotations of that frame.
    
    Returns
    -------
    preprocessed_keypoints_dict : dict
        Preprocessed version of keypoints_dict
    
    """
    
    preprocessed_keypoints_dict = {}
    
    for k, v in keypoints_dict.items():
        
        # Resizing keypoints
        v = list(map(lambda x: list(map(lambda y: round(y * rescale_factor), x)), v))
        
        # Removing unnecessary keypoints
        v = v[:-4]
        
        # Storing in new dictionary of keypoints
        preprocessed_keypoints_dict[int(k)] = v        

    return preprocessed_keypoints_dict
            
def preprocess_videos():
    """
    Function for preprocessing the videos in the BRACE-dataset.
    """
    
    # Loading csv containing meta-information of videos
    meta_info = pd.read_csv(RAW_DATA_FOLDER + METAINFO_NAME)
    
    # Iterating through each video and prepare it
    for _, row in tqdm(meta_info.iterrows(), desc="Preparing videos", leave=True, total=len(meta_info)):
        # Extracting meta info about video
        video_id = row["video_id"]
        year = str(row["year"])
        fps = row["fps"]
        
        # Loading keypoints of video
        all_keypoints = load_keypoints(year, video_id)
        
        # Loading video 
        video = load_video(video_id)
        
        # Iterating through the keypoint-annotations of each clip of the current video
        for keypoint_dict in tqdm(all_keypoints, desc="Processing clip", leave=False):
            
            # Extracting the number of the start/end-frames
            start = str(min(keypoint_dict.keys()))
            end = str(max(keypoint_dict.keys()))
            
            # Path for storing stuff
            clip_storing_path = OVERALL_DATA_FOLDER + SUB_DATA_FOLDERS["videos"] + video_id + "_" + start + "-" + end + ".mp4"
            keypoints_storing_path = OVERALL_DATA_FOLDER + SUB_DATA_FOLDERS["keypoints"] + video_id + "_" + start + "-" + end + ".json"
            
            # Loading clip
            clip = load_clip(video, keypoint_dict)
            
            # Preprocessing the loaded clip
            processed_clip = _preprocess_clip(clip, keypoint_dict)
            
            # Storing clip
            torchvision.io.write_video(clip_storing_path, processed_clip, fps, "libx264")
            
            # Freeing clip om memory
            del processed_clip
            del clip
            
            # Processing keypoints of the loaded clip
            preprocessed_keypoints = preprocess_keypoints(keypoint_dict)
            
            # Saving JSON
            json.dump(preprocessed_keypoints, open(keypoints_storing_path, "w"))