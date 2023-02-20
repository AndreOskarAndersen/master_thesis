import os
import numpy as np
import pandas as pd
import pims
import torch
import cv2
import json
import torchvision
from global_variables import *
from tqdm import tqdm

# Rescaling factors
original_height = 1080
original_width = 1920
target_width = 800
rescale_factor = target_width/original_width

new_width = int(original_width * rescale_factor)
new_height = int(original_height * rescale_factor)

def load_clip(video: pims.Video, keypoints_dict: dict):
    """
    Function for loading a clip as a numpy-array.
    
    Parameters
    ----------
    video_path : str
        Path of the video containing the frames of the clip (and some other frames)
        
    keypoints_dict : dict
        Dict containing the keypoint annotations of the clip.
        Each key is a frame-number and the corresponding value
        is the keypoint annotations of that frame.
        
    Return
    ------
    clip : np.array
        Numpy-representation of the clip to return
    """   
    
    # Extracting frames of clip
    start_frame = min(keypoints_dict.keys()) - 1 # NOTE: NOT SURE IF ANNOTATED FRAMES ARE 0 INDEXED
    end_frame = max(keypoints_dict.keys()) # NOTE: NOT SURE IF ANNOTATED FRAMES ARE 0 INDEXED
    clip = video[start_frame:end_frame]
    
    return clip

def load_video(video_id: str):
    """
    loads a video.
    
    Parameters
    ----------
    video_id : str
        ID of the video to load
        
    Returns
    -------
    video : pims.Video
        The loaded video
    """
    
    video_folder = RAW_DATA_FOLDER + RAW_CORPUS_FOLDERS["videos_folder"] + video_id + "/"
    video_name = list(filter(lambda x: x != ".DS_Store", os.listdir(video_folder)))[0]
    video_path = video_folder + video_name
    video = pims.Video(video_path)
    
    return video

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
    all_keypoints : list[dict]
        List with length of the amount of clips of the video,
        where each entrance is a dict
        where the key is a frame-name and the value is the
        keypoints of that frame
    """
    
    all_keypoints = []
    
    # loading automatic directories
    path = RAW_DATA_FOLDER + RAW_CORPUS_FOLDERS["keypoints_folder"] + "/" + RAW_KEYPOINT_FOLDERS["automatic"] + RAW_KEYPOINTS_SUBFOLDERS["automatic"] + year + "/" + video_id + "/"
    keypoints_listdir = os.listdir(path)
    
    # Looping through all of the automatic annotations
    for keypoint_file in keypoints_listdir:
        keypoint_path = path + keypoint_file
        
        # Reading annotations
        keypoints = pd.read_json(keypoint_path)
        
        # Deleting bbox-annotation
        keypoints = keypoints.drop(index=("box"))
        
        # Renaming columns
        columns = list(keypoints.columns)
        columns = list(map(lambda x: int(x[-10:-4].lstrip("0")), columns))
        keypoints.columns = columns
        
        # Removing "score"-attribute
        keypoints = keypoints.loc["keypoints"]
        keypoints = keypoints.apply(lambda x: list(map(lambda y: [y[0], y[1]], x)))
        
        # Casting to dict
        keypoints = keypoints.to_dict()
        
        # Appending the keypoints of this clip
        # to the list
        all_keypoints.append(keypoints)
            
    # Loading manual annotations
    path = RAW_DATA_FOLDER + RAW_CORPUS_FOLDERS["keypoints_folder"] + "/" + RAW_KEYPOINT_FOLDERS["manual"] + RAW_KEYPOINTS_SUBFOLDERS["manual"] + year + "/" + video_id + "/"
    keypoints_listdir = os.listdir(path)
    
    # Looping through all of the manual annotations
    for keypoint_file in keypoints_listdir:
        keypoint_path = path + keypoint_file
        
        # Reading annotations
        keypoints = np.load(keypoint_path)['coco_joints2d'][:, :2]
        
        # Getting key-name
        key_name = int(keypoint_file[-10:-4].lstrip("0"))
        
        # Inserting manual annotation in the correct dict of the list
        for clip_dict in all_keypoints:
            if min(clip_dict.keys()) <= key_name <= max(clip_dict.keys()):
                clip_dict[key_name] = keypoints
    
    return all_keypoints

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
        v = np.concatenate(([v[0]], v[5:]))
        
        # Storing in new dictionary of keypoints
        preprocessed_keypoints_dict[int(k)] = v.tolist()

    return preprocessed_keypoints_dict
            
def preprocess():
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