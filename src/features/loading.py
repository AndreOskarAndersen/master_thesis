import os
import numpy as np
import pandas as pd
import pims
from global_variables import *

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