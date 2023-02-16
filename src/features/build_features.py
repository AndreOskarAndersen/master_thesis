import numpy as np
import os
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from skimage.transform import rescale
from global_variables import *
from time import sleep

def _make_dir(path):
    """
    Makes a directory if it does not already exist.
    
    Parameters
    ----------
    path : str
        Path of the directory to create.
    """
    
    try:
        os.mkdir(path)
    except:
        print(f"Folder {path} already exists. Using existing folder.")

def _make_corpus_folders():
    """
    Function for making various folders for storing the data
    """
    
    # Making overall folder for the processed-data, in case it does not already exist.
    _make_dir(OVERALL_DATA_FOLDER)
    
    # Making the folders for storing the unprocessed keypoints and videos folders
    # in case they do not already exist
    for sub_folder in SUB_DATA_FOLDERS.values():
        _make_dir(OVERALL_DATA_FOLDER + sub_folder) 
            
def _load_video(video_path: str, number_of_frames: int):
    """
    Function for loading a video as a numpy-array.
    
    Parameters
    ----------
    video_path : str
        Path of the video to load
        
    Return
    ------
    video : np.array
        Numpy-representation of the video to return
        
    number_of_frames : int
        The total number of frames in the video
        
    Note
    ----
    The implementation of this function made use of some code from the following link
    
    https://stackoverflow.com/a/65447018/12905157
    """
    
    # Rescaling factors
    height = 1080
    width = 1920
    target_width = 800
    rescale_factor = target_width/width
    
    width = int(width * rescale_factor)
    height = int(height * rescale_factor)
    dim = (width, height)
    
    # List for storing video.
    frames = torch.zeros((number_of_frames, height, width, 3), dtype=torch.uint8)
    
    # Index for keeping track of placement of frame
    idx = 0
    
    # Video capture 
    cap = cv2.VideoCapture(video_path)
    
    # "ret" indicates whether a frame was retrieved
    ret = True
    
    # Looping over all frames
    with tqdm(total=number_of_frames, desc="Loading video", leave=False) as pbar:
        while ret:
            
            # Retrieving a frame
            ret, frame = cap.read()
            
            # If a frame was correctly retrieved
            if ret:
                
                # Rescaling the frame to (H, W) = (450, 800)
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                
                # Casting to pytorch tensor
                frame = torch.from_numpy(frame)
            
                # Storing the rescaled frame
                #frames.append(frame)
                frames[idx] = frame
            else:
                print(idx)
                
            # Updating TQDM-bar
            pbar.update(1)
            
            # Updating index
            idx += 1
    
    return frames
            
def _preprocess_video(video_id: str, duration: float, fps: float):
    """
    Function for preprocessing a single video from the BRACE-dataset.
    
    Parameters
    ----------
    video_id : str
        ID of the video to preprocess.
        
    """
    
    # Loading the video
    video_folder = RAW_DATA_FOLDER + RAW_CORPUS_FOLDERS["videos_folder"] + video_id + "/"
    video_name = list(filter(lambda x: x != ".DS_Store", os.listdir(video_folder)))[0]
    video_path = video_folder + video_name
    video = _load_video(video_path, np.ceil(duration * fps).astype(int))
    
    return video

def _load_keypoints(year: str, video_id: str):
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
        
        # Inserting manual annotation to the list
        for clip_dict in all_keypoints:
            if min(clip_dict.keys()) <= key_name <= max(clip_dict.keys()):
                clip_dict[key_name] = keypoints
    
    return all_keypoints
            
def _preprocess_videos():
    """
    Function for preprocessing the videos in the BRACE-dataset.
    """
    
    # Loading csv containing meta-information of videos
    meta_info = pd.read_csv(RAW_DATA_FOLDER + METAINFO_NAME)
    
    # Iterating through each video and prepare it
    for _, row in tqdm(meta_info.iterrows(), desc="Preparing videos", leave=True, total=len(meta_info)):
        
        # Loading keypoints of video
        keypoints = _load_keypoints(str(row["year"]), row["video_id"])
        
        # Preprocessing video
        processed_video = _preprocess_video(row["video_id"], row["duration"], row["fps"])
        
        # Storing video
        storing_path = OVERALL_DATA_FOLDER + SUB_DATA_FOLDERS["videos"] + row["video_id"] + ".pt"
        torch.save(processed_video, storing_path)
        
        # Freeing video om memory
        del processed_video
        break
            
if __name__ == "__main__":
    _make_corpus_folders()
    
    # Preprocessing video
    _preprocess_videos()