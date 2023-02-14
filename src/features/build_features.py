import numpy as np
import os
import pandas as pd
import cv2
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
    for sub_folder in SUB_DATA_FOLDERS:
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
    # Preallocating memory to speed up the loading of the video.
    frames = np.zeros((number_of_frames, height, width, 3))
    
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
            
                # Storing the rescaled frame
                #frames.append(frame)
                frames[idx] = frame
                
            # Updating TQDM-bar
            pbar.update(1)
            
            # Updating index
            idx += 1
            
    # Casting to numpy array
    #video = np.stack(frames, axis=0)
    
    return frames
            
def _preprocess_video(video_id: str, duration: float, fps: float):
    """
    Function for preprocessing a single video from the BRACE-dataset.
    
    Parameters
    ----------
    video_id : str
        ID of the video to preprocess.
        
    """
    
    video_folder = RAW_DATA_FOLDER + RAW_CORPUS_FOLDERS["videos_folder"] + video_id + "/"
    video_name = os.listdir(video_folder)[1]
    video_path = video_folder + video_name
    video = _load_video(video_path, np.ceil(duration * fps).astype(int))
    print(video.shape)
    
    return video
            
def _preprocess_videos():
    """
    Function for preprocessing the videos in the BRACE-dataset.
    """
    
    # Loading csv containing meta-information of videos
    meta_info = pd.read_csv(RAW_DATA_FOLDER + METAINFO_NAME)
    
    # Iterating through each video and prepare it
    for _, row in tqdm(meta_info.iterrows(), desc="Preparing videos", leave=True, total=len(meta_info)):
        _preprocess_video(row["video_id"], row["duration"], row["fps"])
        break
            
if __name__ == "__main__":
    #_make_corpus_folders()
    _preprocess_videos()