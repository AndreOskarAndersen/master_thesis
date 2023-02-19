import numpy as np
import os
import cv2
import scipy.io
import json
from global_variables import *
from tqdm.auto import tqdm

class npEncoder(json.JSONEncoder):
    """
    Class used for extending JSON-encoder to int32.
    
    Note
    ----
    Inspired by the following code-snippet
    
    https://stackoverflow.com/a/56254172
    """
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def process_video(loading_path: str, saving_path: str):
    """
    Function for preprocessing a Penn Action video.
    
    Parameters
    ----------
    loading_path : str
        Path to the folder containing the frames of the video
        
    saving_path : str 
        Path to where the processed video should be stored
    
    Note
    ----
    Inspired by the following code-snippet
    
    https://stackoverflow.com/a/44948030
    """
    
    # Number of frames in the current video
    num_frames = len(os.listdir(loading_path))
    
    # List of names of the frames of the video
    images = [f"{i}".zfill(6) + ".jpg" for i in range(1, num_frames + 1)]
    
    # Creating the VideoWriter-object
    frame = cv2.imread(loading_path + images[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(saving_path, 0, 30, (width, height))
    
    # Looping through each frame and writing it to the video
    for image in images:
        video.write(cv2.imread(loading_path + image))

    cv2.destroyAllWindows()
    video.release()
    
def process_keypoints(label: dict, saving_path: str):
    """
    Processes and saves keypoints
    
    Parameters
    ----------
    label : dict
        dictionary of the labels
        
    saving_path : str
        path of where to store the keypoints
    
    """
    
    # Loads the keypoints and round them from subpixel precision.
    x = label["x"].round().astype(int)
    y = label["y"].round().astype(int)
    
    # Mapping of indices to a normalized index
    indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Dictionary of where the processed keypoints will be stored
    kps = {}
    
    # Looping through the koordinates of each keypoint
    for i in range(len(x)):
        # Zipping the keypoints to a single 2D array
        xs = x[i]
        ys = y[i]
        xy = list(map(list, zip(xs, ys)))
        
        # Stores the keypoint using the normalized indices
        #kp = {indx: xy[j] for j, indx in enumerate(indices)}
        kp = {}
        for j, indx in enumerate(indices):
            kp[indx] = xy[j]
        
        kps[i] = kp
        
    # Saves the keypoints
    json.dump(kps, open(saving_path, "w"), cls=npEncoder)
        
def preprocess():
    """
    Function for preprocessing all the relevant Penn Action samples.
    """
    
    # Paths for data
    frames_folders_path = PENN_ACTION_RAW_PATH + SUB_PENN_ACTION_FOLDERS["frames"]
    labels_path = PENN_ACTION_RAW_PATH + SUB_PENN_ACTION_FOLDERS["keypoints"]
    
    # List of sample-names
    samples = os.listdir(frames_folders_path)
    
    # Looping through each sample
    for frame_folder in tqdm(samples, desc="Processing videos"):
        
        # Various paths of this sample
        label_path = labels_path + frame_folder + ".mat"
        frame_folder_path = frames_folders_path + frame_folder + "/"
        video_saving_path = OVERALL_DATA_FOLDER + SUB_DATA_FOLDERS["videos"] + frame_folder + ".mp4"
        label_saving_path = OVERALL_DATA_FOLDER + SUB_DATA_FOLDERS["keypoints"] + frame_folder + ".json"
        
        # Loading the label and the action
        label = scipy.io.loadmat(label_path)
        action = label["action"][0]
        
        # If the action is not relevant, skip to the next sample
        if action not in RELEVANT_ACTIONS:
            continue
        
        # Process and save the video and keypoints.
        process_video(frame_folder_path, video_saving_path)
        process_keypoints(label, label_saving_path)