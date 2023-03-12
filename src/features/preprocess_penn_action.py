import numpy as np
import os
import scipy.io
import json
import torch
from tqdm.auto import tqdm
from utils import make_dir, turn_keypoint_to_featuremap
from global_variables import *

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
    
def _preprocess_keypoints(label, bboxes):
    keypoints = np.dstack((label["x"], label["y"])) # (num_frames, num_keypoints, 2)
    
    # Extracting cornors of bboxes
    x_mins, y_mins, x_maxs, y_maxs = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
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
    
    # Shifts keypoints, corresponding to such that the upper left koordinate
    # of the bbox has coordinates (0, 0)
    x_maxs -= x_mins
    y_maxs -= y_mins
        
    keypoints[:, :, 0] -= x_mins.reshape((-1, 1))
    keypoints[:, :, 1] -= y_mins.reshape((-1, 1))
    
    # Rescaling keypoints to the correct range
    rescale_width = TARGET_WIDTH / np.round(x_maxs)
    rescale_height = TARGET_HEIGHT / np.round(y_maxs)

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
                # TODO: SOMETIMES ENTERS THIS IF_STATEMENT
                # DUNNO IF IT IS SUPPOSED TO.
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
    for video_id in tqdm(samples, desc="Processing videos", disable=True):
        
        # Removing file type
        video_id = video_id[:-4]
        
        # Path for storing the keypoints of this video
        keypoints_path = OVERALL_DATA_FOLDER + video_id + "/"
        
        # Making folder for storing keypoints of current video
        make_dir(keypoints_path)
        
        # Path for storing meta-info of this sample
        label_path = PENN_ACTION_RAW_KEYPOINTS_PATH + video_id + ".mat"
        
        # Loading the label and the action
        label = scipy.io.loadmat(label_path)
        action = label["action"][0]
        bboxes = label["bbox"]
        
        # If the action is not relevant, skip to the next sample
        if action not in RELEVANT_ACTIONS:
            continue
        
        # Process the keypoints
        preprocessed_keypoints = _preprocess_keypoints(label, bboxes)
        
        # Storing keypoints as individual frames
        for frame_number, peprocessed_keypoint in enumerate(preprocessed_keypoints):
        
            # Path for storing stuff
            keypoints_storing_path = keypoints_path + str(frame_number) + ".pt"
            
            # Saving keypoints of frame as tensor
            torch.save(peprocessed_keypoint, keypoints_storing_path)
            
        exit(1)
        
if __name__ == "__main__":
    preprocess()