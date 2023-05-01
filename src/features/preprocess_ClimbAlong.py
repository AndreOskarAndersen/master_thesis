import os
import numpy as np
import torch
from tqdm.auto import tqdm
from skimage.filters import gaussian
from utils import make_dir
from global_variables import CA_PROCESSED_PATH, CA_PROCESSED_SUBDIRS, CA_RAW_PATH, CA_RAW_SUBDIRS, TARGET_HEIGHT, TARGET_WIDTH, NUM_KEYPOINTS

np.set_printoptions(suppress=True)

def _make_dirs():
    """
    Function for making directories to be used.
    """
    
    for subdir in CA_PROCESSED_SUBDIRS.values():
        make_dir(CA_PROCESSED_PATH + subdir)
        
def _make_heatmaps(xs, ys):
    heatmaps = torch.zeros((NUM_KEYPOINTS, TARGET_HEIGHT, TARGET_WIDTH))
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        x = np.clip(x, 0, TARGET_WIDTH - 1)
        y = np.clip(y, 0, TARGET_HEIGHT - 1)
        
        heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH))
        heatmap[y, x] = 1
        heatmaps[i] = torch.from_numpy(gaussian(heatmap, sigma=1))
        
    return heatmaps
        
def _preprocess_sample(sample_name, input_storing_path, target_storing_path):
    input_raw_path = CA_RAW_PATH + CA_RAW_SUBDIRS["x"] + sample_name + ".npz"
    target_raw_path = CA_RAW_PATH + CA_RAW_SUBDIRS["y"] + sample_name + ".npz"
    
    input_data = np.load(input_raw_path)
    target_data = np.load(target_raw_path)
    
    input_bboxes = input_data["bboxes"]
    input_heatmaps = input_data["heatmaps"]
    target_keypoints = target_data["keypoints"]
    
    for i, (frame_input_bbox, frame_input_heatmaps, frame_target_keypoints) in enumerate(zip(input_bboxes, input_heatmaps, target_keypoints)):
        input_frame_storing_path = input_storing_path + str(i) + ".pt"
        target_frame_storing_path = target_storing_path + str(i) + ".pt"
        
        if frame_input_bbox[2] - frame_input_bbox[0] == 0 or frame_input_bbox[3] - frame_input_bbox[1] == 0:
            continue
        
        xs = np.ceil((frame_target_keypoints[:, 0] - frame_input_bbox[0]) * TARGET_WIDTH/(frame_input_bbox[2] - frame_input_bbox[0])).astype(int)
        ys = np.ceil((frame_target_keypoints[:, 1] - frame_input_bbox[1]) * TARGET_HEIGHT/(frame_input_bbox[3] - frame_input_bbox[1])).astype(int)
        
        frame_target_heatmaps = _make_heatmaps(xs, ys)
        
        torch.save(frame_input_heatmaps, input_frame_storing_path)
        torch.save(frame_target_heatmaps, target_frame_storing_path)
        
def _preprocess_samples():
    sample_names = list(map(lambda sample_name: sample_name.split(".")[0], os.listdir(CA_RAW_PATH + CA_RAW_SUBDIRS["x"])))
    
    for sample_name in tqdm(sample_names, disable=True):
        input_storing_path = CA_PROCESSED_PATH + CA_PROCESSED_SUBDIRS["x"] + sample_name + "/"
        target_storing_path = CA_PROCESSED_PATH + CA_PROCESSED_SUBDIRS["y"] + sample_name + "/"
        
        make_dir(input_storing_path)
        make_dir(target_storing_path)
        
        _preprocess_sample(sample_name, input_storing_path, target_storing_path)

def preprocess():
    """
    Main entrypoint for preprocessing the ClimbAlong dataset.
    """
    
    # Making directories to be used
    _make_dirs()
    
    # Preprocesses the samples
    _preprocess_samples()

if __name__ == "__main__":
    preprocess()