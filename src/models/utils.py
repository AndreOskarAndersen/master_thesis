import os
import numpy as np
import torch
from typing import Union, List
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from global_variables import GENERAL_MISSING_INDICIES, PA_MISSING_INDICIES

def make_dir(path):
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
        
def get_torso_diameter(keypoints):
    num_dimensions = 2
    
    num_batches, num_frames, num_heatmaps = keypoints.shape
    num_heatmaps = num_heatmaps//num_dimensions
    
    torso_diameter = []
    
    for batch_idx in range(num_batches):
        for frame_idx in range(num_frames):
            left_shoulder_kp = np.array([keypoints[batch_idx][frame_idx][6], keypoints[batch_idx][frame_idx][7]])
            right_shoulder_kp = np.array([keypoints[batch_idx][frame_idx][8], keypoints[batch_idx][frame_idx][9]])
            left_hip_kp = np.array([keypoints[batch_idx][frame_idx][30], keypoints[batch_idx][frame_idx][31]])
            right_shoulder_kp = np.array([keypoints[batch_idx][frame_idx][32], keypoints[batch_idx][frame_idx][33]])
            torso_diameter.append([(np.linalg.norm(left_shoulder_kp - left_hip_kp) + np.linalg.norm(right_shoulder_kp - right_shoulder_kp))/2] * num_heatmaps)

    return np.array(torso_diameter).reshape((-1, 1))

def heatmaps2coordinates(featuremaps: Union[np.array, torch.Tensor]):
    """
    Turns a video sequence of heatmaps into the equivalent
    sequence of keypoint coordinates.
    
    Parameters
    ----------
    featuremaps : Union[np.array, torch.Tensor]
        Video sequence of heatmaps
        
    Returns
    -------
    keypoints : torch.Tensor
        Keypoint coordinates.
    """
    
    assert len(featuremaps.shape) == 5, f"Your video sequence should have the following shape: (num_batches, num_frames, num_heatmaps, height, width). Yours have shape {featuremaps.shape}"
    
    # Extracting useful info
    num_batches, num_frames, num_heatmaps = featuremaps.shape[:3]
    
    # Casting featuremaps to numpy
    featuremaps = featuremaps.detach().cpu().numpy()
    
    # Dimensions of each keypoints
    num_dimensions = 2
    
    # Preallocating memory
    keypoints = torch.zeros((num_batches, num_frames, num_dimensions * num_heatmaps))
    
    for batch_idx in range(num_batches):
        for frame_idx in range(num_frames):
            for heatmap_idx in range(num_heatmaps):
                
                # Finds the coordinates of the keypoint-heatmap
                heatmap = featuremaps[batch_idx, frame_idx, heatmap_idx]
                coordinate = np.unravel_index(heatmap.argmax(), heatmap.shape)
                
                # Stores the keypoint-coordinate
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx] = coordinate[0]
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx + 1] = coordinate[1]

    return keypoints

def compute_PCK(gt_featuremaps: torch.Tensor, pred_featuremaps: torch.Tensor):
    """
    Computes the Percentage of Correct Keypoints (PCK) between the groundtruth heatmaps and predicted heatmaps.

    Parameters
    ----------
    gt_featuremaps : torch.Tensor
        Tensor of groundtruth heatmaps
        
    pred_featuremaps : torch.Tensor
        Tensor of predicted heatmaps

    Returns
    -------
    ratio : float
        Ratio of correctly predicted joints.
    """

    # Number of dimensions
    num_dimensions = 2
    
    # Casting featuremaps to numpy
    if len(gt_featuremaps.shape) == 5:
        gt_kps = heatmaps2coordinates(gt_featuremaps)
        pred_kps = heatmaps2coordinates(pred_featuremaps)
        
        torso_diameter = get_torso_diameter(gt_kps)
        
        gt_kps = np.array(gt_kps).reshape(-1, num_dimensions)
        pred_kps = np.array(pred_kps).reshape(-1, num_dimensions)
    else:
        gt_kps = gt_featuremaps.detach().cpu()
        pred_kps = pred_featuremaps.detach().cpu()
        
        torso_diameter = get_torso_diameter(gt_kps)
        
        gt_kps = np.array(gt_kps).reshape(-1, num_dimensions)
        pred_kps = np.array(pred_kps).reshape(-1, num_dimensions)
    
    # Removing unnannotated keypoints
    pred_kps = pred_kps[gt_kps.sum(axis=1) != 0]
    torso_diameter = torso_diameter[gt_kps.sum(axis=1) != 0]
    gt_kps = gt_kps[gt_kps.sum(axis=1) != 0]
    
    # Distance between ground truth keypoints and predictions
    dist = np.linalg.norm(gt_kps - pred_kps, axis=1).reshape((-1, 1))
    
    # Threshold
    threshold = torso_diameter * 0.2
    
    # Num correct
    num_correct = np.sum(dist < threshold)
    
    # Percentage of correctly estimated keypoints
    pck = num_correct/gt_kps.shape[0] if gt_kps.shape[0] != 0 else -1

    return pck 

def modify_grad(x: torch.Tensor, is_pa: List[bool], model_type: Union[Baseline, Unipose, DeciWatch]):
    """
    Function for cancelling out gradient.
    
    Parameters
    ----------
    x : torch.Tensor
        Tensor which the gradient should be removed
        
    is_pa : List[bool]
        Indicies that decides whether a batch from the Penn-Action dataset.
        
    model_type : Union[Baseline, Unipose, DeciWatch]
        Type of the model
        
    Returns
    -------
    x : torch.Tensor
        Version of x where gradients have been cancelled out.
        Used to decide whether to indec x as heatmaps or coordinates.
        
    Note
    ----
    Inspired by the following code-snippet
    https://discuss.pytorch.org/t/how-to-detach-a-rows-of-a-tensor/21058/4
    """
    
    if model_type == Baseline or model_type == Unipose:
        x[:, :, GENERAL_MISSING_INDICIES] = 0
        x[np.ix_(is_pa, np.arange(x.shape[1]), PA_MISSING_INDICIES)] = 0
    elif model_type == DeciWatch:
        general_inds_1 = [x * 2 for x in GENERAL_MISSING_INDICIES]
        general_inds_2 = [x * 2 + 1 for x in GENERAL_MISSING_INDICIES]
        pa_inds_1 = [x * 2 for x in PA_MISSING_INDICIES]
        pa_inds_2 = [x * 2 + 1 for x in PA_MISSING_INDICIES]
        
        x[:, :, general_inds_1] = 0
        x[:, :, general_inds_2] = 0
        
        
        x[np.ix_(is_pa, np.arange(x.shape[1]), pa_inds_1)] = 0
        x[np.ix_(is_pa, np.arange(x.shape[1]), pa_inds_2)] = 0
    else:
        raise TypeError(f"Incorrect model_type. You passed {model_type}")
        
    return x